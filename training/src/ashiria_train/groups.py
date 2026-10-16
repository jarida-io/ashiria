"""`ashiria sign-groups`: every dictionary sign sorted by kind (fingerspelling, number, word, phrase) and into
look-alike groups, with what tells each look-alike pair apart (handshape, palm, location, movement, the other hand).

Look-alikes come from the latest recogniser run: signs whose references it places close together, and signs its
simulated attempts mistook for each other (signs.json). Training uses them to put look-alikes in the same batch,
so the model has to learn their difference (embed.py); the report shows which differences it finds hardest.
Writes data/dictionary/manifests/sign_groups.json (local, like everything derived from the videos).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config, load_config, path

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"
KINDS = ("fingerspelling", "number", "word", "phrase")
_KIND_OF = {"Alphabet": "fingerspelling", "Numbers": "number", "Phrases": "phrase"}
TOPICS = ("Medical", "Religious")
LOOKALIKE_SIM = 0.75   # references at least this similar (cosine of embeddings) count as look-alikes
NEIGHBOURS = 8
GROUP_DISTANCE = 0.35  # average-linkage cosine distance that closes a look-alike group
OTHER_HAND_SHARE = 0.3 # one of a pair uses its second hand and the other doesn't, for this share of the sign

# Feature columns per hand (features.hand_features), compared part by part.
PARTS = {"handshape": slice(60, 84), "palm": slice(84, 90), "location": slice(90, 93), "movement": slice(93, 96)}
HAND = 97
OTHER_PRESENT = HAND + 96


def groups_file(cfg: Config | None = None) -> Path:
    return path(load_config(DICTIONARY_CONFIG), "manifests") / "sign_groups.json"


def load_groups() -> dict | None:
    f = groups_file()
    return json.loads(f.read_text()) if f.exists() else None


def sign_kinds(ids: list[str]) -> dict[str, dict]:
    """Kind and topic of each clip from the data sheet's categories (a clip several entries share takes the most
    specific: fingerspelling, then number, then phrase, then word)."""
    from .study import dictionary_index

    order = {k: i for i, k in enumerate(("fingerspelling", "number", "phrase", "word"))}
    out: dict[str, dict] = {}
    for e in dictionary_index():
        clip = e.get("clip_id")
        if not clip:
            continue
        kind = _KIND_OF.get(e["category"], "word")
        cur = out.setdefault(clip, {"kind": kind, "topic": None, "names": []})
        if order[kind] < order[cur["kind"]]:
            cur["kind"] = kind
        if e["category"] in TOPICS:
            cur["topic"] = e["category"]
        cur["names"].append(e["sign"])
    return {c: out.get(c, {"kind": "word", "topic": None, "names": []}) for c in ids}


def part_distances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """How far two reference feature sequences [T, FEATURE_DIM] are apart in each part, on the signing hand (time
    aligned by DTW on the whole features, so a sign done faster still lines up)."""
    from .matching import dtw_path

    _, path_ = dtw_path(a.astype(np.float64), b.astype(np.float64), band=12)
    ia, ib = np.array([i for i, _ in path_]), np.array([j for _, j in path_])
    out = {name: float(np.linalg.norm(a[ia, s] - b[ib, s], axis=1).mean()) for name, s in PARTS.items()}
    out["other hand"] = float(np.abs(a[ia, OTHER_PRESENT] - b[ib, OTHER_PRESENT]).mean())
    return out


def build_groups(cfg: Config, run: str | Path | None = None, sample_pairs: int = 400, seed: int = 3) -> Path:
    from sklearn.cluster import AgglomerativeClustering

    from .embed import _load_model, embed_all, latest_run, load_references, recogniser_params
    from .features import sequence_features
    from .params import EngineParams
    from .train import pick_device

    run = Path(run) if run else latest_run(cfg)
    if run is None:
        raise SystemExit("No embed run yet: run `ashiria embed` first.")
    t0 = time.time()
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    model = _load_model(cfg, run, feats[0].shape[-1], pick_device(str(cfg.train.device)))
    emb = embed_all(model, feats, pick_device(str(cfg.train.device)))
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    kinds = sign_kinds(ids)
    index = {c: i for i, c in enumerate(ids)}

    # Look-alikes: close references, plus the signs simulated attempts were mistaken for.
    near = np.argsort(-sim, axis=1)[:, :NEIGHBOURS]
    confused: dict[int, set[int]] = {}
    signs_json = run / "signs.json"
    if signs_json.exists():
        for clip, r in json.loads(signs_json.read_text())["results"].items():
            for test in ("normal", "harder", "viewpoint"):
                for other in (r.get(test) or {}).get("mistaken_for", []):
                    if clip in index and other in index:
                        confused.setdefault(index[clip], set()).add(index[other])

    # Typical distance in each part between two random signs, so differences can be compared across parts.
    rng = np.random.default_rng(seed)
    pairs = rng.choice(len(refs), (sample_pairs, 2))
    typical = {k: float(np.median(v)) for k, v in _columns(
        [part_distances(feats[i], feats[j]) for i, j in pairs if i != j]).items() if k in PARTS}

    def difference(i: int, j: int) -> dict:
        """The part that differs most for its size, or the other hand when only one of the pair uses it (two random
        signs nearly always agree on hands, so a ratio to their typical difference would mean nothing)."""
        d = part_distances(feats[i], feats[j])
        rel = {k: d[k] / max(typical[k], 1e-6) for k in PARTS}
        main = "other hand" if d["other hand"] >= OTHER_HAND_SHARE else max(rel, key=rel.get)
        return {"main": main, "relative": {k: round(v, 2) for k, v in rel.items()},
                "other_hand_frames": round(d["other hand"], 2)}

    groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=GROUP_DISTANCE).fit_predict(emb)
    out_signs = {}
    for i, clip in enumerate(ids):
        alike = [int(j) for j in near[i] if sim[i, j] >= LOOKALIKE_SIM] + sorted(confused.get(i, set()) - set(near[i].tolist()))
        out_signs[clip] = {
            "label": labels[i], **kinds[clip], "group": int(groups[i]),
            "lookalikes": [{"clip": ids[j], "label": labels[j], "similarity": round(float(sim[i, j]), 3),
                            "confused": j in confused.get(i, set()), **difference(i, j)} for j in alike],
        }
        if i % 500 == 0:
            print(f"  {i} / {len(ids)} signs ({time.time() - t0:.0f} s)", flush=True)

    size = Counter(groups.tolist())
    report = _report(out_signs, size)
    out = {"run": run.name, "created": time.strftime("%Y-%m-%d %H:%M"), "lookalike_similarity": LOOKALIKE_SIM,
           "group_distance": GROUP_DISTANCE, "typical_part_distance": typical, "report": report, "signs": out_signs}
    f = groups_file()
    f.write_text(json.dumps(out))
    _print(report)
    print(f"-> {f} ({time.time() - t0:.0f} s)")
    return f


def _columns(rows: list[dict]) -> dict[str, list[float]]:
    return {k: [r[k] for r in rows] for k in rows[0]}


def _report(signs: dict, size: Counter) -> dict:
    by_kind = {}
    for kind in KINDS:
        members = [s for s in signs.values() if s["kind"] == kind]
        pairs = [l for s in members for l in s["lookalikes"]]
        confused = [l for l in pairs if l["confused"]]
        by_kind[kind] = {
            "signs": len(members),
            "with_lookalikes": sum(bool(s["lookalikes"]) for s in members),
            "lookalike_pairs": len(pairs),
            "confused_pairs": len(confused),
            "same_kind_share": round(sum(signs[l["clip"]]["kind"] == kind for l in pairs) / max(len(pairs), 1), 3),
            "main_difference": dict(Counter(l["main"] for l in pairs).most_common()),
            "main_difference_when_confused": dict(Counter(l["main"] for l in confused).most_common()),
        }
    sizes = sorted(size.values(), reverse=True)
    return {"by_kind": by_kind, "groups": len(sizes), "groups_of_2_or_more": sum(s >= 2 for s in sizes),
            "largest_groups": sizes[:10]}


def _print(r: dict) -> None:
    print(f"{r['groups']} look-alike groups ({r['groups_of_2_or_more']} with 2 or more signs; largest {r['largest_groups'][:5]})")
    for kind, k in r["by_kind"].items():
        diff = ", ".join(f"{d} {n}" for d, n in list(k["main_difference_when_confused"].items())[:4]) or "none"
        print(f"  {kind:14s} {k['signs']:5d} signs, {k['with_lookalikes']:4d} with look-alikes "
              f"({k['same_kind_share']:.0%} of them the same kind); confused pairs {k['confused_pairs']:4d}, "
              f"told apart mainly by: {diff}")

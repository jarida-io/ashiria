"""Sentences from single signs: several dictionary signs joined the way a signer runs them together (the hands go
from one sign straight into the next instead of dropping to rest), for training and testing the reading of signs
in a row. Each sign keeps its core (hands raised: canonical.active_window); a short transition moves the hands
from the end of one sign to the start of the next. Body frame, like every segment.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .canonical import Segment, from_mediapipe, gap_fill, resample, trim
from .config import load_config, path

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"
FPS = 30.0
TRANSITION_S = (0.15, 0.35)  # a signer's move from one sign to the next


@lru_cache(maxsize=4096)
def raw_sign(clip_id: str) -> Segment | None:
    """A dictionary clip as recorded (gap-filled, trimmed to the hands being up), at FPS."""
    dcfg = load_config(DICTIONARY_CONFIG)
    with np.load(path(dcfg, "landmarks") / f"{clip_id}.npz") as d:
        seg, _ = from_mediapipe(d)
    if seg is None:
        return None
    f = dcfg.features
    seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
    if seg is None or len(seg) < 4:
        return None
    return resample(seg, max(4, int(round((len(seg) - 1) / seg.fps * FPS)) + 1))


def core_bounds(seg: Segment, fraction: float = 0.5) -> tuple[int, int]:
    """Where the sign proper starts and ends in a clip: the frames where the hand that lifts more is above
    `fraction` of its height range (mid-sentence a signer's hands stay up, so more of the rise goes than in
    canonical.active_window)."""
    ranges = []
    for s in range(2):
        ok = seg.present[:, s] > 0.5
        y = seg.wrist_pos[:, s, 1]
        ranges.append((float(np.ptp(y[ok])) if ok.sum() >= 3 else 0.0, s))
    span, slot = max(ranges)
    if span < 0.3:
        return 0, len(seg)
    ok = seg.present[:, slot] > 0.5
    y = seg.wrist_pos[:, slot, 1]
    idx = np.flatnonzero(ok & (y >= y[ok].min() + fraction * span))
    return (int(idx[0]), int(idx[-1]) + 1) if idx[-1] - idx[0] >= 5 else (0, len(seg))


def _join(a: Segment, b: Segment, frames: int) -> Segment:
    """Frames moving the hands from a's last pose to b's first (a hand missing at either end stays missing)."""
    w = (np.arange(1, frames + 1) / (frames + 1))[:, None]
    hand = a.hand_rel[-1] * (1 - w[..., None, None]) + b.hand_rel[0] * w[..., None, None]
    wrist = a.wrist_pos[-1] * (1 - w[..., None]) + b.wrist_pos[0] * w[..., None]
    present = np.repeat((a.present[-1] * b.present[0])[None], frames, 0)
    return Segment(hand, wrist, present, FPS)


def _concat(parts: list[Segment]) -> Segment:
    return Segment(np.concatenate([p.hand_rel for p in parts]), np.concatenate([p.wrist_pos for p in parts]),
                   np.concatenate([p.present for p in parts]), FPS)


def sentence(clips: list[str], rng: np.random.Generator) -> tuple[Segment, list[tuple[int, int]]] | None:
    """The clips signed in a row: the first sign's rise, each sign's core joined by transitions, the last sign's
    drop. Returns the segment and each sign's (start, end) frames, or None if a clip has no usable hands."""
    signs = [raw_sign(c) for c in clips]
    if any(s is None for s in signs):
        return None
    parts, spans, at = [], [], 0
    for k, s in enumerate(signs):
        a, b = core_bounds(s)
        if k == 0:
            a = 0  # keep the rise into the first sign
        if k == len(signs) - 1:
            b = len(s)  # and the drop after the last
        piece = s.slice(a, b)
        if parts:
            gap = _join(parts[-1], piece, int(round(rng.uniform(*TRANSITION_S) * FPS)))
            parts.append(gap)
            at += len(gap)
        spans.append((at, at + len(piece)))
        parts.append(piece)
        at += len(piece)
    return _concat(parts), spans


WINDOW_S = (0.5, 0.75, 1.0, 1.35, 1.8, 2.4)  # sign lengths to look for
STRIDE = 3                                   # frames between window starts (0.1 s)


def spot(model, ref_emb: np.ndarray, seg: Segment, frames: int, device) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Every window of the stream (WINDOW_S long, every STRIDE frames) inside its signing (the hands' rise from rest
    and drop back look like signs that move down, so they are left out), and its similarity to every reference."""
    from .embed import embed_all
    from .features import sequence_features

    lo, hi = core_bounds(seg, fraction=0.35)
    wins = [(a, a + int(L * FPS)) for L in WINDOW_S for a in range(lo, hi - int(L * FPS) + 1, STRIDE)]
    if not wins:
        wins = [(lo, hi)]
    feats = []
    for a, b in wins:
        w = resample(seg.slice(a, b), frames)
        feats.append(sequence_features(w.hand_rel, w.wrist_pos, w.present, 1.0 / w.fps))
    return wins, embed_all(model, feats, device) @ ref_emb.T


def decode(wins: list[tuple[int, int]], sims: np.ndarray, length: int, floor: float, per_sign: float = 0.15,
           names: list[str] | None = None, phrases: set[tuple[str, ...]] | None = None, bonus: float = 0.0) -> list[int]:
    """The best run of non-overlapping windows. A window counts only where its sign matches better than in any
    overlapping window (so one sign isn't read three times from shifted windows); each adds its similarity above
    `floor`, less `per_sign` (so reading more signs has to be earned). A reading whose last signs (by `names`) make
    one of `phrases` gains `bonus`, so a likely sentence can beat a slightly closer but unlikely one."""
    longest = max((len(p) for p in phrases or ()), default=0)
    top = sims.argmax(1)
    best_sim = sims[np.arange(len(wins)), top]
    starts, ends = np.array([a for a, _ in wins]), np.array([b for _, b in wins])
    keep = []
    for w, (a, b) in enumerate(wins):
        overlap = (starts < b) & (ends > a)
        same = overlap & (top == top[w])
        if best_sim[w] - floor > per_sign and best_sim[w] >= best_sim[same].max():
            keep.append(w)
    ending: dict[int, list[int]] = {}
    for w in keep:
        ending.setdefault(wins[w][1], []).append(w)
    best = [(0.0, ())] * (length + 1)  # best (score, reading) up to each frame
    for t in range(1, length + 1):
        best[t] = best[t - 1]
        for w in ending.get(t, []):
            score, reading = best[wins[w][0]]
            reading = reading + (int(top[w]),)
            score += best_sim[w] - floor - per_sign
            if bonus and names and any(tuple(names[r] for r in reading[-n:]) in phrases
                                       for n in range(2, min(longest, len(reading)) + 1)):
                score += bonus
            if score > best[t][0]:
                best[t] = (score, reading)
    return list(best[length][1])


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def evaluate_sentences(cfg, run=None, seed: int = 1, count: int = 60) -> dict:
    """How well signs in a row are read, by the run's model: random runs of signs and the phrases (both joined from
    single recordings), and the dictionary's own recordings of whole phrases (real signing), with and without the
    phrase bonus. Sign error rate: wrong, missed and extra signs per sign signed."""
    import json
    import re
    import time
    from pathlib import Path

    from .embed import _load_model, embed_all, latest_run, load_references, recogniser_params
    from .features import sequence_features
    from .params import EngineParams
    from .phrases import load_phrases, phrase_clips, sign_clips
    from .train import pick_device

    run = Path(run) if run else latest_run(cfg)
    t0 = time.time()
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    dev = pick_device(str(cfg.train.device))
    feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    model = _load_model(cfg, run, feats[0].shape[-1], dev)
    ref_emb = embed_all(model, feats, dev)
    index = {c: i for i, c in enumerate(ids)}
    single, recorded = sign_clips()
    from .groups import sign_kinds
    # Read sign by sign: a recorded phrase would otherwise be read whole, as itself (also right, but not this test).
    whole = np.array([k["kind"] == "phrase" for k in sign_kinds(ids).values()])
    name_of = {c: name for name, clips in single.items() for c in clips}
    names = [name_of.get(c, re.sub(r"\s+\d+$", "", labels[i].lower())) for i, c in enumerate(ids)]
    phrase_names = {tuple(t.lower() for t in ph["gloss"]) for ph in load_phrases()}
    rng = np.random.default_rng(seed)

    def cases_from(runs):
        out = []
        for r in runs:
            built = sentence([ids[i] for i in r], rng)
            if built is not None:
                wins, sims = spot(model, ref_emb, built[0], p.seq_len, dev)
                sims[:, whole] = -1.0
                out.append(([names[i] for i in r], wins, sims, len(built[0])))
        return out

    words = np.flatnonzero(~whole)
    random_runs = [[int(i) for i in rng.choice(words, int(rng.integers(2, 5)), replace=False)] for _ in range(count)]
    phrase_runs = []
    for ph in load_phrases():
        pc = phrase_clips(ph)
        if pc:
            for _ in range(2):
                phrase_runs.append([index[c[int(rng.integers(len(c)))]] for c in pc if c] if all(any(x in index for x in c) for c in pc) else None)
    phrase_runs = [[i for i in r] for r in phrase_runs if r]
    real = []
    for ph in load_phrases():
        for clip in recorded.get(ph.get("recorded") or "", []):
            from .sentences import raw_sign
            seg = raw_sign(clip)
            if seg is not None:
                wins, sims = spot(model, ref_emb, seg, p.seq_len, dev)
                sims[:, whole] = -1.0
                real.append(([t.lower() for t in ph["gloss"]], wins, sims, len(seg)))
    sets = {"random signs": cases_from(random_runs), "phrases (joined)": cases_from(phrase_runs),
            "phrases (recorded)": real}

    results = {}
    for floor in (0.7, 0.75, 0.8):
        for per in (0.15, 0.2, 0.25):
            for bonus in (0.0, 0.3):
                key = f"floor {floor} per-sign {per} bonus {bonus}"
                row = {}
                for name, cases in sets.items():
                    errs = n = exact = found = 0
                    for truth, wins, sims, L in cases:
                        got = [names[i] for i in decode(wins, sims, L, floor, per, names, phrase_names, bonus)]
                        errs += edit_distance(got, truth)
                        n += len(truth)
                        exact += got == truth
                        found += len(set(got) & set(truth))
                    row[name] = {"cases": len(cases), "sign_error": round(errs / max(n, 1), 3),
                                 "exact": round(exact / max(len(cases), 1), 3), "signs_found": round(found / max(n, 1), 3)}
                results[key] = row
    best = min(results, key=lambda k: results[k]["phrases (recorded)"]["sign_error"] + results[k]["random signs"]["sign_error"])
    for k in [best] + [k for k in results if k.endswith("bonus 0.0") and k.startswith(best.split(" bonus")[0])]:
        print(k)
        for name, r in results[k].items():
            print(f"  {name:20s} {r['cases']:3d} cases: sign error {r['sign_error']:.0%}, exactly right {r['exact']:.0%}, "
                  f"signs found {r['signs_found']:.0%}")
    out = {"run": run.name, "best": best, "results": results}
    (run / "sentences.json").write_text(json.dumps(out, indent=1))
    print(f"-> {run / 'sentences.json'} ({time.time() - t0:.0f} s)")
    return out

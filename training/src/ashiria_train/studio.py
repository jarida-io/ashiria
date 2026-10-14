"""Studio takes (spec/STUDIO_FORMAT.md): import headset recordings as reference signs, and fake them for tests.

`ashiria import-studio DIR` validates every take, rebuilds canonical frames from the raw poses
(quest.py), cuts each take like a live attempt, drops outlier takes, picks the medoid as the shadow
take, and writes exports/studio/store.json (references) and attempts.json (test and non-sign takes).
`ashiria studio-fake` writes takes from the dictionary references, so the format, importer and
quest.py can be tested end to end before the headset arrives.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import quest
from .canonical import Segment, gap_fill, steadiest_frame, trim
from .config import Config, path, training_relative
from .features import SPEC_VERSION
from .matching import dtw, segment_descriptor
from .params import RUNTIME_VERSION, EngineParams
from .pipeline import cut_reference
from .references import tag

FORMAT = "ashiria.studio-take"
FORMAT_VERSION = 1
SIDES = ("right", "left")
MIN_PRESENT, MIN_SECONDS, MAX_SECONDS = 0.8, 0.3, 5.0
PARITY_TOL = 1e-3


class TakeError(ValueError):
    pass


def _arr(values, shape) -> np.ndarray:
    a = np.asarray(values, np.float64)
    if a.size != int(np.prod(shape)):
        raise TakeError(f"expected {int(np.prod(shape))} values for shape {shape}, got {a.size}")
    return a.reshape(shape)


def load_take(file: Path) -> dict:
    take = json.loads(Path(file).read_text())
    if take.get("format") != FORMAT:
        raise TakeError("not an Ashiria Studio take")
    if take.get("format_version") != FORMAT_VERSION:
        raise TakeError(f"format_version {take.get('format_version')} (this importer reads {FORMAT_VERSION})")
    if take.get("joint_layout") != "openxr26":
        raise TakeError(f"joint layout {take.get('joint_layout')!r}")
    for key in ("take_id", "sign_id", "signer", "handedness", "tick_hz", "frames"):
        if key not in take:
            raise TakeError(f"missing {key}")
    if take["handedness"] not in SIDES:
        raise TakeError("handedness must be right or left")
    n = int(take["frames"]["n"])
    f = take["frames"]
    take["_head_pos"], take["_head_rot"] = _arr(f["head_pos"], (n, 3)), _arr(f["head_rot"], (n, 4))
    for side in SIDES:
        h = f[side]
        take[f"_{side}"] = {"pos": _arr(h["pos"], (n, 26, 3)),
                            **{k: _arr(h[k], (n,)) > 0.5 for k in ("tracked", "high_conf", "inferred", "system_gesture")}}
    c = f.get("canonical")
    take["_canonical"] = None if not c else Segment(_arr(c["hand_rel"], (n, 2, 21, 3)), _arr(c["wrist_pos"], (n, 2, 3)),
                                                    _arr(c["present"], (n, 2)), float(take["tick_hz"]))
    return take


def canonical_from_raw(take: dict) -> Segment:
    n = int(take["frames"]["n"])
    hand_rel, wrist_pos, present = np.zeros((n, 2, 21, 3)), np.zeros((n, 2, 3)), np.zeros((n, 2))
    axes = None
    for t in range(n):
        hands = {side: {"joints": take[f"_{side}"]["pos"][t], "tracked": take[f"_{side}"]["tracked"][t],
                        "high_confidence": take[f"_{side}"]["high_conf"][t], "inferred": take[f"_{side}"]["inferred"][t],
                        "system_gesture": take[f"_{side}"]["system_gesture"][t]} for side in SIDES}
        hand_rel[t], wrist_pos[t], present[t], axes = quest.frame(
            take["_head_pos"][t], quest.rotate(take["_head_rot"][t], [0.0, 0.0, 1.0]), hands, take["handedness"] == "left",
            float(take.get("shoulder_width_m", quest.DEFAULT_SHOULDER_WIDTH_M)), axes)
    return Segment(hand_rel, wrist_pos, present, float(take["tick_hz"]))


def parity_error(a: Segment, b: Segment) -> float:
    if a.present.shape != b.present.shape:
        return float("inf")
    both = (a.present > 0.5) & (b.present > 0.5)
    if (a.present > 0.5).tolist() != (b.present > 0.5).tolist():
        return float("inf")
    if not both.any():
        return 0.0
    return float(max(np.abs(a.hand_rel[both] - b.hand_rel[both]).max(), np.abs(a.wrist_pos[both] - b.wrist_pos[both]).max()))


def prepare_take(take: dict, p: EngineParams) -> tuple[Segment, Segment]:
    """(the take trimmed to signing space, for tags; the reference cut like a live attempt).
    Raises TakeError on quality problems."""
    seg = canonical_from_raw(take)
    if take["_canonical"] is not None:
        err = parity_error(seg, take["_canonical"])
        if err > PARITY_TOL:
            print(f"  warning: {take['take_id']}: app's canonical frames differ from Python's by {err:.2g}")
    kept = take.get("trim") or {}
    seg = seg.slice(int(kept.get("start", 0)), int(kept.get("end", len(seg))))
    seconds = len(seg) / seg.fps
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise TakeError(f"{seconds:.2f} s long (keep {MIN_SECONDS}–{MAX_SECONDS} s)")
    if seg.present[:, 0].mean() < MIN_PRESENT:
        raise TakeError(f"dominant hand present in {seg.present[:, 0].mean():.0%} of frames (need {MIN_PRESENT:.0%})")
    rest = float(take.get("rest_height", p.segmenter.rest_height))
    ref = cut_reference(seg, p, rest)
    signing = trim(gap_fill(seg, p.gap_fill_frames), rest, p.trim_margin_frames)
    if ref is None or signing is None:
        raise TakeError("no hand in signing space")
    return signing, ref


def _segment_json(seg: Segment) -> dict:
    return {"fps": float(seg.fps), "hand_rel": np.round(seg.hand_rel, 5).tolist(),
            "wrist_pos": np.round(seg.wrist_pos, 4).tolist(), "present": seg.present.tolist()}


def import_studio(cfg: Config, folder: str | Path, keep_outliers: bool = False, pack_id: str = "ksl-core") -> Path:
    from .catalogue import catalogue_file, load_catalogue

    p = EngineParams.from_config(cfg)
    known = {s["id"] for s in load_catalogue(catalogue_file(cfg, pack_id))["signs"]} | {"_none"}
    files = sorted(training_relative(folder).rglob("*.json"))
    if not files:
        raise SystemExit(f"No take files under {folder}.")
    refs: dict[str, list] = {}
    attempts, rejected = [], []
    for file in files:
        try:
            take = load_take(file)
            if take.get("status", "keep") != "keep":
                continue
            if take["sign_id"] not in known:
                raise TakeError(f"unknown sign id {take['sign_id']!r}")
            raw, ref = prepare_take(take, p)
        except (TakeError, KeyError, json.JSONDecodeError) as exc:
            rejected.append(f"{file.name}: {exc}")
            continue
        item = {"take_id": take["take_id"], "signer": take["signer"], "sign_id": take["sign_id"], "raw": raw, "ref": ref}
        if take.get("role", "reference") == "reference" and take["sign_id"] != "_none":
            refs.setdefault(take["sign_id"], []).append(item)
        else:
            attempts.append(item)

    store = {"format": "ashiria.studio-store", "spec_version": SPEC_VERSION, "runtime_version": RUNTIME_VERSION,
             "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "signs": {}}
    for sid, items in sorted(refs.items()):
        desc = [segment_descriptor(i["ref"], p.weights) for i in items]
        n = len(items)
        d = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                d[a, b] = d[b, a] = dtw(desc[a], desc[b], p.dtw_band)
        keep = list(range(n))
        excluded = []
        if n >= 3 and not keep_outliers:
            mean = d.sum(axis=1) / (n - 1)
            keep = [k for k in range(n) if mean[k] <= 2 * np.median(mean)]
            excluded = [items[k]["take_id"] for k in range(n) if k not in keep]
        medoid = keep[int(np.argmin(d[np.ix_(keep, keep)].sum(axis=1)))]
        kept = [items[k] for k in keep]
        store["signs"][sid] = {
            "takes": [{"take_id": i["take_id"], "signer": i["signer"], "key_frame": steadiest_frame(i["ref"]),
                       **_segment_json(i["ref"])} for i in kept],
            "shadow_take": keep.index(medoid), "tags": tag(items[medoid]["raw"]), "excluded": excluded}

    out = path(cfg, "exports") / "studio"
    out.mkdir(parents=True, exist_ok=True)
    (out / "store.json").write_text(json.dumps(store))
    (out / "attempts.json").write_text(json.dumps({"attempts": [
        {"take_id": i["take_id"], "signer": i["signer"], "sign_id": i["sign_id"], **_segment_json(i["ref"])}
        for i in attempts]}))
    takes = sum(len(s["takes"]) for s in store["signs"].values())
    print(f"{takes} reference takes for {len(store['signs'])} signs, {len(attempts)} test/non-sign takes -> {out}")
    excluded = [(sid, s["excluded"]) for sid, s in store["signs"].items() if s["excluded"]]
    if excluded:
        print("Outlier takes left out (use --keep-outliers to keep): " + "; ".join(f"{s}: {', '.join(x)}" for s, x in excluded))
    if rejected:
        print(f"Rejected {len(rejected)} files:\n  " + "\n  ".join(rejected))
    return out


# --- fake takes for tests ---------------------------------------------------------------------------------

def _quat(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unity rotation (x, y, z, w): yaw about +y, then pitch about the local +x (positive looks down)."""
    y, p = np.radians(yaw_deg) / 2, np.radians(pitch_deg) / 2
    qy = np.array([0.0, np.sin(y), 0.0, np.cos(y)])
    qx = np.array([np.sin(p), 0.0, 0.0, np.cos(p)])
    x1, y1, z1, w1 = qy
    x2, y2, z2, w2 = qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def fake_take(seg: Segment, sign_id: str, rng: np.random.Generator, signer: str = "S99", handedness: str = "right",
              role: str = "reference", shoulder_width: float = 0.38, rest_height: float = -1.8) -> dict:
    """A Studio take whose raw poses canonicalise back to `seg` (a canonical stream at the tick rate)."""
    n = len(seg)
    head = np.array([rng.uniform(-1, 1), rng.uniform(1.1, 1.7), rng.uniform(-1, 1)])
    yaw, pitch = rng.uniform(-180, 180), rng.uniform(5, 25)
    q = _quat(yaw, pitch)
    R, _ = quest.head_axes(quest.rotate(q, [0, 0, 1]))
    mirror = handedness == "left"
    sides = {"right": None, "left": None}
    frames = {"n": n, "t": (np.arange(n) / seg.fps).round(4).tolist(),
              "head_pos": np.tile(head, n).round(6).tolist(), "head_rot": np.tile(q, n).round(7).tolist()}
    for side in sides:
        slot = quest.slot_for(side == "left", mirror)
        pos = np.stack([quest.to_world(seg.hand_rel[t, slot], seg.wrist_pos[t, slot], head, R, shoulder_width, mirror)
                        for t in range(n)])
        on = seg.present[:, slot] > 0.5
        frames[side] = {"tracked": on.astype(int).tolist(), "high_conf": on.astype(int).tolist(),
                        "inferred": [0] * n, "system_gesture": [0] * n, "pos": pos.round(6).ravel().tolist(),
                        "rot": np.tile([0.0, 0.0, 0.0, 1.0], n * 26).tolist()}
    frames["canonical"] = {"hand_rel": seg.hand_rel.round(6).ravel().tolist(), "wrist_pos": seg.wrist_pos.round(6).ravel().tolist(),
                           "present": seg.present.ravel().tolist()}
    return {"format": FORMAT, "format_version": FORMAT_VERSION, "take_id": str(uuid.UUID(int=int(rng.integers(2**63)))),
            "session_id": "fake", "sign_id": sign_id, "pack_id": "ksl-core", "pack_version": "0.0.0", "signer": signer,
            "role": role, "handedness": handedness, "consent_ref": "fake", "shoulder_width_m": shoulder_width,
            "shoulder_calibrated": False, "rest_height": rest_height, "device": "fake", "app_version": "0.0.0",
            "spec_version": SPEC_VERSION, "runtime_version": RUNTIME_VERSION, "joint_layout": "openxr26",
            "tick_hz": float(seg.fps), "recorded_at": "2026-10-14T00:00:00Z", "trim": {"start": 0, "end": n},
            "status": "keep", "frames": frames}


def write_fake_takes(cfg: Config, out_dir: str | Path, signs: list[str] | None = None, takes: int = 3, seed: int = 3) -> Path:
    """Fake Studio takes from the dictionary references: each a simulated attempt by "someone else"."""
    from .calibrate import simulated_stream
    from .store import DICTIONARY, load_dictionary

    rng = np.random.default_rng(seed)
    dictionary = load_dictionary(DICTIONARY)
    if not dictionary:
        raise SystemExit("No dictionary references: run `ashiria references --config configs/ksl_dictionary.yaml` first.")
    from .catalogue import catalogue_file, load_catalogue
    by_word = {e["sources"].get("dictionary"): e["id"] for e in load_catalogue(catalogue_file(cfg, "ksl-core"))["signs"]}
    words = [w for w in dictionary if by_word.get(w) and (not signs or by_word[w] in signs)]
    out = training_relative(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    aug = dict(cfg.calibrate.augment)
    for w in words:
        for k in range(takes):
            stream = simulated_stream(dictionary[w].takes[0].segment, rng, aug, 30.0)
            take = fake_take(stream, by_word[w], rng, handedness="left" if k == takes - 1 else "right")
            (out / f"{by_word[w]}__{k + 1}.json").write_text(json.dumps(take))
    print(f"{len(words) * takes} fake takes -> {out}")
    return out

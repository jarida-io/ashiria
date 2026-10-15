"""Per-sign reference templates from extracted landmarks.

For each sign: the canonical segment (the app's shadow-hand animation), the dominant hand's finger
angles at its steadiest frame (targets for per-finger feedback), and tags that help pick signs the
headset can check. The tags are hints for choosing the vocabulary, not decisions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import joints as J
from .augment import camera_view
from .canonical import (Segment, active_window, from_mediapipe, gap_fill, hold_frames, left_dominant_votes, mirror,
                        steadiest_frame, trim, wrist_extent)
from .config import Config, path, training_relative
from .features import SPEC_VERSION, flexion
from .matching import distance_matrix, lookalike_table, nearest, segment_descriptor
from .params import EngineParams
from .pipeline import cut_reference

DISTRACTORS = 300  # non-vocabulary dictionary signs kept (locally) to calibrate "not sure"


def load_vocabulary(file: str | Path) -> list[str]:
    words = []
    for line in training_relative(file).read_text().splitlines():
        word = line.split("#", 1)[0].strip().lower()
        if word:
            words.append(word)
    return words


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def names_of(row: pd.Series) -> set[str]:
    aliases = _text(row.get("aliases"))
    return {str(row["label"]).lower()} | {a.strip().lower() for a in aliases.split(";") if a.strip()}


def choose_clip(word: str, names: dict[str, set[str]], labels: dict[str, str]) -> str | None:
    """The dictionary clip for a vocabulary word. Prefer the clip *labelled* with the word, then one listing it as an
    alias (other signs share glosses: "thank you" is also an alias of "gratitude"), then the lowest-numbered
    variant ("i" -> "i 1", not "i 2", which is "me"). Ties keep manifest order."""
    exact = [c for c, label in labels.items() if label == word]
    if exact:
        return exact[0]
    alias = [c for c, n in names.items() if word in n]
    if alias:
        return alias[0]
    variant = re.compile(re.escape(word) + r" (\d+)")
    numbered = []
    for c, label in labels.items():
        m = variant.fullmatch(label) or next((variant.fullmatch(n) for n in names[c] if variant.fullmatch(n)), None)
        if m:
            numbered.append((int(m.group(1)), variant.fullmatch(label) is None, c))
    return min(numbered)[2] if numbered else None


HOLD_SHARE = 0.5          # static if at least this share of the active window is a hold
OTHER_HAND_EXTENT = 0.2   # shoulder widths: the other hand counts as signing if it moves this much...
SHAPE_CHANGE = 0.35       # radians: 90th-percentile bend deviation (mean over 15 joints) during the hold


def hand_lifts(seg: Segment) -> np.ndarray:
    """How far each hand rises above its own resting height (shoulder widths): its 95th percentile height minus its
    10th, over the frames that see it. The dictionary signer rests the spare hand at the stomach, visible and
    jittering, so taking part in a sign means rising well above that, not just being seen."""
    out = np.zeros(2)
    for s in (J.RIGHT, J.LEFT):
        y = seg.wrist_pos[seg.present[:, s] > 0.5, s, 1]
        if y.size >= 3:
            out[s] = np.percentile(y, 95) - np.percentile(y, 10)
    return out


def two_handed(seg: Segment) -> bool:
    """Both hands sign: the other hand lifts at least 0.4 shoulder widths, and nearly half as much as the right."""
    lift = hand_lifts(seg)
    return bool(lift[J.LEFT] > 0.4 and lift[J.LEFT] > 0.45 * lift[J.RIGHT])


def one_handed(seg: Segment) -> Segment:
    """The segment without the left hand, for a one-handed sign: its resting hand is no part of the sign."""
    present = seg.present.copy()
    hand_rel = seg.hand_rel.copy()
    present[:, J.LEFT] = 0
    hand_rel[:, J.LEFT] = 0
    return Segment(hand_rel, seg.wrist_pos, present, seg.fps)


def tag(seg: Segment) -> dict:
    """Hints computed on the active signing window (rise from and drop to rest removed)."""
    full_dom = seg.present[:, J.RIGHT] > 0.5
    rest_level = seg.wrist_pos[full_dom, J.RIGHT, 1].min() if full_dom.any() else -2.0
    win = active_window(seg)
    present = win.present > 0.5
    dom, other = present[:, J.RIGHT], present[:, J.LEFT]
    both = dom & other
    hold = hold_frames(win)
    hold_share = float(hold.sum() / max(dom.sum(), 1))
    held = win.hand_rel[hold, J.RIGHT] if hold.sum() > 1 else win.hand_rel[dom, J.RIGHT]
    bends = flexion(held) if len(held) > 1 else np.zeros((2, 15))
    shape_change = float(np.percentile(np.abs(bends - np.median(bends, axis=0)), 90, axis=0).mean())
    other_extent = wrist_extent(win, J.LEFT)
    other_y = win.wrist_pos[other, J.LEFT, 1]
    other_raised = bool(other_y.size and np.median(other_y) > rest_level + 0.5)
    gap = np.linalg.norm(win.wrist_pos[:, J.LEFT] - win.wrist_pos[:, J.RIGHT], axis=-1)
    height = win.wrist_pos[dom, J.RIGHT, 1]
    static = hold_share >= HOLD_SHARE
    return {
        "hands": "two" if other.mean() > 0.5 and two_handed(seg) else "one",
        "movement": ("handshape change" if shape_change > SHAPE_CHANGE else "static") if static else "moving",
        "near_face": bool(height.size and np.percentile(height, 90) > -0.4),
        "hands_close": bool(both.any() and np.percentile(gap[both], 10) < 0.3),
        "dominant_tracked": round(float(dom.mean()), 2),
        "hold_share": round(hold_share, 2),
        "extent": round(wrist_extent(win, J.RIGHT), 3),
        "other_extent": round(other_extent, 3),
        "shape_change": round(shape_change, 3),
        "active_frames": len(win),
    }


# Extra takes per course sign, as a webcam would see it from beside, below or above (yaw, pitch degrees; depth
# along the line of sight squashed a quarter): checks then accept those viewpoints too.
VIEW_TAKES = [(-25, 0), (25, 0), (0, -18), (0, 18)]


def _view_take(seg: Segment, key: int, yaw: float, pitch: float) -> dict:
    hand_rel, wrist_pos = camera_view(seg.hand_rel, seg.wrist_pos, yaw, pitch, depth=0.75, wrist_depth=0.9)
    hand_rel = (hand_rel - hand_rel[:, :, :1]) * seg.present[:, :, None, None]
    return {"view": [yaw, pitch], "fps": seg.fps, "hand_rel": np.round(hand_rel, 5).tolist(),
            "wrist_pos": np.round(wrist_pos, 4).tolist(), "present": seg.present.tolist(), "key_frame": key}


def build_references(cfg: Config, vocab_file: str | Path | None = None) -> Path:
    manifest = pd.read_csv(path(cfg, "manifests") / "videos.csv", dtype=str)
    manifest = manifest[manifest["label"].notna()]
    landmarks, f = path(cfg, "landmarks"), cfg.features
    p = EngineParams.from_config(cfg)

    segments: dict[str, Segment] = {}
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="references"):
        file = landmarks / f"{row['clip_id']}.npz"
        if not file.exists():
            continue
        with np.load(file) as d:
            seg, _ = from_mediapipe(d)
        if seg is not None:
            seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
        if seg is not None and len(seg) >= 4:
            segments[row["clip_id"]] = seg
    if not segments:
        raise SystemExit("No extracted clips found. Run `ashiria extract` with this config first.")

    # One source, one decision: mirror everything if the signer is left-dominant.
    left, right = left_dominant_votes(segments.values())
    print(f"Clearly one-sided clips: left hand active in {left}, right hand active in {right}: "
          + ("treating the signer as left-dominant (or the footage as mirrored) and mirroring everything."
             if left > right else "treating the signer as right-dominant."))
    if left > right:
        segments = {c: mirror(s) for c, s in segments.items()}

    out = path(cfg, "exports") / "references"
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for _, row in manifest.iterrows():
        if row["clip_id"] in segments:
            table.append({"clip_id": row["clip_id"], "label": row["label"], "aliases": _text(row.get("aliases")),
                          "category": _text(row.get("category")), "frames": len(segments[row["clip_id"]]),
                          **tag(segments[row["clip_id"]])})
    tags = pd.DataFrame(table)
    tags.to_csv(out / "references.csv", index=False)
    print(f"Tagged {len(tags)} signs -> {out / 'references.csv'}")
    print(tags[["hands", "movement"]].value_counts().to_string())

    if vocab_file is None:
        return out
    by_clip = manifest.set_index("clip_id")
    signs, missing, descriptors = [], [], []
    names = {c: names_of(by_clip.loc[c]) for c in segments}
    labels = {c: str(by_clip.loc[c]["label"]).lower() for c in segments}
    for word in load_vocabulary(vocab_file):
        clip = choose_clip(word, names, labels)
        if clip is None:
            missing.append(word)
            continue
        row = by_clip.loc[clip]
        full = segments[clip]
        lift = hand_lifts(full)
        if lift[J.LEFT] > 1.3 * lift[J.RIGHT]:
            full = mirror(full)  # this clip is signed with the left hand: teach it right-handed, like the rest
        seg = cut_reference(full, p, f.rest_height)
        if tag(full)["hands"] == "one":
            seg = one_handed(seg)  # the spare hand rests at the stomach: not part of the sign
        key = steadiest_frame(seg)
        descriptors.append(segment_descriptor(seg))
        signs.append({
            "word": word,
            "label": row["label"],
            "aliases": [a.strip() for a in _text(row.get("aliases")).split(";") if a.strip()],
            "category": _text(row.get("category")),
            "description": _text(row.get("description")),
            "note": _text(row.get("note")),
            "clip_id": clip,
            "tags": tag(full),
            "fps": seg.fps,
            "hand_rel": np.round(seg.hand_rel, 5).tolist(),
            "wrist_pos": np.round(seg.wrist_pos, 4).tolist(),
            "present": seg.present.tolist(),
            "key_frame": key,
            "views": [_view_take(seg, key, *v) for v in VIEW_TAKES],
        })
    (out / "references.json").write_text(json.dumps({
        "spec_version": SPEC_VERSION,
        "seq_len": int(f.seq_len),
        "rights": "Derived from KSL Dictionary videos. Not for distribution without the owners' permission.",
        "source": "ksl_dictionary",
        "signs": signs,
        "missing": missing,
    }))
    print(f"{len(signs)} vocabulary signs -> {out / 'references.json'}; not found: {', '.join(missing) or 'none'}")
    write_distractors(out / "distractors.npz", [c for c in segments if c not in {s["clip_id"] for s in signs}],
                      segments, p, f.rest_height)
    if len(signs) > 1:
        names = [s["word"] for s in signs]
        dist = distance_matrix(descriptors)
        np.save(out / "distances.npy", dist)
        near = pd.DataFrame(lookalike_table(nearest(names, dist)))
        near.to_csv(out / "lookalikes.csv", index=False)
        print("Closest pairs (most likely to be confused):")
        print(near.head(10)[["sign", "near_1", "nearest_distance"]].to_string(index=False))
    return out


def write_distractors(file: Path, clips: list[str], segments: dict[str, Segment], p: EngineParams,
                      rest_height: float) -> None:
    """Prepared non-vocabulary signs (an even spread of up to DISTRACTORS clips): attempts at signs the
    app does not teach, so calibration can check that Free Sign says "not sure" to them. Local only."""
    if not clips:
        return
    chosen = clips[:: max(1, len(clips) // DISTRACTORS)][:DISTRACTORS]
    prepared = [cut_reference(segments[c], p, rest_height) for c in chosen]
    np.savez_compressed(file, hand_rel=np.stack([s.hand_rel for s in prepared]).astype(np.float32),
                        wrist_pos=np.stack([s.wrist_pos for s in prepared]).astype(np.float32),
                        present=np.stack([s.present for s in prepared]).astype(np.float32),
                        fps=np.array([s.fps for s in prepared]), clip_id=np.array(chosen))
    print(f"{len(prepared)} distractor signs -> {file}")

"""Per-sign reference templates from extracted landmarks.

For each sign: the canonical segment (the app's shadow-hand animation), the dominant hand's finger
angles at its steadiest frame (targets for per-finger feedback), and tags that help pick signs the
headset can check. The tags are hints for the Deaf co-creators, not decisions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import joints as J
from .canonical import (Segment, active_window, from_mediapipe, gap_fill, hold_frames, left_dominant_votes, mirror,
                        resample, steadiest_frame, trim, wrist_extent)
from .config import Config, path, training_relative
from .features import SPEC_VERSION, flexion, spread
from .matching import lookalikes, segment_descriptor


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


HOLD_SHARE = 0.5          # static if at least this share of the active window is a hold
OTHER_HAND_EXTENT = 0.2   # shoulder widths: the other hand counts as signing if it moves this much...
SHAPE_CHANGE = 0.35       # radians: 90th-percentile bend deviation (mean over 15 joints) during the hold


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
        "hands": "two" if other.mean() > 0.5 and (other_extent > OTHER_HAND_EXTENT or other_raised) else "one",
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


def build_references(cfg: Config, vocab_file: str | Path | None = None) -> Path:
    manifest = pd.read_csv(path(cfg, "manifests") / "videos.csv", dtype=str)
    manifest = manifest[manifest["label"].notna()]
    landmarks, f = path(cfg, "landmarks"), cfg.features

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
    for word in load_vocabulary(vocab_file):
        hits = [c for c in segments if word in names_of(by_clip.loc[c])]
        if not hits:  # fall back to a numbered variant: "mother" -> "mother 1"
            variant = re.compile(re.escape(word) + r" \d+")
            hits = sorted(c for c in segments if any(variant.fullmatch(n) for n in names_of(by_clip.loc[c])))
        if not hits:
            missing.append(word)
            continue
        clip = hits[0]
        row = by_clip.loc[clip]
        seg = resample(active_window(segments[clip]), int(f.seq_len))
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
            "tags": tag(segments[clip]),
            "fps": seg.fps,
            "hand_rel": np.round(seg.hand_rel, 5).tolist(),
            "wrist_pos": np.round(seg.wrist_pos, 4).tolist(),
            "present": seg.present.tolist(),
            "key_frame": key,
            "target": {"flexion": np.round(flexion(seg.hand_rel[key, J.RIGHT]), 4).tolist(),
                       "spread": np.round(spread(seg.hand_rel[key, J.RIGHT]), 4).tolist()},
        })
    (out / "references.json").write_text(json.dumps({
        "spec_version": SPEC_VERSION,
        "seq_len": int(f.seq_len),
        "rights": "Derived from KSL Dictionary videos. Not for distribution without the owners' permission.",
        "signs": signs,
        "missing": missing,
    }))
    print(f"{len(signs)} vocabulary signs -> {out / 'references.json'}; not found: {', '.join(missing) or 'none'}")
    if len(signs) > 1:
        near = pd.DataFrame(lookalikes([s["word"] for s in signs], descriptors))
        near.to_csv(out / "lookalikes.csv", index=False)
        print("Closest pairs (most likely to be confused):")
        print(near.head(10)[["sign", "near_1", "nearest_distance"]].to_string(index=False))
    return out

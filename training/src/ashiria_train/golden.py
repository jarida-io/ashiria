"""`ashiria golden`: engine test vectors from synthetic hands (no real people, so they can be committed).

Each case runs the shared runtime path (mirror → gap fill → trim → resample → features → descriptor)
and records the expected outputs, plus DTW distances and per-finger errors between cases. It also
records how attempts and references are cut (active window, held frames, steadiest frame). The C#
(and later Kotlin) ports must reproduce them within the stated tolerances.
"""

from __future__ import annotations

import json

import numpy as np

from . import joints as J
from .canonical import active_window, gap_fill, hold_frames, mirror, resample, steadiest_frame, trim
from .config import TRAINING_DIR, Config
from .features import SPEC_VERSION, flexion, sequence_features
from .matching import descriptor, dtw, finger_errors
from .synthetic import random_segment

CASES = [
    {"name": "one_hand_static", "frames": 40, "two_hands": False, "moving": False, "fps": 30.0},
    {"name": "one_hand_moving_gaps", "frames": 55, "two_hands": False, "moving": True, "fps": 25.0,
     "gaps": [(0, 10, 3), (0, 30, 8)]},
    {"name": "two_hands_moving", "frames": 60, "two_hands": True, "moving": True, "fps": 30.0,
     "gaps": [(0, 5, 2), (1, 20, 4), (1, 40, 9)]},
    {"name": "two_hands_left_handed", "frames": 60, "two_hands": True, "moving": True, "fps": 30.0,
     "gaps": [(1, 12, 5)], "mirror": True},
    {"name": "rest_at_both_ends", "frames": 70, "two_hands": True, "moving": True, "fps": 24.0, "rest_frames": 12},
    {"name": "short_odd_length", "frames": 23, "two_hands": False, "moving": True, "fps": 30.0},
]


def _arr(a: np.ndarray, digits: int = 7) -> list:
    return np.round(np.asarray(a, np.float64), digits).tolist()


def build_golden(cfg: Config) -> str:
    f = cfg.features
    seq_len, max_gap, rest, margin = int(f.seq_len), int(f.gap_fill_frames), float(f.rest_height), int(f.trim_margin_frames)
    rng = np.random.default_rng(2026)
    cases, descriptors, key_flexion = [], [], []
    for spec in CASES:
        seg = random_segment(rng, spec["frames"], spec["two_hands"], spec["moving"], spec["fps"],
                             tuple(spec.get("gaps", ())), spec.get("rest_frames", 0))
        work = mirror(seg) if spec.get("mirror") else seg
        work = trim(gap_fill(work, max_gap), rest, margin)
        res = resample(work, seq_len)
        feats = sequence_features(res.hand_rel, res.wrist_pos, res.present, 1.0 / res.fps)
        desc = descriptor(feats)
        descriptors.append(desc)
        key = seq_len // 2
        key_flexion.append(flexion(res.hand_rel[key, J.RIGHT]))
        win = active_window(work)
        win_res = resample(win, seq_len)
        cases.append({
            "name": spec["name"],
            "mirror": bool(spec.get("mirror", False)),
            "segment": {"fps": seg.fps, "hand_rel": _arr(seg.hand_rel), "wrist_pos": _arr(seg.wrist_pos),
                        "present": seg.present.tolist()},
            "trimmed_length": len(work),
            "resampled_fps": res.fps,
            "resampled_present": res.present.tolist(),
            "features": _arr(feats),
            "descriptor": _arr(desc),
            "key_frame": key,
            "key_flexion_dominant": _arr(key_flexion[-1]),
            "window_length": len(win),
            "window_hold": hold_frames(win).tolist(),
            "window_steadiest": steadiest_frame(win_res),
        })

    n = len(cases)
    pairs = [{"a": i, "b": j, "distance": dtw(descriptors[i], descriptors[j])} for i in range(n) for j in range(i + 1, n)]
    errors = [{"attempt": i, "reference": j, "errors": finger_errors(key_flexion[i], key_flexion[j]).tolist()}
              for i in range(n) for j in range(n) if i != j][:12]
    golden = {
        "spec_version": SPEC_VERSION,
        "seq_len": seq_len,
        "gap_fill_frames": max_gap,
        "rest_height": rest,
        "trim_margin_frames": margin,
        "tolerance": {"features": 1e-3, "descriptor": 1e-3, "dtw": 1e-3, "finger_errors": 2e-3},
        "cases": cases,
        "dtw": pairs,
        "finger_errors": errors,
    }
    out = TRAINING_DIR.parent / "spec" / "golden" / f"engine_v{SPEC_VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden))
    cut = sum(c["window_length"] < c["trimmed_length"] for c in cases)
    print(f"active window shortened {cut}/{len(cases)} cases")
    print(f"{len(cases)} cases, {len(pairs)} DTW pairs -> {out} ({out.stat().st_size / 1e3:.0f} KB)")
    return str(out)

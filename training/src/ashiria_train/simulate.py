"""`ashiria simulate`: how well would Free Sign (nearest reference by DTW) separate the vocabulary?

Each vocabulary reference is perturbed with the training augmentations (viewpoint, jitter, speed,
dropouts, sloppy start/end) to make synthetic "attempts", which are then matched against all
references. This is optimistic: real attempts come from other people's hands, so treat the
result as an upper bound and a list of likely confusions, not as accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .augment import augment
from .canonical import Segment, active_window, gap_fill, resample
from .config import Config, path
from .matching import dtw, segment_descriptor


def _segment(sign: dict) -> Segment:
    return Segment(np.array(sign["hand_rel"]), np.array(sign["wrist_pos"]), np.array(sign["present"]), float(sign["fps"]))


def simulate(cfg: Config, attempts: int = 8, seed: int = 7) -> pd.DataFrame:
    refs_file = path(cfg, "exports") / "references" / "references.json"
    if not refs_file.exists():
        raise SystemExit("No references.json. Run `ashiria references --vocab ...` first.")
    signs = json.loads(refs_file.read_text())["signs"]
    if len(signs) < 2:
        raise SystemExit("Need at least two vocabulary signs.")
    seq_len, max_gap = int(cfg.features.seq_len), int(cfg.features.gap_fill_frames)
    aug = dict(cfg.train.augment)
    rng = np.random.default_rng(seed)
    names = [s["word"] for s in signs]
    refs = [segment_descriptor(_segment(s)) for s in signs]

    rows = []
    for i, sign in enumerate(signs):
        base = _segment(sign)
        for _ in range(attempts):
            attempt = resample(active_window(gap_fill(augment(base, rng, aug), max_gap)), seq_len)
            d = segment_descriptor(attempt)
            dist = np.array([dtw(d, r) for r in refs])
            order = np.argsort(dist)
            rows.append({"sign": names[i], "predicted": names[order[0]], "second": names[order[1]],
                         "rank": int(np.where(order == i)[0][0]) + 1,
                         "margin": float(dist[order[1]] - dist[order[0]])})
    df = pd.DataFrame(rows)
    per_sign = df.groupby("sign").agg(top1=("rank", lambda r: (r == 1).mean()), top3=("rank", lambda r: (r <= 3).mean()))
    confusions = df[df["predicted"] != df["sign"]].groupby(["sign", "predicted"]).size().sort_values(ascending=False)
    out = path(cfg, "exports") / "references"
    per_sign.sort_values("top1").to_csv(out / "simulated_recognition.csv")
    print(f"{len(signs)} signs × {attempts} synthetic attempts: top-1 {(df['rank'] == 1).mean():.0%}, "
          f"top-3 {(df['rank'] <= 3).mean():.0%} (optimistic: same signer, synthetic variation)")
    print("Weakest signs:")
    print(per_sign.sort_values("top1").head(10).round(2).to_string())
    if len(confusions):
        print("Most common confusions:")
        print(confusions.head(10).to_string())
    return per_sign

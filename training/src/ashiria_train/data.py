"""Step 3: canonical segments for every clip, the label list, and signer-independent splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import joints as J
from .augment import augment
from .canonical import Segment, from_mediapipe, gap_fill, hand_activity, left_dominant_votes, mirror, resample, trim
from .config import Config, path
from .features import SPEC_VERSION, sequence_features

CACHE_FILE = "segments.npz"


def _signer(value) -> str | None:
    return str(value) if pd.notna(value) and str(value) != "" else None


def build(cfg: Config) -> pd.DataFrame:
    manifest = pd.read_csv(path(cfg, "manifests") / "videos.csv")
    manifest = manifest[manifest["label"].notna()]
    landmarks = path(cfg, "landmarks")
    f = cfg.features

    segments: dict[str, Segment] = {}
    rows = []
    for r in tqdm(manifest.itertuples(), total=len(manifest), desc="build"):
        file = landmarks / f"{r.clip_id}.npz"
        seg, status = None, "not extracted"
        if file.exists():
            with np.load(file) as d:
                seg, status = from_mediapipe(d)
            if seg is not None:
                seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
                if seg is None or len(seg) < 4:
                    seg, status = None, "no signing found"
                elif seg.present.max(axis=1).mean() < f.min_present_ratio:
                    seg, status = None, "hands rarely visible"
        if seg is not None:
            segments[r.clip_id] = seg
        rows.append({"clip_id": r.clip_id, "path": r.path, "label": r.label, "signer": _signer(r.signer), "status": status})
    index = pd.DataFrame(rows)

    # Dominant hand: mirror left-handed signing so slot 0 is always the dominant hand.
    left = {}
    for c, s in segments.items():
        a = hand_activity(s)
        left[c] = bool(a[J.LEFT] > a[J.RIGHT])
    if f.dominant_hand == "per_signer":
        for _, group in index[index["clip_id"].isin(segments)].groupby("signer"):
            n_left, n_right = left_dominant_votes([segments[c] for c in group["clip_id"]])
            left.update({c: n_left > n_right for c in group["clip_id"]})
    elif f.dominant_hand in ("right", "left"):
        left = {c: f.dominant_hand == "left" for c in left}
    for c, is_left in left.items():
        if is_left:
            segments[c] = mirror(segments[c])
    index["mirrored"] = index["clip_id"].map(left)

    ok = index["status"] == "ok"
    counts = index.loc[ok, "label"].value_counts()
    classes = sorted(counts[counts >= cfg.split.min_clips_per_class].index)
    if counts.size and not classes:
        raise SystemExit(
            f"Every label has fewer than {cfg.split.min_clips_per_class} clips (the most is {counts.max()}). "
            "A classifier needs several clips per sign from several signers. For one-video-per-sign data "
            "such as the KSL Dictionary, use `ashiria references` instead."
        )
    index.loc[ok & ~index["label"].isin(classes), "status"] = "too few clips for this label"
    ok = index["status"] == "ok"
    index["split"] = None
    index.loc[ok, "split"] = make_splits(index[ok], cfg.split)

    keep = index.loc[ok, "clip_id"].tolist()
    if not keep:
        raise SystemExit("No usable clips. Check data/manifests/extraction.csv and the statuses printed above.")
    out = path(cfg, "cache")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / CACHE_FILE,
        hand_rel=np.concatenate([segments[c].hand_rel for c in keep]).astype(np.float32),
        wrist_pos=np.concatenate([segments[c].wrist_pos for c in keep]).astype(np.float32),
        present=np.concatenate([segments[c].present for c in keep]).astype(np.float32),
        fps=np.array([segments[c].fps for c in keep], np.float32),
        offsets=np.concatenate([[0], np.cumsum([len(segments[c]) for c in keep])]),
        clip_id=np.array(keep),
    )
    index.to_csv(out / "index.csv", index=False)
    (out / "labels.json").write_text(json.dumps({"spec_version": SPEC_VERSION, "classes": classes}, indent=2))

    print(index["status"].value_counts().to_string())
    print(index.loc[ok, "split"].value_counts().to_string())
    print(f"{len(classes)} labels with at least {cfg.split.min_clips_per_class} clips")
    return index


def make_splits(df: pd.DataFrame, s) -> pd.Series:
    """Test (and, with enough signers, validation) signers never appear in training."""
    rng = np.random.default_rng(s.seed)
    out = pd.Series("train", index=df.index, dtype=object)
    signers = df["signer"].dropna().unique()
    total = len(df)

    def take_signers(pool, fraction):
        chosen, n = [], 0
        sizes = df.groupby("signer").size()
        for sg in pool:
            if n >= fraction * total:
                break
            chosen.append(sg)
            n += sizes[sg]
        return chosen

    rest = df
    if len(signers) >= 3:
        order = list(rng.permutation(signers))
        test = take_signers(order, s.test_fraction)
        out[df["signer"].isin(test)] = "test"
        rest = df[~df["signer"].isin(test)]
        if len(signers) - len(test) >= 4:
            val = take_signers([sg for sg in order if sg not in test], s.val_fraction)
            out[df["signer"].isin(val)] = "val"
            return out
    else:
        print("Warning: fewer than 3 signers identified, so test clips come from the same people as training "
              "clips and accuracy will look better than it will be on the headset.")

    for _, g in rest.groupby("label"):
        idx = rng.permutation(g.index.values)
        fractions = (s.val_fraction,) if len(signers) >= 3 else (s.test_fraction, s.val_fraction)
        names = ("val",) if len(signers) >= 3 else ("test", "val")
        start = 0
        for name, frac in zip(names, fractions):
            n = max(1, round(len(idx) * frac)) if len(idx) >= 3 else 0
            out[idx[start : start + n]] = name
            start += n
    return out


class Cache:
    """The built dataset: canonical segments plus labels and splits."""

    def __init__(self, cfg: Config):
        d = path(cfg, "cache")
        if not (d / CACHE_FILE).exists():
            raise SystemExit("No dataset cache. Run `ashiria build` first.")
        with np.load(d / CACHE_FILE) as z:
            self.hand_rel, self.wrist_pos, self.present = z["hand_rel"], z["wrist_pos"], z["present"]
            self.fps, self.offsets = z["fps"], z["offsets"]
            self.clip_ids = [str(c) for c in z["clip_id"]]
        meta = pd.read_csv(d / "index.csv").set_index("clip_id")
        self.meta = meta.loc[self.clip_ids].reset_index()
        self.classes = json.loads((d / "labels.json").read_text())["classes"]
        self.y = self.meta["label"].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        self.split = self.meta["split"].to_numpy()
        self.seq_len = int(cfg.features.seq_len)
        self.max_gap = int(cfg.features.gap_fill_frames)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def segment(self, i: int) -> Segment:
        a, b = self.offsets[i], self.offsets[i + 1]
        return Segment(self.hand_rel[a:b].astype(np.float64), self.wrist_pos[a:b].astype(np.float64),
                       self.present[a:b].astype(np.float64), float(self.fps[i]))

    def features(self, i: int, rng: np.random.Generator | None = None, aug: dict | None = None) -> np.ndarray:
        seg = self.segment(i)
        if aug is not None:
            seg = gap_fill(augment(seg, rng, aug), self.max_gap)
        seg = resample(seg, self.seq_len)
        return sequence_features(seg.hand_rel, seg.wrist_pos, seg.present, 1.0 / seg.fps)

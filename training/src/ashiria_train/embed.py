"""`ashiria embed`: an experiment in recognising any of the 2,548 KSL Dictionary signs (plan §4.5).

The dictionary has one video per sign, all by one signer, so a classifier can't be trained the usual way. Instead
SignNet's encoder learns an **embedding**: two differently augmented copies of the same sign land close together,
different signs far apart (contrastive learning, NT-Xent). Recognition is then the nearest reference embedding.

Evaluation uses simulated attempts (the calibration augmentations, cut by the real attempt segmenter) and compares
with DTW matching (the current Free Sign). Both use the same signer as the references, so this measures
robustness to the variation we simulate, not accuracy on other people: that needs Jerry's Quest attempts.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .augment import augment
from .calibrate import VIDEO_REST_HEIGHT, simulate_attempt
from .canonical import Segment, active_window, from_mediapipe, gap_fill, resample, trim
from .config import Config, load_config, path
from .features import sequence_features
from .matching import dtw_many, segment_descriptor
from .model import SignNet
from .params import EngineParams
from .pipeline import cut_reference
from .train import pick_device

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"


def load_references(p: EngineParams) -> tuple[list[str], list[Segment]]:
    """Every extracted dictionary clip as a reference cut like a live attempt; returns (labels, segments).
    Cached in the dictionary cache folder (local, git-ignored) because cutting 2,548 clips takes about a minute."""
    dcfg = load_config(DICTIONARY_CONFIG)
    cache = path(dcfg, "cache") / f"embed_refs_seq{p.seq_len}.npz"
    labels_csv = path(dcfg, "manifests") / "dictionary_labels.csv"
    if cache.exists() and (not labels_csv.exists() or cache.stat().st_mtime > labels_csv.stat().st_mtime):
        with np.load(cache) as d:
            return (d["labels"].tolist(), [Segment(d["hand_rel"][i].astype(np.float64), d["wrist_pos"][i].astype(np.float64),
                                                   d["present"][i].astype(np.float64), float(d["fps"][i])) for i in range(len(d["fps"]))])
    labels, refs = _cut_references(p, dcfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=np.array(labels), hand_rel=np.stack([r.hand_rel for r in refs]).astype(np.float32),
                        wrist_pos=np.stack([r.wrist_pos for r in refs]).astype(np.float32),
                        present=np.stack([r.present for r in refs]).astype(np.float32), fps=np.array([r.fps for r in refs]))
    return labels, refs


def _cut_references(p: EngineParams, dcfg) -> tuple[list[str], list[Segment]]:
    manifest = pd.read_csv(path(dcfg, "manifests") / "videos.csv", dtype=str)
    landmarks, f = path(dcfg, "landmarks"), dcfg.features
    labels, refs = [], []
    for _, row in manifest.iterrows():
        file = landmarks / f"{row['clip_id']}.npz"
        if not file.exists() or not isinstance(row["label"], str):
            continue
        with np.load(file) as d:
            seg, _ = from_mediapipe(d)
        if seg is None:
            continue
        seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
        if seg is None or len(seg) < 4:
            continue
        ref = cut_reference(seg, p, f.rest_height)
        if ref is not None:
            labels.append(row["label"])
            refs.append(ref)
    return labels, refs


class Views(Dataset):
    """Two augmented views of each reference, as feature sequences."""

    def __init__(self, refs: list[Segment], aug: dict, seq_len: int, seed: int):
        self.refs, self.aug, self.seq_len, self.seed = refs, aug, seq_len, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.refs)

    def view(self, seg: Segment, rng) -> np.ndarray:
        s = resample(active_window(gap_fill(augment(seg, rng, self.aug), 5)), self.seq_len)
        return sequence_features(s.hand_rel, s.wrist_pos, s.present, 1.0 / s.fps)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        return torch.from_numpy(self.view(self.refs[i], rng)), torch.from_numpy(self.view(self.refs[i], rng))


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([a, b])
    sim = z @ z.T / temperature
    n = a.shape[0]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


class Embedder(nn.Module):
    """SignNet whose head outputs an L2-normalised embedding."""

    def __init__(self, in_dim: int, out_dim: int = 128, **kw):
        super().__init__()
        self.net = SignNet(in_dim, out_dim, **kw)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


@torch.no_grad()
def embed_all(model: Embedder, feats: list[np.ndarray], device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(np.stack(feats[i:i + batch])).to(device)).cpu().numpy() for i in range(0, len(feats), batch)]
    return np.concatenate(out)


def evaluate(model, labels, refs, ref_feats, device, p: EngineParams, aug: dict, signs: list[int], attempts: int, seed: int,
             with_dtw: bool = True) -> dict:
    """Simulated attempts at `signs` (indices into refs) → nearest of all references, by embedding and by DTW."""
    rng = np.random.default_rng(seed)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    ref_emb = embed_all(model, ref_feats, device)
    ref_desc = np.stack([segment_descriptor(r, p.weights) for r in refs]).astype(np.float64) if with_dtw else None
    rank_e, rank_d, learned_e, learned_d = [], [], [], []
    learned = np.array(signs)
    for i in signs:
        for _ in range(attempts):
            att = simulate_attempt(refs[i], rng, aug, p_sim)
            if att is None:
                continue
            f = sequence_features(att.hand_rel, att.wrist_pos, att.present, 1.0 / att.fps)
            e = embed_all(model, [f], device)[0]
            sim = ref_emb @ e
            rank_e.append(int((sim > sim[i]).sum()) + 1)
            learned_e.append(int(learned[np.argmax(sim[learned])] == i))
            if with_dtw:
                d = dtw_many(segment_descriptor(att, p.weights).astype(np.float64), ref_desc, p.dtw_band)
                rank_d.append(int((d < d[i]).sum()) + 1)
                learned_d.append(int(learned[np.argmin(d[learned])] == i))

    def summary(r):
        r = np.array(r)
        return {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()), "median_rank": float(np.median(r))} if len(r) else {}

    out = {"attempts": len(rank_e), "embedding_all_signs": summary(rank_e),
           "embedding_learned_set": float(np.mean(learned_e)) if learned_e else None}
    if with_dtw:
        out["dtw_all_signs"] = summary(rank_d)
        out["dtw_learned_set"] = float(np.mean(learned_d)) if learned_d else None
    return out


def run_embed(cfg: Config, epochs: int = 60, batch: int = 256, temperature: float = 0.1, seed: int = 7,
              attempts: int = 4) -> Path:
    torch.manual_seed(seed)
    p = EngineParams.from_config(cfg)
    t0 = time.time()
    labels, refs = load_references(p)
    print(f"{len(refs)} dictionary references ({time.time() - t0:.0f} s)")
    aug = {**dict(cfg.calibrate.augment), "edge_crop": 0.1}
    data = Views(refs, aug, p.seq_len, seed)
    ref_feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    stack = np.stack(ref_feats).reshape(-1, ref_feats[0].shape[-1])
    device = pick_device(str(cfg.train.device))
    m = cfg.train.model
    model = Embedder(stack.shape[-1], 128, dim=int(m.dim), stages=int(m.stages), conv_per_stage=int(m.conv_per_stage),
                     kernel_size=int(m.kernel_size), heads=int(m.heads), dropout=0.1)
    model.net.set_normalization(stack.mean(0), stack.std(0))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * ((len(data) + batch - 1) // batch))
    loader = DataLoader(data, batch_size=batch, shuffle=True, drop_last=True, num_workers=4, persistent_workers=False)

    vocab = [line.split("#")[0].strip() for line in open("configs/vocabulary_v1.txt") if line.split("#")[0].strip()]
    index = {lab: k for k, lab in enumerate(labels)}
    chosen = [index[w] for w in vocab if w in index] + [index[w + " 1"] for w in vocab if w not in index and w + " 1" in index]
    rng = np.random.default_rng(seed)
    others = [int(k) for k in rng.choice(len(refs), 100, replace=False) if int(k) not in chosen]

    out = path(cfg, "runs") / f"embed-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Training on {device} for {epochs} epochs; {len(chosen)} vocabulary signs and {len(others)} others for evaluation")
    for epoch in range(epochs):
        model.train()
        data.epoch = epoch
        total = 0.0
        for a, b in loader:
            loss = nt_xent(model(a.to(device)), model(b.to(device)), temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  loss {total / max(len(loader), 1):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"state": model.state_dict(), "labels": labels}, out / "model.pt")

    print("Evaluating on simulated attempts...")
    report = {
        "note": "one signer, simulated variation: robustness to augmentation, not accuracy on other people",
        "references": len(refs), "epochs": epochs,
        "vocabulary": evaluate(model, labels, refs, ref_feats, device, p, dict(cfg.calibrate.augment), chosen, attempts, seed),
        "other_signs": evaluate(model, labels, refs, ref_feats, device, p, dict(cfg.calibrate.augment), others, 2, seed + 1),
        # Harder: the training augmentations (bigger rotations, more jitter) plus finger noise.
        "vocabulary_hard": evaluate(model, labels, refs, ref_feats, device, p,
                                    {**dict(cfg.train.augment), "finger_noise_rad": 0.15}, chosen, attempts, seed + 2),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for name in ("vocabulary", "other_signs", "vocabulary_hard"):
        r = report[name]
        print(f"{name}: {r['attempts']} attempts | among all {len(refs)} signs: embedding top-1 "
              f"{r['embedding_all_signs'].get('top1', 0):.0%} top-5 {r['embedding_all_signs'].get('top5', 0):.0%}, "
              f"DTW top-1 {r['dtw_all_signs'].get('top1', 0):.0%} top-5 {r['dtw_all_signs'].get('top5', 0):.0%}")
    print(f"-> {out}")
    return out

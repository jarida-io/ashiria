"""`ashiria embed`: an experiment in recognising any of the 2,548 KSL Dictionary signs (plan §4.5).

The dictionary has one video per sign, all by one signer, so a classifier can't be trained the usual way. Instead
SignNet's encoder learns an **embedding**: two differently augmented copies of the same sign land close together,
different signs far apart (contrastive learning, NT-Xent). Recognition is then the nearest reference embedding.

Evaluation uses simulated attempts (the calibration augmentations, cut by the real attempt segmenter set to take
whole signs, see WHOLE_SIGN) and compares with DTW matching (the current Free Sign) on the same references. Both
use the same signer as the references, so this measures robustness to the variation we simulate, not accuracy on
other people: that needs Jerry's Quest attempts.
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
# The recogniser takes whole signs: an attempt ends when the hands drop (or after 10 s), not when a hand holds still.
# Verify ends a single sign at its hold, but that cut a phrase down to its first sign: "I Lost My Bag" kept 0.7 s of
# 4.6 s, so phrases that start alike became the same reference. Only segmenter settings change, not engine code.
WHOLE_SIGN = {"hold_s": 11.0, "max_attempt_s": 10.0}


def recogniser_params(p: EngineParams) -> EngineParams:
    return dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, **WHOLE_SIGN))


def load_references(p: EngineParams) -> tuple[list[str], list[Segment], list[str]]:
    """Every extracted dictionary clip as a reference cut like a live attempt; returns (labels, segments, clip ids).
    Cached in the dictionary cache folder (local, git-ignored) because cutting 2,646 clips takes about a minute, and
    rebuilt when `ashiria import-dictionary` changes the labels."""
    dcfg = load_config(DICTIONARY_CONFIG)
    sp = p.segmenter
    cache = path(dcfg, "cache") / f"embed_refs_seq{p.seq_len}_hold{sp.hold_s:g}_max{sp.max_attempt_s:g}.npz"
    labels_csv = path(dcfg, "manifests") / "dictionary_labels.csv"
    if cache.exists() and (not labels_csv.exists() or cache.stat().st_mtime > labels_csv.stat().st_mtime):
        with np.load(cache) as d:  # each d[key] decompresses the whole array, so read each once
            labels, fps = d["labels"].tolist(), d["fps"]
            hand_rel, wrist_pos, present = (d[k].astype(np.float64) for k in ("hand_rel", "wrist_pos", "present"))
            ids = d["clip_ids"].tolist() if "clip_ids" in d.files else _clip_ids_in_order(dcfg, labels)
        refs = [Segment(hand_rel[i], wrist_pos[i], present[i], float(fps[i])) for i in range(len(fps))]
        if ids is not None:
            return labels, refs, ids
    labels, refs, ids = _cut_references(p, dcfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=np.array(labels), clip_ids=np.array(ids),
                        hand_rel=np.stack([r.hand_rel for r in refs]).astype(np.float32),
                        wrist_pos=np.stack([r.wrist_pos for r in refs]).astype(np.float32),
                        present=np.stack([r.present for r in refs]).astype(np.float32), fps=np.array([r.fps for r in refs]))
    return labels, refs, ids


def _labelled_clips(dcfg) -> list[tuple[str, str]]:
    """(clip id, label) of every labelled clip with landmarks, in manifest order."""
    manifest = pd.read_csv(path(dcfg, "manifests") / "videos.csv", dtype=str)
    landmarks = path(dcfg, "landmarks")
    return [(r["clip_id"], r["label"]) for _, r in manifest.iterrows()
            if isinstance(r["label"], str) and (landmarks / f"{r['clip_id']}.npz").exists()]


def _clip_ids_in_order(dcfg, labels: list[str]) -> list[str] | None:
    """Clip ids for a cache written before it stored them: valid only if every clip was cut (the usual case)."""
    clips = _labelled_clips(dcfg)
    return [c for c, _ in clips] if [lab for _, lab in clips] == labels else None


def _cut_references(p: EngineParams, dcfg) -> tuple[list[str], list[Segment], list[str]]:
    landmarks, f = path(dcfg, "landmarks"), dcfg.features
    labels, refs, ids = [], [], []
    for clip, label in _labelled_clips(dcfg):
        with np.load(landmarks / f"{clip}.npz") as d:
            seg, _ = from_mediapipe(d)
        if seg is None:
            continue
        seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
        if seg is None or len(seg) < 4:
            continue
        ref = cut_reference(seg, p, f.rest_height)
        if ref is not None:
            labels.append(label)
            refs.append(ref)
            ids.append(clip)
    return labels, refs, ids


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
    p = recogniser_params(EngineParams.from_config(cfg))
    t0 = time.time()
    labels, refs, _ = load_references(p)
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
        "cut": "whole signs: an attempt ends when the hands drop (up to 10 s), not at a hold",
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
    sign_results(cfg, out)
    return out


def latest_run(cfg: Config) -> Path | None:
    """The newest `ashiria embed` run with a saved model."""
    runs = sorted(path(cfg, "runs").glob("embed-*/model.pt"))
    return runs[-1].parent if runs else None


def _load_model(cfg: Config, run: Path, in_dim: int, device) -> Embedder:
    m = cfg.train.model
    model = Embedder(in_dim, 128, dim=int(m.dim), stages=int(m.stages), conv_per_stage=int(m.conv_per_stage),
                     kernel_size=int(m.kernel_size), heads=int(m.heads), dropout=0.1)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu")["state"])
    return model.to(device).eval()


def sign_results(cfg: Config, run: Path | None = None, attempts: int = 3, seed: int = 11, closest: int = 5) -> Path:
    """Per-sign results of a trained run, for the sign browser: each reference's closest signs by embedding, and how
    often simulated attempts at it (normal and harder variation) are recognised first among every dictionary sign.
    The current matcher (DTW) is scored per sign for the v1 vocabulary only: against 2,646 signs it takes seconds per
    attempt. Writes `signs.json` into the run folder."""
    run = Path(run) if run else latest_run(cfg)
    if run is None:
        raise SystemExit("No embed run with a model yet: run `ashiria embed` first.")
    t0 = time.time()
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    device = pick_device(str(cfg.train.device))
    model = _load_model(cfg, run, feats[0].shape[-1], device)
    trained_on = len(torch.load(run / "model.pt", map_location="cpu")["labels"])
    ref_emb = embed_all(model, feats, device)

    sim = ref_emb @ ref_emb.T
    np.fill_diagonal(sim, -np.inf)
    near = np.argsort(-sim, axis=1)[:, :closest]
    results = {ids[i]: {"closest": [[ids[j], round(float(sim[i, j]), 3)] for j in near[i]]} for i in range(len(refs))}

    vocab = [line.split("#")[0].strip() for line in open("configs/vocabulary_v1.txt") if line.split("#")[0].strip()]
    index = {lab: k for k, lab in enumerate(labels)}
    v1 = {index[w] if w in index else index.get(w + " 1") for w in vocab} - {None}
    ref_desc = np.stack([segment_descriptor(r, p.weights) for r in refs]).astype(np.float64)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    tests = {"normal": dict(cfg.calibrate.augment), "harder": {**dict(cfg.train.augment), "finger_noise_rad": 0.15}}
    summary = {}
    for k, (name, aug) in enumerate(tests.items()):
        rng = np.random.default_rng((seed, k))
        owner, att_feats, att_segs = [], [], []
        for i, r in enumerate(refs):
            for _ in range(attempts):
                a = simulate_attempt(r, rng, aug, p_sim)
                if a is not None:
                    owner.append(i)
                    att_feats.append(sequence_features(a.hand_rel, a.wrist_pos, a.present, 1.0 / a.fps))
                    att_segs.append(a)
        top = (embed_all(model, att_feats, device) @ ref_emb.T).argmax(1)
        dtw_top = {}
        if name == "normal":
            for a, (i, seg) in enumerate(zip(owner, att_segs)):
                if i in v1:
                    dtw_top[a] = int(np.argmin(dtw_many(segment_descriptor(seg, p.weights).astype(np.float64), ref_desc, p.dtw_band)))
        for i in range(len(refs)):
            results[ids[i]][name] = {"tried": attempts, "tests": 0, "first": 0, "mistaken_for": []}
        for a, i in enumerate(owner):
            r = results[ids[i]][name]
            r["tests"] += 1
            if top[a] == i:
                r["first"] += 1
            elif ids[top[a]] not in r["mistaken_for"]:
                r["mistaken_for"].append(ids[top[a]])
            if a in dtw_top:
                r["dtw_tests"] = r.get("dtw_tests", 0) + 1
                r["dtw_first"] = r.get("dtw_first", 0) + int(dtw_top[a] == i)
        hits = np.array([top[a] == i for a, i in enumerate(owner)])
        summary[name] = {"tests": len(owner), "first": float(hits.mean()) if len(hits) else None,
                         "signs_always_first": int(sum(results[c][name]["tests"] > 0 and
                                                       results[c][name]["first"] == results[c][name]["tests"] for c in ids))}
        if dtw_top:
            v1_hits = [top[a] == owner[a] for a in dtw_top]
            summary[name]["v1_first"] = float(np.mean(v1_hits))
            summary[name]["v1_dtw_first"] = float(np.mean([dtw_top[a] == owner[a] for a in dtw_top]))
        print(f"{name}: {len(owner)} attempts, recognised first {summary[name]['first']:.1%} ({time.time() - t0:.0f} s)", flush=True)

    report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
    out = {"run": run.name, "trained_on": trained_on, "signs": len(refs), "attempts_per_sign": attempts,
           "note": "one signer, simulated variation: robustness to variation, not accuracy on other people",
           "summary": summary, "report": {k: report[k] for k in ("vocabulary", "other_signs", "vocabulary_hard") if k in report},
           "results": results}
    (run / "signs.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {run / 'signs.json'}")
    return run / "signs.json"

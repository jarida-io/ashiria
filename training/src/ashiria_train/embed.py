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
from torch.utils.data import DataLoader, Dataset, Sampler

from .augment import augment
from .calibrate import VIDEO_REST_HEIGHT, simulated_stream
from .canonical import Segment, active_window, from_mediapipe, gap_fill, resample, trim
from .config import Config, load_config, path, training_relative
from .features import sequence_features
from .matching import dtw_many, segment_descriptor
from .model import SignNet
from .params import EngineParams
from .pipeline import cut_reference, prepare
from .segmenter import segment_stream
from .train import pick_device

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"
# The recogniser takes whole signs: an attempt ends when the hands drop (or after 10 s), not when a hand holds still.
# Verify ends a single sign at its hold, but that cut a phrase down to its first sign: "I Lost My Bag" kept 0.7 s of
# 4.6 s, so phrases that start alike became the same reference. Only segmenter settings change, not engine code.
WHOLE_SIGN = {"hold_s": 11.0, "max_attempt_s": 10.0}
# Whole signs are longer, so the recogniser resamples them to 64 frames (32 left too few for the handshape once the
# rise and drop were included). The current matcher (DTW) is compared at the engine's own frame count.
RECOGNISER_FRAMES = 64


def whole_sign(p: EngineParams) -> EngineParams:
    return dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, **WHOLE_SIGN))


def recogniser_params(p: EngineParams) -> EngineParams:
    return dataclasses.replace(whole_sign(p), seq_len=RECOGNISER_FRAMES)


def _simulate(ref: Segment, rng, aug: dict, p_sim: EngineParams, p_dtw: EngineParams) -> tuple[Segment | None, Segment | None]:
    """One simulated attempt, prepared for the recogniser (p_sim's frame count) and for DTW (p_dtw's)."""
    sp = p_sim.segmenter
    attempts = segment_stream(sp, simulated_stream(ref, rng, aug, sp.tick_hz, drop_after_hold=sp.hold_s >= sp.max_attempt_s))
    if not attempts:
        return None, None
    raw = attempts[0].segment
    return prepare(raw, p_sim, sp.rest_height), prepare(raw, p_dtw, sp.rest_height)


def _matcher_references(cfg: Config, ids: list[str]) -> tuple[EngineParams, list[Segment]]:
    """The same whole-sign references at the engine's frame count, for the DTW comparison."""
    p_dtw = whole_sign(EngineParams.from_config(cfg))
    _, refs, dtw_ids = load_references(p_dtw)
    if dtw_ids != ids:
        raise SystemExit("Recogniser and matcher references differ; delete data/dictionary/cache and rerun.")
    return p_dtw, refs


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
    """Two augmented views of each reference, as feature sequences. Each view draws its variation from one of `augs`
    at random (mild and stronger: see `training_variation`)."""

    def __init__(self, refs: list[Segment], augs: list[dict], seq_len: int, seed: int):
        self.refs, self.augs, self.seq_len, self.seed = refs, augs, seq_len, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.refs)

    def view(self, seg: Segment, rng) -> np.ndarray:
        aug = self.augs[int(rng.integers(len(self.augs)))]
        s = resample(active_window(gap_fill(augment(seg, rng, aug), 5)), self.seq_len)
        return sequence_features(s.hand_rel, s.wrist_pos, s.present, 1.0 / s.fps)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        return torch.from_numpy(self.view(self.refs[i], rng)), torch.from_numpy(self.view(self.refs[i], rng))


class LookalikeBatches(Sampler):
    """Batches where `share` of the signs come as look-alike neighbourhoods (a sign and the signs a previous run
    found close to it or mistook it for: groups.py), the rest at random. Contrastive training only learns what
    tells signs apart within a batch, so look-alikes need to meet there. No sign appears twice in a batch."""

    def __init__(self, n: int, lookalikes: list[list[int]], batch: int, share: float, seed: int):
        self.n, self.lookalikes, self.batch, self.share, self.seed = n, lookalikes, batch, share, seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.n // self.batch

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(self.n)
        seeds = [i for i in rng.permutation(self.n) if self.lookalikes[i]]
        pos, s = 0, 0
        for _ in range(len(self)):
            chosen, taken = [], set()
            while len(chosen) < self.share * self.batch and s < len(seeds):
                for j in [seeds[s], *self.lookalikes[seeds[s]]]:
                    if j not in taken and len(chosen) < self.batch:
                        chosen.append(j)
                        taken.add(j)
                s += 1
            while len(chosen) < self.batch:
                j = int(order[pos % self.n])
                pos += 1
                if j not in taken:
                    chosen.append(j)
                    taken.add(j)
            yield chosen


def lookalike_lists(ids: list[str]) -> list[list[int]] | None:
    """Each reference's look-alikes as indices, from the last `ashiria sign-groups` (None if there is none yet)."""
    from .groups import load_groups

    g = load_groups()
    if not g:
        return None
    index = {c: i for i, c in enumerate(ids)}
    return [[index[l["clip"]] for l in g["signs"].get(c, {}).get("lookalikes", []) if l["clip"] in index] for c in ids]


def training_variation(cfg: Config) -> list[dict]:
    """What the recogniser trains on: half mild variation (the calibration's "someone else", which the normal test
    uses) and half stronger. The stronger half stops short of the harder test (15 degree turns, 0.006 m jitter,
    0.15 rad finger noise), so that test still asks for more than training showed."""
    mild = {**dict(cfg.calibrate.augment), "edge_crop": 0.1, **VIEW_MILD}
    stronger = {**dict(cfg.train.augment), "rotate_deg": 12.0, "joint_noise_m": 0.005, "finger_noise_rad": 0.13, **VIEW_STRONG}
    return [mild, stronger]


# Camera viewpoints (augment.camera_view): a webcam beside, above or below the signer misjudges depth along its line
# of sight and the hand's tilt. Training sees these; the viewpoint test goes further (wider angles, bigger errors).
VIEW_MILD = {"view_prob": 0.5, "view_yaw_deg": 25, "view_pitch_deg": 15, "view_depth": [0.7, 1.1], "view_depth_noise_m": 0.003,
             "view_tilt_deg": 6, "view_wrist_depth": 0.1}
VIEW_STRONG = {"view_prob": 0.8, "view_yaw_deg": 35, "view_pitch_deg": 25, "view_depth": [0.55, 1.2], "view_depth_noise_m": 0.004,
               "view_tilt_deg": 12, "view_wrist_depth": 0.15}
VIEW_TEST = {"view_prob": 1.0, "view_yaw_deg": 40, "view_pitch_deg": 30, "view_depth": [0.5, 1.25], "view_depth_noise_m": 0.005,
             "view_tilt_deg": 15, "view_wrist_depth": 0.2}


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
             p_dtw: EngineParams | None = None, dtw_refs: list[Segment] | None = None) -> dict:
    """Simulated attempts at `signs` (indices into refs) → nearest of all references, by embedding and, given
    `dtw_refs` (the same references at the matcher's frame count), by DTW."""
    rng = np.random.default_rng(seed)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    with_dtw = dtw_refs is not None
    p_dtw = p_dtw or p
    ref_emb = embed_all(model, ref_feats, device)
    ref_desc = np.stack([segment_descriptor(r, p_dtw.weights) for r in dtw_refs]).astype(np.float64) if with_dtw else None
    rank_e, rank_d, learned_e, learned_d = [], [], [], []
    not_cut = 0
    learned = np.array(signs)
    for i in signs:
        for _ in range(attempts):
            att, att_dtw = _simulate(refs[i], rng, aug, p_sim, p_dtw)
            if att is None:
                not_cut += 1
                continue
            f = sequence_features(att.hand_rel, att.wrist_pos, att.present, 1.0 / att.fps)
            e = embed_all(model, [f], device)[0]
            sim = ref_emb @ e
            rank_e.append(int((sim > sim[i]).sum()) + 1)
            learned_e.append(int(learned[np.argmax(sim[learned])] == i))
            if with_dtw:
                d = dtw_many(segment_descriptor(att_dtw, p_dtw.weights).astype(np.float64), ref_desc, p_dtw.dtw_band)
                rank_d.append(int((d < d[i]).sum()) + 1)
                learned_d.append(int(learned[np.argmin(d[learned])] == i))

    def summary(r):
        r = np.array(r)
        return {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()), "median_rank": float(np.median(r))} if len(r) else {}

    out = {"attempts": len(rank_e), "segmentation_misses": not_cut, "embedding_all_signs": summary(rank_e),
           "embedding_learned_set": float(np.mean(learned_e)) if learned_e else None}
    if with_dtw:
        out["dtw_all_signs"] = summary(rank_d)
        out["dtw_learned_set"] = float(np.mean(learned_d)) if learned_d else None
    return out


def run_embed(cfg: Config, epochs: int = 60, batch: int = 256, temperature: float = 0.1, seed: int = 7,
              attempts: int = 4, lookalike_share: float = 0.5) -> Path:
    torch.manual_seed(seed)
    p = recogniser_params(EngineParams.from_config(cfg))
    t0 = time.time()
    labels, refs, ref_ids = load_references(p)
    print(f"{len(refs)} dictionary references ({time.time() - t0:.0f} s)")
    data = Views(refs, training_variation(cfg), p.seq_len, seed)
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
    lookalikes = lookalike_lists(ref_ids) if lookalike_share > 0 else None
    sampler = LookalikeBatches(len(data), lookalikes, batch, lookalike_share, seed) if lookalikes else None
    loader = (DataLoader(data, batch_sampler=sampler, num_workers=4, persistent_workers=False) if sampler else
              DataLoader(data, batch_size=batch, shuffle=True, drop_last=True, num_workers=4, persistent_workers=False))
    if sampler:
        print(f"Look-alikes share batches: {lookalike_share:.0%} of each batch, "
              f"{sum(bool(x) for x in lookalikes)} signs with look-alikes (groups.py)")

    vocab = [line.split("#")[0].strip() for line in open(training_relative("configs/vocabulary_v1.txt")) if line.split("#")[0].strip()]
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
        if sampler:
            sampler.epoch = epoch
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
    torch.save({"state": model.state_dict(), "labels": labels, "epochs": epochs, "frames": p.seq_len,
                "variation": training_variation(cfg), "lookalike_share": lookalike_share if sampler else 0.0},
               out / "model.pt")
    return evaluate_run(cfg, out, attempts, seed)


def evaluate_run(cfg: Config, run: Path | None = None, attempts: int = 4, seed: int = 7) -> Path:
    """Test a trained run on simulated attempts against DTW (report.json), then per sign (signs.json)."""
    run = Path(run) if run else latest_run(cfg)
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    p_dtw, dtw_refs = _matcher_references(cfg, ids)
    ref_feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    device = pick_device(str(cfg.train.device))
    model = _load_model(cfg, run, ref_feats[0].shape[-1], device)
    vocab = [line.split("#")[0].strip() for line in open(training_relative("configs/vocabulary_v1.txt")) if line.split("#")[0].strip()]
    index = {lab: k for k, lab in enumerate(labels)}
    chosen = [index[w] for w in vocab if w in index] + [index[w + " 1"] for w in vocab if w not in index and w + " 1" in index]
    rng = np.random.default_rng(seed)
    others = [int(k) for k in rng.choice(len(refs), 100, replace=False) if int(k) not in chosen]

    print("Evaluating on simulated attempts...")
    report = {
        "note": "one signer, simulated variation: robustness to augmentation, not accuracy on other people",
        "cut": "whole signs: an attempt ends when the hands drop (up to 10 s), not at a hold",
        "references": len(refs),
        "frames": {"recogniser": p.seq_len, "matcher": p_dtw.seq_len},
        "vocabulary": evaluate(model, labels, refs, ref_feats, device, p, dict(cfg.calibrate.augment), chosen, attempts, seed,
                               p_dtw, dtw_refs),
        "other_signs": evaluate(model, labels, refs, ref_feats, device, p, dict(cfg.calibrate.augment), others, 2, seed + 1,
                                p_dtw, dtw_refs),
        # Harder: the training augmentations (bigger rotations, more jitter) plus finger noise.
        "vocabulary_hard": evaluate(model, labels, refs, ref_feats, device, p,
                                    {**dict(cfg.train.augment), "finger_noise_rad": 0.15}, chosen, attempts, seed + 2,
                                    p_dtw, dtw_refs),
        # Viewpoint: the normal variation, seen by a camera up to 40 degrees off to the side and 30 above or below.
        "vocabulary_viewpoint": evaluate(model, labels, refs, ref_feats, device, p, {**dict(cfg.calibrate.augment), **VIEW_TEST},
                                         chosen, attempts, seed + 3, p_dtw, dtw_refs),
    }
    (run / "report.json").write_text(json.dumps(report, indent=1))
    for name in ("vocabulary", "other_signs", "vocabulary_hard", "vocabulary_viewpoint"):
        r = report[name]
        print(f"{name}: {r['attempts']} attempts ({r['segmentation_misses']} not cut) | among all {len(refs)} signs: embedding top-1 "
              f"{r['embedding_all_signs'].get('top1', 0):.0%} top-5 {r['embedding_all_signs'].get('top5', 0):.0%}, "
              f"DTW top-1 {r['dtw_all_signs'].get('top1', 0):.0%} top-5 {r['dtw_all_signs'].get('top5', 0):.0%}")
    print(f"-> {run}")
    sign_results(cfg, run)
    return run


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


PERSPECTIVE_YAW = (-60, -40, -20, 0, 20, 40, 60)  # camera to the signer's right (+) or left (-), degrees
PERSPECTIVE_PITCH = (-30, 0, 30)                  # camera below (-) or above (+)


def perspective_results(cfg: Config, run: Path | None = None, attempts: int = 1, seed: int = 5) -> Path:
    """How the recogniser holds up from fixed camera positions around the signer, wider than training saw (35 and
    25 degrees at most), and with a mirrored camera: every dictionary sign, `attempts` simulated attempts per
    position, recognised first among all signs. "Both ways" scores each attempt as it is and mirrored and keeps the
    better match, so a mirrored camera or a left-handed signer needs no setting. Simulated camera errors (depth and
    tilt along the line of sight), not hands hiding each other. Writes `perspectives.json` into the run folder."""
    from .canonical import mirror

    run = Path(run) if run else latest_run(cfg)
    if run is None:
        raise SystemExit("No embed run with a model yet: run `ashiria embed` first.")
    t0 = time.time()
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    p_dtw = whole_sign(EngineParams.from_config(cfg))
    feats_of = lambda s: sequence_features(s.hand_rel, s.wrist_pos, s.present, 1.0 / s.fps)
    device = pick_device(str(cfg.train.device))
    model = _load_model(cfg, run, feats_of(refs[0]).shape[-1], device)
    ref_emb = embed_all(model, [feats_of(r) for r in refs], device)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    base = {**dict(cfg.calibrate.augment), **VIEW_TEST}

    def attempts_at(aug: dict, k: int) -> tuple[np.ndarray, list[Segment]]:
        rng = np.random.default_rng((seed, k))
        owner, segs = [], []
        for i, r in enumerate(refs):
            for _ in range(attempts):
                a, _ = _simulate(r, rng, aug, p_sim, p_dtw)
                if a is not None:
                    owner.append(i)
                    segs.append(a)
        return np.array(owner), segs

    def first(owner: np.ndarray, segs: list[Segment], both: bool = False) -> float:
        sim = embed_all(model, [feats_of(s) for s in segs], device) @ ref_emb.T
        if both:
            sim = np.maximum(sim, embed_all(model, [feats_of(mirror(s)) for s in segs], device) @ ref_emb.T)
        return float((sim.argmax(1) == owner).mean())

    grid, k = {}, 0
    for pitch in PERSPECTIVE_PITCH:
        for yaw in PERSPECTIVE_YAW:
            owner, segs = attempts_at({**base, "view_yaw_at": yaw, "view_pitch_at": pitch}, k)
            grid[f"{yaw},{pitch}"] = {"yaw": yaw, "pitch": pitch, "tests": len(owner), "first": first(owner, segs)}
            print(f"yaw {yaw:+3d} pitch {pitch:+3d}: {grid[f'{yaw},{pitch}']['first']:.1%} of {len(owner)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            k += 1
    owner, segs = attempts_at({**base, "view_yaw_at": 0, "view_pitch_at": 0}, k)
    mirrored = [mirror(s) for s in segs]
    mirror_tests = {"as_is": first(owner, segs), "both_ways": first(owner, segs, both=True),
                    "mirrored_as_is": first(owner, mirrored), "mirrored_both_ways": first(owner, mirrored, both=True)}
    print("front view, as is {as_is:.1%}, both ways {both_ways:.1%}; mirrored camera or left-handed signer: as is "
          "{mirrored_as_is:.1%}, both ways {mirrored_both_ways:.1%}".format(**mirror_tests), flush=True)
    out = {"run": run.name, "signs": len(refs), "attempts_per_position": attempts, "grid": grid, "mirror": mirror_tests,
           "trained_up_to": {"yaw": VIEW_STRONG["view_yaw_deg"], "pitch": VIEW_STRONG["view_pitch_deg"]},
           "note": "simulated camera errors on one signer's videos; not occlusion, not other people"}
    (run / "perspectives.json").write_text(json.dumps(out, indent=1))
    print(f"-> {run / 'perspectives.json'} ({time.time() - t0:.0f} s)")
    return run / "perspectives.json"


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
    p_dtw, dtw_refs = _matcher_references(cfg, ids)
    feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    device = pick_device(str(cfg.train.device))
    model = _load_model(cfg, run, feats[0].shape[-1], device)
    trained_on = len(torch.load(run / "model.pt", map_location="cpu")["labels"])
    ref_emb = embed_all(model, feats, device)

    sim = ref_emb @ ref_emb.T
    np.fill_diagonal(sim, -np.inf)
    near = np.argsort(-sim, axis=1)[:, :closest]
    results = {ids[i]: {"closest": [[ids[j], round(float(sim[i, j]), 3)] for j in near[i]]} for i in range(len(refs))}

    vocab = [line.split("#")[0].strip() for line in open(training_relative("configs/vocabulary_v1.txt")) if line.split("#")[0].strip()]
    index = {lab: k for k, lab in enumerate(labels)}
    v1 = {index[w] if w in index else index.get(w + " 1") for w in vocab} - {None}
    ref_desc = np.stack([segment_descriptor(r, p_dtw.weights) for r in dtw_refs]).astype(np.float64)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    from .groups import KINDS, sign_kinds
    kinds = sign_kinds(ids)
    tests = {"normal": dict(cfg.calibrate.augment), "harder": {**dict(cfg.train.augment), "finger_noise_rad": 0.15},
             "viewpoint": {**dict(cfg.calibrate.augment), **VIEW_TEST}}
    summary = {}
    for k, (name, aug) in enumerate(tests.items()):
        rng = np.random.default_rng((seed, k))
        owner, att_feats, att_segs = [], [], []
        for i, r in enumerate(refs):
            for _ in range(attempts):
                a, a_dtw = _simulate(r, rng, aug, p_sim, p_dtw)
                if a is not None:
                    owner.append(i)
                    att_feats.append(sequence_features(a.hand_rel, a.wrist_pos, a.present, 1.0 / a.fps))
                    att_segs.append(a_dtw)
        top = (embed_all(model, att_feats, device) @ ref_emb.T).argmax(1)
        dtw_top = {}
        if name == "normal":
            for a, (i, seg) in enumerate(zip(owner, att_segs)):
                if i in v1:
                    dtw_top[a] = int(np.argmin(dtw_many(segment_descriptor(seg, p_dtw.weights).astype(np.float64), ref_desc, p_dtw.dtw_band)))
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
        kind_of = np.array([kinds[ids[i]]["kind"] for i in owner])
        by_kind = {k: round(float(hits[kind_of == k].mean()), 4) for k in KINDS if (kind_of == k).any()}
        summary[name] = {"tests": len(owner), "first": float(hits.mean()) if len(hits) else None, "by_kind": by_kind,
                         "signs_always_first": int(sum(results[c][name]["tests"] > 0 and
                                                       results[c][name]["first"] == results[c][name]["tests"] for c in ids))}
        if dtw_top:
            v1_hits = [top[a] == owner[a] for a in dtw_top]
            summary[name]["v1_first"] = float(np.mean(v1_hits))
            summary[name]["v1_dtw_first"] = float(np.mean([dtw_top[a] == owner[a] for a in dtw_top]))
        print(f"{name}: {len(owner)} attempts, recognised first {summary[name]['first']:.1%} ("
              + ", ".join(f"{k} {v:.1%}" for k, v in by_kind.items()) + f") ({time.time() - t0:.0f} s)", flush=True)

    report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
    out = {"run": run.name, "trained_on": trained_on, "signs": len(refs), "attempts_per_sign": attempts, "frames": p.seq_len,
           "note": "one signer, simulated variation: robustness to variation, not accuracy on other people",
           "summary": summary, "report": {k: report[k] for k in ("vocabulary", "other_signs", "vocabulary_hard", "vocabulary_viewpoint") if k in report},
           "results": results}
    (run / "signs.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {run / 'signs.json'}")
    return run / "signs.json"

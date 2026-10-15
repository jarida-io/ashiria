"""`ashiria export-recogniser`: the sign recogniser for the web app's Interpret mode (signer → text).

Writes exports/recogniser/ (git-ignored, like every export derived from the dictionary):
  model.onnx   the trained embedder: features [1, frames, D] -> unit embedding [1, 128]
  refs.f32     one embedding per dictionary sign, float32 little-endian [signs, 128]
  meta.json    sign names, engine parameters (whole-sign attempts), the acceptance thresholds and how they were set

Acceptance: the web app shows the nearest sign only when its similarity and its lead over the runner-up clear the
thresholds, otherwise "not sure". They are chosen on simulated attempts: in-vocabulary attempts must be right at
least 97% of the time when accepted, and attempts at a sign missing from the references (standing in for signs the
dictionary doesn't have) should rarely be accepted.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, load_config, path
from .embed import (DICTIONARY_CONFIG, VIDEO_REST_HEIGHT, _load_model, _simulate, embed_all, latest_run,
                    load_references, recogniser_params, sequence_features, whole_sign)
from .export import _to_onnx
from .params import EngineParams

PRECISION = 0.97          # accepted in-vocabulary answers must be right this often
MAX_UNKNOWN_ACCEPT = 0.15  # and attempts at signs it doesn't know accepted at most this often


def _names(ids: list[str]) -> list[dict]:
    """Display names per clip from the dictionary manifest (first gloss as the data sheet writes it, then aliases)."""
    dcfg = load_config(DICTIONARY_CONFIG)
    m = pd.read_csv(path(dcfg, "manifests") / "videos.csv", dtype=str).fillna("").set_index("clip_id")
    out = []
    for c in ids:
        aliases = [a.strip() for a in m.loc[c, "aliases"].split(";") if a.strip()] if c in m.index else []
        name = aliases[0] if aliases else (m.loc[c, "label"] if c in m.index else c)
        out.append({"clip": c, "name": name, "also": aliases[1:], "category": m.loc[c, "category"] if c in m.index else ""})
    return out


def _choose_thresholds(sims: np.ndarray, right: np.ndarray, unknown: np.ndarray) -> dict:
    """sims [n, 2]: best and second-best similarity of each in-vocabulary attempt; right [n]: was the best right;
    unknown [k, 2]: the same for attempts whose sign was left out. Highest coverage that meets both targets."""
    best = None
    for margin in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        for sim in np.arange(0.30, 0.96, 0.01):
            ok = (sims[:, 0] >= sim) & (sims[:, 0] - sims[:, 1] >= margin)
            if not ok.any():
                continue
            precision = right[ok].mean()
            unknown_accept = ((unknown[:, 0] >= sim) & (unknown[:, 0] - unknown[:, 1] >= margin)).mean()
            if precision >= PRECISION and unknown_accept <= MAX_UNKNOWN_ACCEPT:
                coverage = ok.mean()
                if best is None or coverage > best["coverage"]:
                    best = {"sim": round(float(sim), 2), "margin": margin, "coverage": float(coverage),
                            "precision": float(precision), "unknown_accepted": float(unknown_accept)}
    return best or {"sim": 0.9, "margin": 0.05, "coverage": None, "precision": None, "unknown_accepted": None}


def export_recogniser(cfg: Config, run: str | Path | None = None, sample: int = 900, seed: int = 21) -> Path:
    run = Path(run) if run else latest_run(cfg)
    if run is None:
        raise SystemExit("No embed run with a model yet: run `ashiria embed` first.")
    t0 = time.time()
    p = recogniser_params(EngineParams.from_config(cfg))
    labels, refs, ids = load_references(p)
    feats = [sequence_features(r.hand_rel, r.wrist_pos, r.present, 1.0 / r.fps) for r in refs]
    device = torch.device("cpu")
    model = _load_model(cfg, run, feats[0].shape[-1], device)
    ref_emb = embed_all(model, feats, device).astype(np.float32)

    out = path(cfg, "exports") / "recogniser"
    out.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(feats[0][None].astype(np.float32))
    _to_onnx(model, dummy, out / "model.onnx")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"features": dummy.numpy()})[0][0]
    drift = float(np.abs(got - ref_emb[0]).max())
    if drift > 1e-4:
        raise SystemExit(f"ONNX model disagrees with PyTorch by {drift:.2e}")
    ref_emb.astype("<f4").tofile(out / "refs.f32")

    # Thresholds from simulated attempts at a sample of signs (normal variation), each also scored against the
    # references with its own sign removed.
    rng = np.random.default_rng(seed)
    p_dtw = whole_sign(EngineParams.from_config(cfg))
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    chosen = rng.choice(len(refs), min(sample, len(refs)), replace=False)
    owners, att_feats = [], []
    for i in chosen:
        a, _ = _simulate(refs[i], rng, dict(cfg.calibrate.augment), p_sim, p_dtw)
        if a is not None:
            owners.append(int(i))
            att_feats.append(sequence_features(a.hand_rel, a.wrist_pos, a.present, 1.0 / a.fps))
    S = embed_all(model, att_feats, device) @ ref_emb.T
    owners = np.array(owners)
    order = np.argsort(-S, axis=1)
    sims = np.take_along_axis(S, order[:, :2], axis=1)
    right = order[:, 0] == owners
    S_unknown = S.copy()
    S_unknown[np.arange(len(owners)), owners] = -np.inf
    unknown = -np.sort(-S_unknown, axis=1)[:, :2]
    accept = _choose_thresholds(sims, right, unknown)

    meta = {
        "format": "ashiria.recogniser", "version": 1, "run": run.name, "created": time.strftime("%Y-%m-%d %H:%M"),
        "frames": p.seq_len, "feature_dim": int(feats[0].shape[-1]), "embedding_dim": int(ref_emb.shape[1]),
        "engine": p.to_json(), "accept": accept, "signs": _names(ids),
        # Hands below this (shoulder widths under the nose) are resting: the dictionary videos' value, a fair start
        # for a webcam (same view of the signer); the web app lets people set their own.
        "rest_height": VIDEO_REST_HEIGHT,
        "note": "Single signs from one signer's videos (KSL Dictionary, used with permission): not sentences, and "
                "untested on other people. Dev only: never shipped without Jerry's go-ahead.",
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"{len(ids)} signs, {size:.1f} MB -> {out} ({time.time() - t0:.0f} s)")
    print(f"Accept when similarity >= {accept['sim']} and lead >= {accept['margin']}: answers {accept['coverage']:.0%} "
          f"of attempts, right {accept['precision']:.1%} of those; unknown signs accepted {accept['unknown_accepted']:.0%}")
    return out

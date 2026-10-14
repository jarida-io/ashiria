"""Step 7: ONNX model for Unity Sentis, plus metadata and golden vectors for the C# parity tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, path
from .data import Cache
from .evaluate import load_model
from .features import SPEC_VERSION

OPSET = 17
PARITY_TOLERANCE = 1e-3
GOLDEN_CLIPS = 6


def _to_onnx(model: torch.nn.Module, dummy: torch.Tensor, file: Path) -> None:
    kwargs = {"input_names": ["features"], "output_names": ["logits"], "opset_version": OPSET}
    try:
        torch.onnx.export(model, dummy, str(file), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(model, dummy, str(file), **kwargs)


def export(cfg: Config, run_dir: str | Path, name: str | None = None) -> Path:
    import onnx
    import onnxruntime as ort

    run = Path(run_dir)
    model, ckpt = load_model(run)
    seq_len, dim = ckpt["seq_len"], ckpt["model_args"]["in_dim"]
    out = path(cfg, "exports") / (name or run.name)
    out.mkdir(parents=True, exist_ok=True)
    onnx_file = out / "model.onnx"
    _to_onnx(model, torch.zeros(1, seq_len, dim), onnx_file)
    onnx.checker.check_model(onnx.load(str(onnx_file)))

    cache = Cache(cfg)
    if cache.classes != ckpt["classes"]:
        raise SystemExit("The label list changed since this model was trained.")
    sample = np.flatnonzero(cache.split == "test")
    if not len(sample):
        sample = np.arange(len(cache))
    sample = sample[:64]

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    worst, golden = 0.0, []
    with torch.no_grad():
        for k, i in enumerate(sample):
            x = cache.features(int(i))[None]
            expected = model(torch.from_numpy(x)).numpy()
            got = session.run(["logits"], {"features": x})[0]
            worst = max(worst, float(np.abs(expected - got).max()))
            if k < GOLDEN_CLIPS:
                seg = cache.segment(int(i))
                golden.append({
                    "clip_id": cache.clip_ids[i],
                    "label": cache.classes[cache.y[i]],
                    "segment": {"fps": seg.fps, "hand_rel": seg.hand_rel.tolist(),
                                "wrist_pos": seg.wrist_pos.tolist(), "present": seg.present.tolist()},
                    "features": x[0].tolist(),
                    "logits": expected[0].tolist(),
                })
    if worst > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output differs from PyTorch by {worst:.2e} (limit {PARITY_TOLERANCE}).")

    evaluation = json.loads((run / "eval.json").read_text()) if (run / "eval.json").exists() else {}
    meta = {
        "name": out.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run": run.name,
        "spec_version": SPEC_VERSION,
        "seq_len": seq_len,
        "feature_dim": dim,
        "input": "features",
        "output": "logits",
        "onnx_opset": OPSET,
        "classes": ckpt["classes"],
        "temperature": evaluation.get("temperature", 1.0),
        "test_top1": evaluation.get("test_top1"),
        "per_class": evaluation.get("per_class", {}),
        "parity_max_abs_diff": worst,
        "size_mb": round(onnx_file.stat().st_size / 1e6, 2),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "labels.json").write_text(json.dumps(ckpt["classes"], indent=2))
    (out / "golden.json").write_text(json.dumps({"spec_version": SPEC_VERSION, "seq_len": seq_len,
                                                 "tolerance": {"features": 1e-3, "logits": 5e-3}, "clips": golden}))
    print(f"Exported {onnx_file} ({meta['size_mb']} MB), max ONNX/PyTorch difference {worst:.1e}")
    return out

"""Step 6: test-set accuracy, per-sign results, common confusions and a confidence temperature."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import Cache
from .model import SignNet


def load_model(run: Path) -> tuple[SignNet, dict]:
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = SignNet(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model: SignNet, cache: Cache, indices: np.ndarray, batch: int = 256) -> np.ndarray:
    out = [model(torch.from_numpy(np.stack([cache.features(i) for i in indices[k : k + batch]]))).numpy()
           for k in range(0, len(indices), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(cache.classes)), np.float32)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that makes softmax(logits / T) best calibrated on held-out clips."""
    grid = np.linspace(0.5, 5.0, 46)
    nll = [-log_softmax(logits / t)[np.arange(len(y)), y].mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def evaluate(cfg: Config, run_dir: str | Path) -> dict:
    run = Path(run_dir)
    model, ckpt = load_model(run)
    cache = Cache(cfg)
    if cache.classes != ckpt["classes"]:
        raise SystemExit("The label list changed since this model was trained. Retrain or rebuild the matching cache.")

    va, te = np.flatnonzero(cache.split == "val"), np.flatnonzero(cache.split == "test")
    temperature = fit_temperature(predict(model, cache, va), cache.y[va]) if len(va) else 1.0
    if not len(te):
        raise SystemExit("Test split is empty.")
    logits, y = predict(model, cache, te), cache.y[te]
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    classes = cache.classes

    per_class = {}
    for c, name in enumerate(classes):
        mask = y == c
        if mask.any():
            per_class[name] = {"n": int(mask.sum()), "acc": round(float((pred[mask] == c).mean()), 3)}
    confusions = Counter((classes[a], classes[b]) for a, b in zip(y, pred) if a != b).most_common(20)
    result = {
        "test_top1": round(float((pred == y).mean()), 4),
        "test_top5": round(float((top5 == y[:, None]).any(1).mean()), 4),
        "n_test": int(len(te)),
        "temperature": temperature,
        "per_class": per_class,
        "confusions": [{"true": a, "predicted": b, "count": n} for (a, b), n in confusions],
    }
    (run / "eval.json").write_text(json.dumps(result, indent=2))

    matrix = pd.crosstab(pd.Series([classes[i] for i in y], name="true"),
                         pd.Series([classes[i] for i in pred], name="predicted"))
    matrix.to_csv(run / "confusion.csv")

    worst = sorted(per_class.items(), key=lambda kv: kv[1]["acc"])[:15]
    lines = [
        f"# Evaluation: {run.name}",
        "",
        f"- Test clips: {len(te)}; top-1 {result['test_top1']:.1%}; top-5 {result['test_top5']:.1%}",
        f"- Temperature: {temperature:.2f}",
        "",
        "## Weakest signs",
        "",
        "| Sign | Clips | Accuracy |",
        "|---|---|---|",
        *[f"| {k} | {v['n']} | {v['acc']:.0%} |" for k, v in worst],
        "",
        "## Most common confusions",
        "",
        "| True | Predicted | Count |",
        "|---|---|---|",
        *[f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in result["confusions"]],
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    return result

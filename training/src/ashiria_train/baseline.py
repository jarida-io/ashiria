"""Step 4: nearest-neighbour baselines. If these score badly, check the labels and extraction first."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .config import Config
from .data import Cache


def baseline(cfg: Config) -> dict:
    cache = Cache(cfg)
    X = np.stack([cache.features(i) for i in range(len(cache))])  # [N, T, F]
    train = cache.split == "train"
    views = {
        "whole sequence, 1-NN": X.reshape(len(X), -1),
        "mean + std over time, 5-NN": np.concatenate([X.mean(1), X.std(1)], axis=1),
    }
    results = {}
    for name, Z in views.items():
        mu, sd = Z[train].mean(0), Z[train].std(0) + 1e-6
        Z = (Z - mu) / sd
        knn = KNeighborsClassifier(n_neighbors=1 if "1-NN" in name else 5, metric="cosine")
        knn.fit(Z[train], cache.y[train])
        for split in ("val", "test"):
            mask = cache.split == split
            if mask.any():
                results[(name, split)] = float((knn.predict(Z[mask]) == cache.y[mask]).mean())
    chance = 1 / len(cache.classes)
    print(f"{len(cache.classes)} labels, chance {chance:.1%}")
    for (name, split), acc in results.items():
        print(f"{name:28s} {split:5s} {acc:.1%}")
    return results

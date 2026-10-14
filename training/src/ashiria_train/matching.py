"""Reference matching: the v1 engine for Verify ("is this sign X?") and Recognise ("which sign is this?").

This is the reference implementation that the C# engine ports. It needs no training data: an
attempt is compared with each sign's reference recording.

- Per-frame descriptor: a weighted subset of the spec features (finger bends, spreads, fingertip
  distances, palm orientation, wrist position, presence) for both hands.
- Distance: dynamic time warping over the two resampled sequences, so speed differences don't count.
- Per-finger errors at the steadiest frame drive the feedback ("bend your ring finger").
"""

from __future__ import annotations

import numpy as np

from .features import HAND_DIM, sequence_features
from .params import DEFAULT_WEIGHTS

# Feature slices inside one hand block (spec §6) used by the descriptor. Weights live in params.py.
SLICES = {
    "flexion": slice(60, 75),
    "spread": slice(75, 79),
    "tips": slice(79, 84),
    "palm_normal": slice(84, 87),
    "direction": slice(87, 90),
    "wrist": slice(90, 93),
    "present": slice(96, 97),
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def descriptor(features: np.ndarray, weights: dict | None = None) -> np.ndarray:
    """[T, 197] spec features -> [T, 68] weighted matching descriptor for both hands."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    parts = []
    for offset in (0, HAND_DIM):
        for name, sl in SLICES.items():
            parts.append(features[:, offset + sl.start : offset + sl.stop] * weights[name])
    return np.concatenate(parts, axis=1)


def segment_features(seg) -> np.ndarray:
    return sequence_features(seg.hand_rel, seg.wrist_pos, seg.present, 1.0 / seg.fps)


def segment_descriptor(seg, weights: dict | None = None) -> np.ndarray:
    return descriptor(segment_features(seg), weights)


def _cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean distances in float64 (as the C# port computes them)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def _accumulate(cost: np.ndarray, band: int) -> np.ndarray:
    n, m = cost.shape
    band = max(band, abs(n - m))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return acc


def dtw(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Dynamic time warping distance with a Sakoe-Chiba band, normalised by n + m."""
    acc = _accumulate(_cost(a, b), band)
    return float(acc[-1, -1] / (len(a) + len(b)))


def dtw_path(a: np.ndarray, b: np.ndarray, band: int = 8) -> tuple[float, list[tuple[int, int]]]:
    """DTW distance and the alignment [(i, j), ...] from (0, 0) to (n-1, m-1).
    Backtracking prefers the diagonal, then (i-1, j), then (i, j-1) on ties, so ports agree exactly."""
    acc = _accumulate(_cost(a, b), band)
    i, j = len(a), len(b)
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        steps = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        i, j = min(steps, key=lambda s: acc[s])  # min keeps the first of equal values
        path.append((i - 1, j - 1))
    return float(acc[-1, -1] / (len(a) + len(b))), path[::-1]


def dtw_many(a: np.ndarray, bs: np.ndarray, band: int = 8) -> np.ndarray:
    """`dtw(a, b)` for every b in bs [B, m, D] at once (same result, vectorised over B). Used by calibration."""
    n, m = len(a), bs.shape[1]
    band = max(band, abs(n - m))
    a, bs = np.asarray(a, np.float64), np.asarray(bs, np.float64)
    cost = np.sqrt(((a[None, :, None, :] - bs[:, None, :, :]) ** 2).sum(-1))  # [B, n, m]
    acc = np.full((len(bs), n + 1, m + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            acc[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]),
                                                              acc[:, i - 1, j - 1])
    return acc[:, n, m] / (n + m)


def distance_matrix(descriptors: list[np.ndarray], band: int = 8) -> np.ndarray:
    """Symmetric DTW distances between all pairs; the diagonal is 0."""
    n = len(descriptors)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = dtw(descriptors[i], descriptors[j], band)
    return dist


def finger_errors(attempt_flexion: np.ndarray, reference_flexion: np.ndarray) -> np.ndarray:
    """Largest bend difference per finger (radians, signed: + means bend more), thumb first. [5]"""
    diff = np.asarray(reference_flexion, np.float64).reshape(5, 3) - np.asarray(attempt_flexion, np.float64).reshape(5, 3)
    return np.take_along_axis(diff, np.abs(diff).argmax(axis=1)[:, None], axis=1)[:, 0]


def nearest(names: list[str], dist: np.ndarray, top: int = 3) -> list[dict]:
    """For each sign, its nearest other signs: [{sign, near: [(name, distance), ...]}], closest pairs first."""
    rows = []
    for i, name in enumerate(names):
        d = dist[i].copy()
        d[i] = np.inf
        order = [j for j in np.argsort(d, kind="stable")[:top] if np.isfinite(d[j])]
        rows.append({"sign": name, "near": [(names[j], float(d[j])) for j in order]})
    return sorted(rows, key=lambda r: r["near"][0][1] if r["near"] else np.inf)


def lookalike_table(rows: list[dict]) -> list[dict]:
    """`nearest` rows flattened for a CSV report."""
    return [{"sign": r["sign"], **{f"near_{k + 1}": f"{n} ({d:.2f})" for k, (n, d) in enumerate(r["near"])},
             "nearest_distance": round(r["near"][0][1], 3) if r["near"] else None} for r in rows]

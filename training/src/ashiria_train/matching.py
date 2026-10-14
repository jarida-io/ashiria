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

# Feature slices inside one hand block (spec §6) and their weights in the descriptor.
# Provisional (14 Oct 2026): wrist 3.0 and orientation 1.0 raised simulated top-1 on the 72-sign vocabulary
# from 82% to 90% (`ashiria simulate`: one signer, synthetic variation). Re-tune on real Quest attempts.
BLOCK = {
    "flexion": (slice(60, 75), 1.0),
    "spread": (slice(75, 79), 1.0),
    "tips": (slice(79, 84), 0.5),
    "palm_normal": (slice(84, 87), 1.0),
    "direction": (slice(87, 90), 1.0),
    "wrist": (slice(90, 93), 3.0),
    "present": (slice(96, 97), 1.0),
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def descriptor(features: np.ndarray) -> np.ndarray:
    """[T, 197] spec features -> [T, D] weighted matching descriptor for both hands."""
    parts = []
    for offset in (0, HAND_DIM):
        for sl, weight in BLOCK.values():
            parts.append(features[:, offset + sl.start : offset + sl.stop] * weight)
    return np.concatenate(parts, axis=1)


def segment_descriptor(seg) -> np.ndarray:
    return descriptor(sequence_features(seg.hand_rel, seg.wrist_pos, seg.present, 1.0 / seg.fps))


def dtw(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Dynamic time warping distance with a Sakoe-Chiba band, normalised by path length."""
    n, m = len(a), len(b)
    band = max(band, abs(n - m))
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def finger_errors(attempt_flexion: np.ndarray, reference_flexion: np.ndarray) -> dict[str, float]:
    """Largest bend difference (radians, signed: + means bend more) per finger."""
    diff = reference_flexion.reshape(5, 3) - attempt_flexion.reshape(5, 3)
    worst = np.take_along_axis(diff, np.abs(diff).argmax(axis=1)[:, None], axis=1)[:, 0]
    return dict(zip(FINGERS, worst.round(3).tolist()))


def lookalikes(names: list[str], descriptors: list[np.ndarray], top: int = 3) -> list[dict]:
    """For each sign, its nearest other signs by DTW distance: candidates for confusion."""
    n = len(names)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = dtw(descriptors[i], descriptors[j])
    rows = []
    for i in range(n):
        order = np.argsort(dist[i])[:top]
        rows.append({"sign": names[i], **{f"near_{k + 1}": f"{names[j]} ({dist[i, j]:.2f})" for k, j in enumerate(order)},
                     "nearest_distance": round(float(dist[i, order[0]]), 3) if n > 1 else None})
    return sorted(rows, key=lambda r: r["nearest_distance"] if r["nearest_distance"] is not None else np.inf)

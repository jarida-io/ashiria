"""Synthetic hands and segments: rights-free inputs for tests and cross-language golden vectors."""

from __future__ import annotations

import numpy as np

from . import joints as J
from .augment import rotation
from .canonical import Segment


def open_hand() -> np.ndarray:
    """A flat right hand in metres, wrist at the origin, fingers along +y, palm facing +z."""
    pts = np.zeros((21, 3))
    knuckles = {J.THUMB: (-0.03, 0.02), J.INDEX: (-0.02, 0.09), J.MIDDLE: (0.0, 0.095),
                J.RING: (0.02, 0.09), J.PINKY: (0.038, 0.08)}
    for chain, (x, y) in knuckles.items():
        direction = np.array([x, y, 0.0]) / np.hypot(x, y) if chain is J.THUMB else np.array([x * 0.15, 1.0, 0.0])
        direction /= np.linalg.norm(direction)
        pts[chain[0]] = [x, y, 0.0]
        for k in range(1, 4):
            pts[chain[k]] = pts[chain[k - 1]] + direction * 0.025
    return pts


def _rotate_about(points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    return (points - pivot) @ r.T + pivot


def bent_hand(bends: np.ndarray) -> np.ndarray:
    """Open hand with each finger bent at its three joints by bends[5, 3] radians (towards the palm)."""
    h = open_hand()
    lateral = h[J.PINKY[0]] - h[J.INDEX[0]]
    for f, chain in enumerate(J.FINGERS):
        for k in range(3):
            h[list(chain[k + 1 :])] = _rotate_about(h[list(chain[k + 1 :])], h[chain[k]], lateral, bends[f, k])
    return h


def random_segment(rng: np.random.Generator, frames: int, two_hands: bool, moving: bool, fps: float = 30.0,
                   gaps: tuple[tuple[int, int, int], ...] = (), rest_frames: int = 0) -> Segment:
    """A plausible segment: handshapes blend from a start to an end pose, wrists travel, chosen frames drop out.
    gaps = (slot, start, length) dropouts; rest_frames at each end put the hands below signing space."""
    hand_rel = np.zeros((frames, 2, 21, 3))
    wrist = np.zeros((frames, 2, 3))
    present = np.zeros((frames, 2))
    for slot in range(2 if two_hands else 1):
        start_b, end_b = rng.uniform(0, 1.3, (5, 3)), rng.uniform(0, 1.3, (5, 3))
        if not moving:
            end_b = start_b
        orient = rotation(*rng.uniform(-50, 50, 3))
        w0 = np.array([0.4 if slot == 0 else -0.4, -1.0, 0.6]) + rng.uniform(-0.2, 0.2, 3)
        w1 = w0 + (rng.uniform(-0.6, 0.6, 3) if moving else 0)
        for t in range(frames):
            a = t / max(frames - 1, 1)
            h = bent_hand(start_b * (1 - a) + end_b * a) @ orient.T
            if slot == 1:
                h = h * [-1, 1, 1]  # a left hand
            hand_rel[t, slot] = h - h[0]
            wrist[t, slot] = w0 * (1 - a) + w1 * a + [0, 0.05 * np.sin(6 * a), 0]
            present[t, slot] = 1.0
    for slot, start, length in gaps:
        present[start : start + length, slot] = 0.0
    if rest_frames:
        wrist[:rest_frames, :, 1] = -2.6
        wrist[-rest_frames:, :, 1] = -2.6
    hand_rel *= present[:, :, None, None]
    return Segment(hand_rel, wrist, present, fps)

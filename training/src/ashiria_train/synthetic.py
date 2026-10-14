"""Synthetic hands and segments: rights-free inputs for tests and cross-language golden vectors."""

from __future__ import annotations

import numpy as np

from . import joints as J
from .augment import bend_fingers, rotation
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


def bent_hand(bends: np.ndarray) -> np.ndarray:
    """Open hand with each finger bent at its three joints by bends[5, 3] radians (towards the palm)."""
    return bend_fingers(open_hand(), np.asarray(bends, float))


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


def sign_stream(bends: np.ndarray, orient: np.ndarray | None = None, at=(0.4, -1.0, 0.6), path=None,
                rest_frames: int = 10, rise: int = 8, hold: int = 24, drop: int = 8, after: int = 10,
                fps: float = 30.0, rest_y: float = -2.4, other_at=None, lost: tuple[tuple[int, int], ...] = ()) -> Segment:
    """A live-style stream of one sign: hands at rest, rise to `at`, hold (or move by `path` while holding the
    handshape), drop back to rest. `other_at` adds the other hand (mirrored shape) doing the same.
    `lost` = (start, length) spans where the dominant hand is not tracked."""
    orient = np.eye(3) if orient is None else orient
    at = np.asarray(at, float)
    end = at + (np.zeros(3) if path is None else np.asarray(path, float))
    rest = np.array([at[0], rest_y, at[2]])
    track = ([rest] * rest_frames + [rest + (at - rest) * (k + 1) / rise for k in range(rise)]
             + [at + (end - at) * k / max(hold - 1, 1) for k in range(hold)]
             + [end + (rest - end) * (k + 1) / drop for k in range(drop)] + [rest] * after)
    n = len(track)
    hand = bent_hand(np.asarray(bends, float).reshape(5, 3)) @ orient.T
    hand_rel = np.zeros((n, 2, 21, 3))
    wrist = np.zeros((n, 2, 3))
    present = np.zeros((n, 2))
    hand_rel[:, 0] = hand - hand[0]
    wrist[:, 0] = track
    present[:, 0] = 1.0
    if other_at is not None:
        offset = np.asarray(other_at, float) - at
        hand_rel[:, 1] = (hand - hand[0]) * [-1, 1, 1]
        wrist[:, 1] = np.asarray(track) + offset * (np.asarray(track)[:, 1:2] > rest_y + 1e-9)
        present[:, 1] = 1.0
    for start, length in lost:
        present[start : start + length, 0] = 0.0
    hand_rel *= present[:, :, None, None]
    return Segment(hand_rel, wrist, present, fps)

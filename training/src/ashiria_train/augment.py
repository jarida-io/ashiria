"""Training-time augmentation of canonical segments.

Each transform imitates a way headset data differs from the training videos:
a different viewpoint and body, tracking jitter, dropped frames, a lost hand,
signing faster or slower, and imperfect start/end detection.
"""

from __future__ import annotations

import numpy as np

from . import joints as J
from .canonical import Segment


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def rotate_about(points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points [..., 3] by `angle` radians about the line through `pivot` along `axis`."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    return (points - pivot) @ r.T + pivot


def bend_fingers(hand: np.ndarray, bends: np.ndarray) -> np.ndarray:
    """Bend each finger of one hand [21, 3] at its three joints by bends [5, 3] radians, about the palm's
    lateral axis (index knuckle to pinky knuckle). Positive bends towards the palm for a right hand."""
    h = hand.copy()
    lateral = h[J.PINKY[0]] - h[J.INDEX[0]]
    if np.linalg.norm(lateral) < 1e-9:
        return h
    for f, chain in enumerate(J.FINGERS):
        for k in range(3):
            rest = list(chain[k + 1 :])
            h[rest] = rotate_about(h[rest], h[chain[k]], lateral, bends[f, k])
    return h


def _random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    yaw, pitch, roll = rng.uniform(-max_deg, max_deg), rng.uniform(-max_deg, max_deg) / 2, rng.uniform(-max_deg, max_deg) / 2
    return rotation(yaw, pitch, roll)


def augment(seg: Segment, rng: np.random.Generator, a) -> Segment:
    length = len(seg)
    crop = int(length * a["edge_crop"])
    if crop > 0 and length - 2 * crop >= 4:
        seg = seg.slice(int(rng.integers(0, crop + 1)), length - int(rng.integers(0, crop + 1)))

    body = _random_rotation(rng, a["rotate_deg"])
    hand_rel = np.einsum("ij,tskj->tski", body, seg.hand_rel)
    wrist_pos = np.einsum("ij,tsj->tsi", body, seg.wrist_pos)
    for s in range(2):
        hand_rel[:, s] = np.einsum("ij,tkj->tki", _random_rotation(rng, a["hand_rotate_deg"]), hand_rel[:, s])

    wrist_pos = wrist_pos * rng.uniform(1 - a["wrist_scale"], 1 + a["wrist_scale"])
    wrist_pos = wrist_pos + rng.uniform(-a["wrist_shift"], a["wrist_shift"], size=3)
    finger_noise = a.get("finger_noise_rad", 0.0)
    if finger_noise:  # someone else's handshape: each finger bent a little differently for the whole attempt
        for s in range(2):
            bends = rng.normal(0.0, finger_noise, (5, 3))
            hand_rel[:, s] = np.stack([bend_fingers(h, bends) for h in hand_rel[:, s]])
    hand_rel = hand_rel + rng.normal(0.0, a["joint_noise_m"], size=hand_rel.shape)
    hand_rel = hand_rel - hand_rel[:, :, :1, :]

    present = seg.present.copy()
    present[rng.random(present.shape) < a["frame_drop"]] = 0.0
    if rng.random() < a["hand_drop"] and len(present) >= 4:
        span = int(len(present) * rng.uniform(0.2, 0.5))
        start = int(rng.integers(0, len(present) - span + 1))
        present[start : start + span, int(rng.integers(0, 2))] = 0.0

    fps = seg.fps * rng.uniform(*a["speed"])
    return Segment(hand_rel * present[:, :, None, None], wrist_pos, present, fps)

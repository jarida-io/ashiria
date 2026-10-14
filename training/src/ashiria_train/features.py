"""Per-frame features: the reference implementation of spec/FEATURE_SPEC.md §6.

The headset's C# code must produce the same numbers. `ashiria export` writes golden
vectors (golden.json) that the Unity tests compare against.
"""

from __future__ import annotations

import numpy as np

from . import joints as J

SPEC_VERSION = 1
EPS = 1e-6
HAND_DIM = 97
FEATURE_DIM = 2 * HAND_DIM + 3
TIP_PAIRS = ((4, 8), (4, 12), (4, 16), (4, 20), (8, 12))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS)


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle in radians via atan2(|u×v|, u·v): stable near 0 and π, even in 32-bit floats."""
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1))


def hand_axes(hand_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand-local axes for joints [..., 21, 3] with the wrist at the origin.

    Returns R [..., 3, 3] whose rows are the hand's x, y (wrist to middle knuckle) and
    z (palm normal) axes expressed in the input frame, and the palm length [...].
    """
    y = hand_rel[..., J.MIDDLE[0], :] - hand_rel[..., J.WRIST, :]
    palm = np.linalg.norm(y, axis=-1)
    y = normalize(y)
    n = np.cross(
        hand_rel[..., J.INDEX[0], :] - hand_rel[..., J.WRIST, :],
        hand_rel[..., J.PINKY[0], :] - hand_rel[..., J.WRIST, :],
    )
    x = normalize(np.cross(y, n))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-2), palm


def flexion(hand_rel: np.ndarray) -> np.ndarray:
    """15 bend angles in radians, three per finger, thumb first. 0 means straight."""
    out = []
    for chain in J.FINGERS:
        pts = (J.WRIST,) + chain
        for k in range(1, 4):
            a = hand_rel[..., pts[k], :] - hand_rel[..., pts[k - 1], :]
            b = hand_rel[..., pts[k + 1], :] - hand_rel[..., pts[k], :]
            out.append(angle_between(a, b))
    return np.stack(out, axis=-1)


def spread(hand_rel: np.ndarray) -> np.ndarray:
    """4 angles between neighbouring fingers' first bones: thumb-index ... ring-pinky."""
    first = [hand_rel[..., c[1], :] - hand_rel[..., c[0], :] for c in J.FINGERS]
    return np.stack([angle_between(first[i], first[i + 1]) for i in range(4)], axis=-1)


def tip_distances(hand_rel: np.ndarray, palm: np.ndarray) -> np.ndarray:
    """5 fingertip distances (thumb to each finger, index to middle) in palm lengths."""
    d = [np.linalg.norm(hand_rel[..., a, :] - hand_rel[..., b, :], axis=-1) for a, b in TIP_PAIRS]
    return np.stack(d, axis=-1) / np.maximum(palm[..., None], EPS)


def wrist_velocity(wrist_pos: np.ndarray, present: np.ndarray, dt: float) -> np.ndarray:
    """Backward difference in shoulder widths per second; 0 unless both frames have the hand."""
    vel = np.zeros_like(wrist_pos)
    both = present[1:] * present[:-1]
    vel[1:] = (wrist_pos[1:] - wrist_pos[:-1]) / dt * both[:, None]
    return vel


def hand_features(hand_rel, wrist_pos, wrist_vel, present) -> np.ndarray:
    """One hand slot: [T,21,3], [T,3], [T,3], [T] -> [T, HAND_DIM]. All zeros where absent."""
    t = len(hand_rel)
    R, palm = hand_axes(hand_rel)
    local = np.einsum("tij,tkj->tki", R, hand_rel) / np.maximum(palm, EPS)[:, None, None]
    f = np.concatenate(
        [
            local[:, 1:, :].reshape(t, 60),  # 0:60   joints 1..20 in the hand's own frame
            flexion(hand_rel),               # 60:75
            spread(hand_rel),                # 75:79
            tip_distances(hand_rel, palm),   # 79:84
            R[:, 2, :],                      # 84:87  palm normal, reference frame
            R[:, 1, :],                      # 87:90  hand direction, reference frame
            wrist_pos,                       # 90:93  shoulder widths from the nose
            wrist_vel,                       # 93:96  shoulder widths per second
            np.ones((t, 1)),                 # 96     present
        ],
        axis=-1,
    )
    return f * present[:, None]


def sequence_features(hand_rel, wrist_pos, present, dt: float) -> np.ndarray:
    """Features for a resampled, canonical segment.

    hand_rel [T,2,21,3] metres in reference-frame axes, wrist at the origin;
    wrist_pos [T,2,3] shoulder widths; present [T,2] in {0,1}; dt seconds per frame.
    Returns float32 [T, FEATURE_DIM]: dominant hand, other hand, other-minus-dominant wrist.
    """
    hand_rel = np.asarray(hand_rel, np.float64)
    wrist_pos = np.asarray(wrist_pos, np.float64)
    present = (np.asarray(present) > 0.5).astype(np.float64)
    blocks = []
    for s in (J.RIGHT, J.LEFT):
        vel = wrist_velocity(wrist_pos[:, s], present[:, s], dt)
        blocks.append(hand_features(hand_rel[:, s], wrist_pos[:, s], vel, present[:, s]))
    both = (present[:, J.RIGHT] * present[:, J.LEFT])[:, None]
    blocks.append((wrist_pos[:, J.LEFT] - wrist_pos[:, J.RIGHT]) * both)
    return np.concatenate(blocks, axis=-1).astype(np.float32)

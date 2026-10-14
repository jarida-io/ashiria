"""Quest hand tracking -> canonical frames: the Python twin of sdk/.../Runtime/QuestCanonical.cs (spec §4.2).

Inputs are Unity world space in metres (left-handed: x right, y up, z forward): the centre-eye pose
and the 26 OpenXR hand joints per hand. The reference frame follows the head's yaw only (y = world up),
with its origin moved from the eyes to roughly the nose tip, where the video frame's origin is.
"""

from __future__ import annotations

import numpy as np

from . import joints as J

DEFAULT_SHOULDER_WIDTH_M = 0.38
# Nose tip relative to the centre eye, in head-yaw axes (x right, y up, z forward), metres. An average-
# adult approximation so headset wrist positions line up with the video references' nose origin.
NOSE_OFFSET_M = np.array([0.0, -0.045, 0.025])
UP = np.array([0.0, 1.0, 0.0])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v [..., 3] by unit quaternion q = (x, y, z, w), as Unity's `q * v` does."""
    u, w = np.asarray(q[:3], float), float(q[3])
    v = np.asarray(v, float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def head_axes(head_forward: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Rows x, y, z of the head-yaw frame; (None, False) when looking almost straight up or down."""
    flat = np.array([head_forward[0], 0.0, head_forward[2]], float)
    norm = np.linalg.norm(flat)
    if norm <= 1e-3:
        return None, False
    z = flat / norm
    x = np.cross(UP, z)
    return np.stack([x, UP, z]), True


def origin(head_pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.asarray(head_pos, float) + NOSE_OFFSET_M @ R


def hand(joints26: np.ndarray, head_pos: np.ndarray, R: np.ndarray, shoulder_width: float, mirror: bool
         ) -> tuple[np.ndarray, np.ndarray]:
    """One hand: 26 OpenXR joint positions -> hand_rel [21,3] (metres, reference axes, wrist at the origin)
    and wrist_pos [3] (shoulder widths from the nose). `mirror` for left-handed users."""
    joints26 = np.asarray(joints26, float)
    pts = joints26[J.QUEST_INDEX]
    hand_rel = (pts - pts[J.WRIST]) @ R.T
    wrist_pos = R @ (pts[J.WRIST] - origin(head_pos, R)) / shoulder_width
    if mirror:
        hand_rel = hand_rel * [-1, 1, 1]
        wrist_pos = wrist_pos * [-1, 1, 1]
    return hand_rel, wrist_pos


def slot_for(is_left_hand: bool, left_handed: bool) -> int:
    """Right-handed users: right hand in slot 0. Left-handed users: the (mirrored) left hand in slot 0."""
    return J.RIGHT if is_left_hand == left_handed else J.LEFT


def usable(tracked: bool, high_confidence: bool, inferred: bool, system_gesture: bool) -> bool:
    """A hand counts as present only when tracked with high confidence, not pose-inferred, and not
    performing the system gesture."""
    return bool(tracked and high_confidence and not inferred and not system_gesture)


def frame(head_pos, head_rot, hands: dict, left_handed: bool, shoulder_width: float = DEFAULT_SHOULDER_WIDTH_M,
          last_axes: np.ndarray | None = None):
    """One canonical frame from one tick of tracking.
    hands = {"left"/"right": {"joints": [26,3], "tracked", "high_confidence", "inferred", "system_gesture"}}.
    Returns (hand_rel [2,21,3], wrist_pos [2,3], present [2], axes) — reuse `axes` as last_axes next tick."""
    R, ok = head_axes(rotate(head_rot, [0.0, 0.0, 1.0]))
    if not ok:
        R = last_axes if last_axes is not None else np.eye(3)
    hand_rel, wrist_pos, present = np.zeros((2, 21, 3)), np.zeros((2, 3)), np.zeros(2)
    for side, h in hands.items():
        is_left = side == "left"
        s = slot_for(is_left, left_handed)
        if not usable(h.get("tracked", False), h.get("high_confidence", False), h.get("inferred", False),
                      h.get("system_gesture", False)):
            continue
        hand_rel[s], wrist_pos[s] = hand(h["joints"], head_pos, R, shoulder_width, mirror=left_handed)
        present[s] = 1.0
    return hand_rel, wrist_pos, present, R


def to_world(hand_rel21: np.ndarray, wrist_pos: np.ndarray, head_pos: np.ndarray, R: np.ndarray,
             shoulder_width: float, mirror: bool) -> np.ndarray:
    """The inverse of `hand` for test fixtures: canonical joints -> 26 OpenXR world positions. The palm and
    finger metacarpals OpenXR adds are placed halfway between the wrist and the knuckles."""
    if mirror:
        hand_rel21 = hand_rel21 * [-1, 1, 1]
        wrist_pos = wrist_pos * [-1, 1, 1]
    wrist = origin(head_pos, R) + (wrist_pos * shoulder_width) @ R
    pts = wrist + hand_rel21 @ R
    out = np.zeros((26, 3))
    out[J.QUEST_INDEX] = pts
    out[0] = 0.5 * (pts[J.WRIST] + pts[J.MIDDLE[0]])  # palm
    for openxr, mcp in ((6, J.INDEX[0]), (11, J.MIDDLE[0]), (16, J.RING[0]), (21, J.PINKY[0])):
        out[openxr] = 0.5 * (pts[J.WRIST] + pts[mcp])
    return out

"""Training-time augmentation of canonical segments.

Each transform imitates a way headset data differs from the training videos:
a different viewpoint and body, tracking jitter, dropped frames, a lost hand,
signing faster or slower, and imperfect start/end detection. `view_*` keys add a
webcam's viewpoint errors (camera_view): off by default.
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


# Roughly the middle of the torso, in the body frame (shoulder widths from the nose): wrists' depth errors grow
# with their distance from it along the camera's line of sight.
TORSO = np.array([0.0, -0.65, -0.35])


def camera_view(hand_rel: np.ndarray, wrist_pos: np.ndarray, yaw_deg: float, pitch_deg: float, depth: float,
                wrist_depth: float = 1.0, tilts: tuple[float, float] = (0.0, 0.0), tilt_axis: np.ndarray | None = None,
                depth_noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """What one camera gets wrong, seen from a viewpoint `yaw_deg`/`pitch_deg` off straight ahead. The body frame
    already turns the signer to face forward, so a clean 3D landmark set looks the same from anywhere; what changes is
    MediaPipe's error. A single camera judges distance along its line of sight poorly: hand depth there is scaled by
    `depth` (and `wrist_depth` for the wrists about the torso) and noisier, and each hand's tilt is misread by
    `tilts` degrees about an axis across the line of sight. hand_rel [T,2,21,3] metres, wrist_pos [T,2,3]."""
    look = rotation(yaw_deg, pitch_deg, 0.0) @ np.array([0.0, 0.0, -1.0])
    hand_rel, wrist_pos = hand_rel.copy(), wrist_pos.copy()
    axis = tilt_axis if tilt_axis is not None else np.cross(look, [0.0, 1.0, 0.0])
    for s in range(2):
        along = hand_rel[:, s] @ look
        hand_rel[:, s] += ((depth - 1.0) * along)[..., None] * look
        if depth_noise is not None:
            hand_rel[:, s] += depth_noise[:, s][..., None] * look
        if tilts[s]:
            hand_rel[:, s] = rotate_about(hand_rel[:, s], np.zeros(3), axis, np.radians(tilts[s]))
    along = (wrist_pos - TORSO) @ look
    wrist_pos = wrist_pos + ((wrist_depth - 1.0) * along)[..., None] * look
    return hand_rel, wrist_pos


def random_camera_view(hand_rel, wrist_pos, rng: np.random.Generator, a) -> tuple[np.ndarray, np.ndarray]:
    """A random viewpoint for augmentation (keys view_yaw_deg, view_pitch_deg, view_depth [lo, hi],
    view_depth_noise_m, view_tilt_deg, view_wrist_depth); view_yaw_at / view_pitch_at fix the angles (for testing
    one camera position)."""
    lo, hi = a["view_depth"]
    w = a.get("view_wrist_depth", 0.15)
    yaw = a["view_yaw_at"] if "view_yaw_at" in a else rng.uniform(-a["view_yaw_deg"], a["view_yaw_deg"])
    pitch = a["view_pitch_at"] if "view_pitch_at" in a else rng.uniform(-a["view_pitch_deg"], a["view_pitch_deg"])
    look = rotation(yaw, pitch, 0.0) @ np.array([0.0, 0.0, -1.0])
    axis = np.cross(look, rng.normal(size=3))  # any direction across the line of sight
    return camera_view(hand_rel, wrist_pos, yaw, pitch, rng.uniform(lo, hi), 1.0 + rng.uniform(-w, w),
                       tuple(rng.normal(0.0, a.get("view_tilt_deg", 0.0), 2)), axis / max(np.linalg.norm(axis), 1e-9),
                       rng.normal(0.0, a.get("view_depth_noise_m", 0.0), hand_rel.shape[:3]))


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

    if a.get("view_prob", 0.0) and rng.random() < a["view_prob"]:  # a webcam off to one side, above or below
        hand_rel, wrist_pos = random_camera_view(hand_rel, wrist_pos, rng, a)
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

"""MediaPipe output to canonical sign segments (spec/FEATURE_SPEC.md §4–5).

A Segment holds both hands in the signer's reference frame: x = signer's right,
y = up, z = signer's forward, origin at the nose. The headset builds the same
thing from Quest hand tracking, so everything after `from_mediapipe` is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import joints as J
from .features import normalize


@dataclass
class Segment:
    hand_rel: np.ndarray   # [L,2,21,3] metres, reference-frame axes, wrist at the origin
    wrist_pos: np.ndarray  # [L,2,3] reference frame, shoulder widths from the nose
    present: np.ndarray    # [L,2] 0/1
    fps: float

    def __len__(self) -> int:
        return len(self.present)

    def slice(self, a: int, b: int) -> Segment:
        return Segment(self.hand_rel[a:b], self.wrist_pos[a:b], self.present[a:b], self.fps)


def nearest_valid(valid: np.ndarray) -> np.ndarray:
    """For every frame, the index of the nearest frame where `valid` is true."""
    idx = np.flatnonzero(valid)
    t = np.arange(len(valid))
    pos = np.searchsorted(idx, t)
    left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
    right = idx[np.clip(pos, 0, len(idx) - 1)]
    return np.where(np.abs(t - left) <= np.abs(right - t), left, right)


def body_frame(data, min_shoulder_width_m: float = 0.1):
    """The signer's body frame per frame of a clip: (pose [T, 33, 3], R [T, 3, 3], nose [T, 3], shoulder width [T]),
    or a reason string. Frames without a usable pose borrow the nearest frame that has one. R's rows are the axes:
    x towards the signer's right shoulder, y up, z the way the face points."""
    pose_ok = data["pose_present"].astype(bool)
    pose = data["pose_world"].astype(np.float64)
    ls, rs = pose[:, J.POSE_LEFT_SHOULDER], pose[:, J.POSE_RIGHT_SHOULDER]
    width = np.linalg.norm(rs - ls, axis=-1)
    valid = pose_ok & (width > min_shoulder_width_m)
    if not valid.any():
        return "no body pose"
    near = nearest_valid(valid)
    pose, width = pose[near], width[near]
    ls, rs, nose = pose[:, J.POSE_LEFT_SHOULDER], pose[:, J.POSE_RIGHT_SHOULDER], pose[:, J.POSE_NOSE]
    ears = 0.5 * (pose[:, J.POSE_LEFT_EAR] + pose[:, J.POSE_RIGHT_EAR])

    # Up runs from the shoulders to the ears, which sit roughly above them. The nose would tilt it
    # forward by ~20°, and the headset's frame is gravity-aligned.
    x = normalize(rs - ls)
    up = ears - 0.5 * (ls + rs)
    y = normalize(up - (up * x).sum(-1, keepdims=True) * x)
    z = np.cross(y, x)
    # Forward is the way the face points: the nose is in front of the ears. This needs no
    # assumption about MediaPipe's axis conventions.
    if np.median(((nose - ears) * z).sum(-1)) < 0:
        z = -z
    R = np.stack([x, y, z], axis=1)  # rows are the reference axes in MediaPipe world coordinates
    return pose, R, nose, width


def from_mediapipe(data, min_shoulder_width_m: float = 0.1) -> tuple[Segment | None, str]:
    """Build a Segment from one clip's extraction output (see extract.py for the arrays)."""
    present = data["hand_present"].astype(np.float64)
    if len(present) < 4:
        return None, "too few frames"
    frame = body_frame(data, min_shoulder_width_m)
    if isinstance(frame, str):
        return None, frame
    pose, R, nose, width = frame

    hand = data["hand_world"].astype(np.float64)
    hand_rel = np.einsum("tij,tskj->tski", R, hand - hand[:, :, :1, :])
    hand_rel *= present[:, :, None, None]
    wrists = np.stack([pose[:, J.POSE_RIGHT_WRIST], pose[:, J.POSE_LEFT_WRIST]], axis=1)
    wrist_pos = np.einsum("tij,tsj->tsi", R, wrists - nose[:, None, :]) / width[:, None, None]
    return Segment(hand_rel, wrist_pos, present, float(data["fps"])), "ok"


def body_points(data, joints: list[int], min_shoulder_width_m: float = 0.1) -> np.ndarray | None:
    """MediaPipe pose joints in the same frame and units as Segment.wrist_pos: [T, len(joints), 3], shoulder widths
    from the nose. None without a usable pose."""
    frame = body_frame(data, min_shoulder_width_m)
    if isinstance(frame, str):
        return None
    pose, R, nose, width = frame
    return np.einsum("tij,tsj->tsi", R, pose[:, joints] - nose[:, None, :]) / width[:, None, None]


def mirror(seg: Segment) -> Segment:
    """Left-right mirror: flips x and swaps the hand slots, turning a left-handed signer right-handed."""
    hand_rel = seg.hand_rel[:, ::-1].copy()
    wrist_pos = seg.wrist_pos[:, ::-1].copy()
    hand_rel[..., 0] *= -1
    wrist_pos[..., 0] *= -1
    return Segment(hand_rel, wrist_pos, seg.present[:, ::-1].copy(), seg.fps)


def wrist_extent(seg: Segment, slot: int, smooth: int = 5) -> float:
    """How far the wrist strays from its median position (90th percentile, shoulder widths), after a
    short moving average. Summing frame-to-frame steps instead would mostly measure tracking jitter."""
    ok = seg.present[:, slot] > 0.5
    if ok.sum() < 3:
        return 0.0
    p = seg.wrist_pos[ok, slot]
    if len(p) >= smooth:
        kernel = np.ones(smooth) / smooth
        p = np.stack([np.convolve(p[:, i], kernel, mode="valid") for i in range(3)], axis=1)
    return float(np.percentile(np.linalg.norm(p - np.median(p, axis=0), axis=1), 90))


HOLD_RADIUS = 0.12  # shoulder widths (~4.5 cm): the wrist counts as held within this of its median


def hold_frames(seg: Segment, slot: int = J.RIGHT, smooth: int = 5) -> np.ndarray:
    """Frames where the wrist, after a centred moving average (zero-padded at the ends, like
    numpy's convolve 'same'), stays within HOLD_RADIUS of its median over the present frames."""
    ok = seg.present[:, slot] > 0.5
    if ok.sum() < 3:
        return np.zeros(len(seg), bool)
    p = seg.wrist_pos[:, slot].copy()
    if len(p) >= smooth:
        kernel = np.ones(smooth) / smooth
        p = np.stack([np.convolve(p[:, i], kernel, mode="same") for i in range(3)], axis=1)
    return ok & (np.linalg.norm(p - np.median(p[ok], axis=0), axis=1) < HOLD_RADIUS)


def steadiest_frame(seg: Segment) -> int:
    """The frame in the middle 60% where the dominant wrist moves least: the held handshape."""
    n = len(seg)
    lo, hi = int(n * 0.2), max(int(n * 0.8), int(n * 0.2) + 1)
    speed = np.r_[np.inf, np.linalg.norm(np.diff(seg.wrist_pos[:, J.RIGHT], axis=0), axis=-1)]
    speed[seg.present[:, J.RIGHT] < 0.5] = np.inf
    window = speed[lo:hi]
    return lo + int(np.argmin(window)) if np.isfinite(window).any() else n // 2


def hand_activity(seg: Segment) -> np.ndarray:
    """Per slot: how much a hand is signing = median wrist height + how far it moves (shoulder widths).
    The passive hand rests low; the dominant hand is raised into signing space and moves."""
    out = []
    for s in (0, 1):
        ok = seg.present[:, s] > 0.5
        out.append(float(np.median(seg.wrist_pos[ok, s, 1])) + wrist_extent(seg, s) if ok.sum() >= 3 else -np.inf)
    return np.array(out)


def left_dominant_votes(segments, margin: float = 0.3) -> tuple[int, int]:
    """Count clips where the left hand is clearly the active one, and clips where the right is.
    Clips with no clear difference (symmetric two-handed signs) don't vote."""
    left = right = 0
    for seg in segments:
        a = hand_activity(seg)
        if np.isfinite(a).all() and abs(a[1] - a[0]) > margin:
            left += a[1] > a[0]
            right += a[0] > a[1]
        elif np.isfinite(a[1]) and not np.isfinite(a[0]):
            left += 1
        elif np.isfinite(a[0]) and not np.isfinite(a[1]):
            right += 1
    return int(left), int(right)


def gap_fill(seg: Segment, max_gap: int) -> Segment:
    """Linearly fill dropouts of at most `max_gap` frames between two tracked frames."""
    hand_rel, wrist_pos, present = seg.hand_rel.copy(), seg.wrist_pos.copy(), seg.present.copy()
    for s in range(2):
        idx = np.flatnonzero(present[:, s] > 0.5)
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
                wrist_pos[a + 1 : b, s] = wrist_pos[a, s] * (1 - w) + wrist_pos[b, s] * w
                hand_rel[a + 1 : b, s] = hand_rel[a, s] * (1 - w[..., None]) + hand_rel[b, s] * w[..., None]
                present[a + 1 : b, s] = 1.0
    return Segment(hand_rel, wrist_pos, present, seg.fps)


def trim(seg: Segment, rest_height: float, margin: int) -> Segment | None:
    """Keep the frames from the first to the last moment a hand is up in signing space."""
    up = seg.present > 0.5
    height = np.where(up, seg.wrist_pos[..., 1], -np.inf).max(axis=1)
    active = up.any(axis=1) & (height >= rest_height)
    if not active.any():
        active = up.any(axis=1)
    if not active.any():
        return None
    idx = np.flatnonzero(active)
    return seg.slice(max(0, idx[0] - margin), min(len(seg), idx[-1] + margin + 1))


def active_window(seg: Segment, fraction: float = 0.3, margin: int = 2, min_frames: int = 6,
                  min_range: float = 0.3) -> Segment:
    """Cut the rise from rest and the drop back to rest: keep the frames where the dominant wrist is above
    `fraction` of its own height range. Unchanged if the hand barely changes height (< min_range shoulder widths)."""
    dom = seg.present[:, J.RIGHT] > 0.5
    if dom.sum() < 3:
        return seg
    y = seg.wrist_pos[:, J.RIGHT, 1]
    lo, hi = y[dom].min(), y[dom].max()
    if hi - lo < min_range:
        return seg
    idx = np.flatnonzero(dom & (y >= lo + fraction * (hi - lo)))
    a, b = max(0, idx[0] - margin), min(len(seg), idx[-1] + margin + 1)
    return seg.slice(a, b) if b - a >= min_frames else seg


def resample(seg: Segment, n: int) -> Segment:
    """Resample to exactly n frames. Positions interpolate linearly; presence takes the nearest frame."""
    length = len(seg)
    src = np.linspace(0.0, length - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, length - 1)
    f = src - i0
    hand_rel, wrist_pos = seg.hand_rel.copy(), seg.wrist_pos.copy()
    for s in range(2):
        ok = seg.present[:, s] > 0.5
        if ok.any():  # hold the nearest tracked pose through untracked frames before interpolating
            near = nearest_valid(ok)
            hand_rel[:, s] = hand_rel[near, s]
            wrist_pos[:, s] = wrist_pos[near, s]

    def lerp(a: np.ndarray) -> np.ndarray:
        w = f.reshape(-1, *([1] * (a.ndim - 1)))
        return a[i0] * (1 - w) + a[i1] * w

    present = seg.present[np.rint(src).astype(int)]
    duration = max(length - 1, 1) / seg.fps
    return Segment(lerp(hand_rel), lerp(wrist_pos), present, fps=(n - 1) / duration)

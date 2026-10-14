"""Cuts the live stream of canonical frames into attempts (spec/RUNTIME_SPEC.md §3).

The headset feeds one frame per fixed tick (30 Hz). The segmenter is deliberately generous about
where an attempt starts and ends: `pipeline.prepare` then makes the precise cut with trim and the
active window, exactly as it does for references.

    idle      hand up (wrist above the user's rest height) for start_s -> active, keeping pre_roll_s before
    active    ends on: hand down or lost for drop_s ("drop"); dominant hand still for hold_s ("hold");
              max_attempt_s ("timeout"). Attempts with less than min_attempt_s of hand-up frames are dropped.
    cooldown  after a hold or timeout: waits for the hand to drop below rest, or to move rearm_move away
              (other than downwards), so one held sign does not fire twice and lowering the hand is not a sign.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .canonical import Segment
from .features import flexion
from .params import SegmenterParams

DOM = 0


@dataclass
class Attempt:
    segment: Segment  # raw canonical frames at the tick rate, not yet prepared
    start: int        # tick of the first frame
    end: int          # tick of the last frame (inclusive)
    reason: str       # hold | drop | timeout


@dataclass
class _Frame:
    tick: int
    hand_rel: np.ndarray   # [2,21,3]
    wrist_pos: np.ndarray  # [2,3]
    present: np.ndarray    # [2]
    up: bool
    bends: np.ndarray | None  # [15] dominant hand, None if absent


class AttemptSegmenter:
    def __init__(self, p: SegmenterParams):
        self.p = p
        self.start_n = p.frames(p.start_s)
        self.pre_n = p.frames(p.pre_roll_s)
        self.drop_n = p.frames(p.drop_s)
        self.hold_n = p.frames(p.hold_s)
        self.hold_allow = self.hold_n // 10  # frames in the hold window that may be off (tracking blips)
        self.min_n = p.frames(p.min_attempt_s)
        self.max_n = p.frames(p.max_attempt_s)
        self.reset()

    def reset(self) -> None:
        self.tick = -1
        self.state = "idle"
        self._pre: deque[_Frame] = deque(maxlen=self.pre_n + self.start_n)
        self._buf: list[_Frame] = []
        self._up_run = 0
        self._down_run = 0
        self._held_at: np.ndarray | None = None

    def push(self, hand_rel, wrist_pos, present) -> Attempt | None:
        """One canonical frame: hand_rel [2,21,3], wrist_pos [2,3], present [2]. Returns an attempt when one ends."""
        self.tick += 1
        hand_rel, wrist_pos, present = (np.asarray(a, np.float64) for a in (hand_rel, wrist_pos, present))
        ok = present > 0.5
        up = bool(any(ok[s] and wrist_pos[s, 1] >= self.p.rest_height for s in (0, 1)))
        f = _Frame(self.tick, hand_rel, wrist_pos, present, up, flexion(hand_rel[DOM]) if ok[DOM] else None)

        if self.state == "cooldown":
            self._down_run = 0 if up else self._down_run + 1
            moved = False
            if ok[DOM] and self._held_at is not None:
                d = wrist_pos[DOM] - self._held_at
                # Moving on to a new sign re-arms; lowering the hand (on its way to rest) does not.
                moved = np.linalg.norm(d) > self.p.rearm_move and d[1] > -self.p.rearm_move
            if self._down_run < self.drop_n and not moved:
                return None
            self._to_idle()

        if self.state == "idle":
            self._pre.append(f)
            self._up_run = self._up_run + 1 if up else 0
            if self._up_run >= self.start_n:
                self.state, self._buf, self._down_run = "active", list(self._pre), 0
            return None

        self._buf.append(f)
        self._down_run = 0 if up else self._down_run + 1
        if self._down_run >= self.drop_n:
            return self._finish("drop")
        if self._held():
            return self._finish("hold")
        if len(self._buf) >= self.max_n:
            return self._finish("timeout")
        return None

    def _to_idle(self) -> None:
        self.state, self._up_run, self._down_run, self._held_at = "idle", 0, 0, None
        self._pre.clear()

    def _held(self) -> bool:
        """The last hold_n frames: dominant wrist within hold_radius of its median and bends within
        hold_shape (mean over the 15 joints) of their median, allowing hold_n // 10 frames off."""
        if len(self._buf) < self.hold_n:
            return False
        window = [f for f in self._buf[-self.hold_n:] if f.bends is not None]
        if len(window) < self.hold_n - self.hold_allow:
            return False
        wrists = np.stack([f.wrist_pos[DOM] for f in window])
        bends = np.stack([f.bends for f in window])
        still = (np.linalg.norm(wrists - np.median(wrists, axis=0), axis=1) < self.p.hold_radius) & \
                (np.abs(bends - np.median(bends, axis=0)).mean(axis=1) < self.p.hold_shape)
        return int(still.sum()) >= self.hold_n - self.hold_allow

    def _finish(self, reason: str) -> Attempt | None:
        buf = self._buf
        self._buf = []
        if reason == "drop":
            self._to_idle()
        else:
            self.state, self._down_run = "cooldown", 0
            dom = [f.wrist_pos[DOM] for f in buf[-self.hold_n:] if f.present[DOM] > 0.5]
            self._held_at = np.median(np.stack(dom), axis=0) if dom else None
        if sum(f.up for f in buf) < self.min_n:
            return None
        seg = Segment(np.stack([f.hand_rel for f in buf]), np.stack([f.wrist_pos for f in buf]),
                      np.stack([f.present for f in buf]), self.p.tick_hz)
        return Attempt(seg, buf[0].tick, buf[-1].tick, reason)


def segment_stream(p: SegmenterParams, seg: Segment) -> list[Attempt]:
    """Run a whole recorded stream (at p.tick_hz) through a fresh segmenter."""
    s = AttemptSegmenter(p)
    out = []
    for t in range(len(seg)):
        a = s.push(seg.hand_rel[t], seg.wrist_pos[t], seg.present[t])
        if a is not None:
            out.append(a)
    return out

"""The one path from raw canonical frames to a matchable sequence (spec/RUNTIME_SPEC.md §2).

References (dictionary video, Studio takes), simulated attempts and live headset attempts all go
through `prepare`, so they are cut and resampled the same way. Only the rest height differs: about
-1.8 shoulder widths for standing signers on video, and the user's own (onboarding) level on Quest.
"""

from __future__ import annotations

import dataclasses

from .canonical import Segment, active_window, gap_fill, resample, trim
from .params import EngineParams
from .segmenter import segment_stream


def prepare(seg: Segment, p: EngineParams, rest_height: float) -> Segment | None:
    """gap fill → trim to signing space → active window → resample to p.seq_len. None if no hand is ever seen."""
    s = trim(gap_fill(seg, p.gap_fill_frames), rest_height, p.trim_margin_frames)
    if s is None:
        return None
    s = active_window(s, p.active_fraction, p.active_margin, p.active_min_frames, p.active_min_range)
    return resample(s, p.seq_len)


def cut_reference(seg: Segment, p: EngineParams, rest_height: float) -> Segment | None:
    """A reference recording cut exactly like a live attempt: retimed to the headset tick, streamed through
    the attempt segmenter (so a held sign ends at its hold, as attempts do), then prepared. Falls back to
    the whole recording if the segmenter never fires."""
    sp = dataclasses.replace(p.segmenter, rest_height=rest_height)
    n = max(2, int(round((len(seg) - 1) / seg.fps * sp.tick_hz)) + 1)
    attempts = segment_stream(sp, resample(gap_fill(seg, p.gap_fill_frames), n))
    return prepare(attempts[0].segment if attempts else seg, p, rest_height)

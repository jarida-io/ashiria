/** The one path from raw canonical frames to a matchable sequence (port of pipeline.py). */
import { activeWindow, gapFill, resample, trim } from './canonical.ts';
import type { EngineParams } from './params.ts';
import type { Segment } from './segment.ts';

/** Gap fill → trim to signing space → active window → resample to seq_len. Null if no hand is ever present. */
export function prepare(seg: Segment, p: EngineParams, restHeight: number): Segment | null {
  const s = trim(gapFill(seg, p.gap_fill_frames), restHeight, p.trim_margin_frames);
  if (s === null) return null;
  return resample(activeWindow(s, p.active_fraction, p.active_margin, p.active_min_frames, p.active_min_range), p.seq_len);
}

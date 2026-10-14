/** Per-user settings measured at onboarding (port of onboarding.py). */
import { median } from './math.ts';

export const REST_MARGIN = 0.25, REST_MIN = -1.9, REST_MAX = -1.2;

/** The user's rest height (shoulder widths) from resting wrist heights: median + 0.25, clamped to [-1.9, -1.2]. */
export function restHeight(restingWristY: ArrayLike<number>, fallback = -1.5): number {
  if (restingWristY.length < 10) return fallback;
  return Math.min(REST_MAX, Math.max(REST_MIN, median(restingWristY) + REST_MARGIN));
}

/** Small 3D vector helpers on [x, y, z] tuples (doubles). */

export type V3 = [number, number, number];

export const EPS = 1e-6;

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3): number => Math.sqrt(dot(a, a));
export const normalize = (a: V3): V3 => scale(a, 1 / Math.max(norm(a), EPS));
export const lerp = (a: V3, b: V3, w: number): V3 => [a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w, a[2] * (1 - w) + b[2] * w];

/** Angle in radians via atan2(|a×b|, a·b): accurate near 0 and π (spec §6). */
export const angle = (a: V3, b: V3): number => Math.atan2(norm(cross(a, b)), dot(a, b));

/** numpy's median: the middle value, or the mean of the two middle values. */
export function median(values: ArrayLike<number>): number {
  const s = Array.from(values).sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/** Round half to even, as Python's round() and numpy's rint. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

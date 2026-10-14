/** Reference matching: descriptor, DTW, per-finger errors. Port of matching.py. */
import { FEATURE_DIM, HAND_DIM } from './features.ts';
import type { Weights } from './params.ts';

/** Feature slices inside one 97-float hand block, in descriptor order. */
export const SLICES = {
  flexion: [60, 15], spread: [75, 4], tips: [79, 5], palm_normal: [84, 3], direction: [87, 3], wrist: [90, 3], present: [96, 1],
} as const;
export type Part = keyof typeof SLICES;
export const PARTS = Object.keys(SLICES) as Part[];
export const PER_HAND = 34;
export const DESCRIPTOR_DIM = 2 * PER_HAND;
export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;

/** A [rows × cols] float matrix stored row-major. */
export interface Matrix { rows: number; cols: number; data: Float32Array | Float64Array }

/** [frames × 197] features → [frames × 68] weighted descriptor for both hands (float32). */
export function descriptor(features: Float32Array, weights: Weights): Matrix {
  const n = features.length / FEATURE_DIM;
  const d = new Float32Array(n * DESCRIPTOR_DIM);
  for (let t = 0; t < n; t++) {
    let k = t * DESCRIPTOR_DIM;
    for (let slot = 0; slot < 2; slot++)
      for (const part of PARTS) {
        const [start, length] = SLICES[part];
        const w = weights[part];
        for (let i = 0; i < length; i++) d[k++] = features[t * FEATURE_DIM + slot * HAND_DIM + start + i] * w;
      }
  }
  return { rows: n, cols: DESCRIPTOR_DIM, data: d };
}

function accumulate(a: Matrix, b: Matrix, band: number): Float64Array {
  const n = a.rows, m = b.rows, dim = a.cols, w = m + 1;
  band = Math.max(band, Math.abs(n - m));
  const acc = new Float64Array((n + 1) * w).fill(Infinity);
  acc[0] = 0;
  for (let i = 1; i <= n; i++)
    for (let j = Math.max(1, i - band); j <= Math.min(m, i + band); j++) {
      let sum = 0;
      for (let c = 0; c < dim; c++) {
        const diff = a.data[(i - 1) * dim + c] - b.data[(j - 1) * dim + c];
        sum += diff * diff;
      }
      acc[i * w + j] = Math.sqrt(sum) + Math.min(acc[(i - 1) * w + j], acc[i * w + j - 1], acc[(i - 1) * w + j - 1]);
    }
  return acc;
}

/** Dynamic time warping distance with a Sakoe-Chiba band, normalised by n + m. */
export function dtw(a: Matrix, b: Matrix, band = 8): number {
  return accumulate(a, b, band)[a.rows * (b.rows + 1) + b.rows] / (a.rows + b.rows);
}

/** DTW distance and the alignment from (0, 0) to (n-1, m-1). Backtracking prefers the diagonal, then (i-1, j),
 * then (i, j-1) on ties, exactly as matching.dtw_path. */
export function dtwPath(a: Matrix, b: Matrix, band = 8): { distance: number; path: [number, number][] } {
  const n = a.rows, m = b.rows, w = m + 1;
  const acc = accumulate(a, b, band);
  let i = n, j = m;
  const path: [number, number][] = [[i - 1, j - 1]];
  while (i !== 1 || j !== 1) {
    const diag = acc[(i - 1) * w + j - 1], up = acc[(i - 1) * w + j], left = acc[i * w + j - 1];
    if (diag <= up && diag <= left) { i--; j--; } else if (up <= left) i--; else j--;
    path.push([i - 1, j - 1]);
  }
  path.reverse();
  return { distance: acc[n * w + m] / (n + m), path };
}

/** Largest bend difference per finger (radians, signed: + means bend more), thumb first. */
export function fingerErrors(attempt15: ArrayLike<number>, reference15: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let f = 0; f < 5; f++) {
    let worst = 0;
    for (let j = 0; j < 3; j++) {
      const diff = reference15[f * 3 + j] - attempt15[f * 3 + j];
      if (Math.abs(diff) > Math.abs(worst)) worst = diff;
    }
    out.push(worst);
  }
  return out;
}

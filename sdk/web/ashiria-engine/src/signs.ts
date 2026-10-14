/** What the engine knows about a sign: reference takes and tolerances (port of signs.py). */
import { FEATURE_DIM, HAND_DIM, sequenceFeatures } from './features.ts';
import { DOMINANT } from './joints.ts';
import { descriptor, dtw, SLICES, type Matrix } from './matching.ts';
import { EPS, median } from './math.ts';
import type { EngineParams } from './params.ts';
import { wristAt, type Segment } from './segment.ts';

export interface KeyPose {
  flexion: number[]; spread: number[]; palmNormal: number[]; direction: number[]; wrist: number[];
  otherPresent: boolean; otherRel: number[];
}

const OTHER = HAND_DIM, INTER = 2 * HAND_DIM;

/** Per-component median over `rows` of [frames × 197] features, using rows where the dominant hand is present
 * (all rows if none are); directions re-normalised. */
export function keyPose(f: Float32Array, rows: number[]): KeyPose {
  const P = SLICES.present[0];
  const dom = rows.filter((r) => f[r * FEATURE_DIM + P] > 0.5);
  const use = dom.length ? dom : rows;
  const other = rows.filter((r) => f[r * FEATURE_DIM + OTHER + P] > 0.5).length;
  const both = rows.filter((r) => f[r * FEATURE_DIM + P] > 0.5 && f[r * FEATURE_DIM + OTHER + P] > 0.5);
  const med = (rs: number[], start: number, n: number) =>
    Array.from({ length: n }, (_, c) => median(rs.map((r) => f[r * FEATURE_DIM + start + c])));
  const unit = (v: number[]) => { const n = Math.max(Math.hypot(v[0], v[1], v[2]), EPS); return v.map((x) => x / n); };
  return {
    flexion: med(use, SLICES.flexion[0], 15), spread: med(use, SLICES.spread[0], 4),
    palmNormal: unit(med(use, SLICES.palm_normal[0], 3)), direction: unit(med(use, SLICES.direction[0], 3)),
    wrist: med(use, SLICES.wrist[0], 3), otherPresent: other * 2 > rows.length,
    otherRel: both.length ? med(both, INTER, 3) : [0, 0, 0],
  };
}

/** Distance the dominant wrist travels after a 5-frame moving average ('valid'). Only words movement hints. */
export function pathLength(seg: Segment, smooth = 5): number {
  const n = seg.length;
  const pts: number[][] = [];
  if (n >= smooth) {
    for (let t = 0; t + smooth <= n; t++)
      pts.push([0, 1, 2].map((c) => { let s = 0; for (let m = t; m < t + smooth; m++) s += seg.wristPos[wristAt(m, DOMINANT) + c] / smooth; return s; }));
  } else for (let t = 0; t < n; t++) pts.push([0, 1, 2].map((c) => seg.wristPos[wristAt(t, DOMINANT) + c]));
  let length = 0;
  for (let k = 1; k < pts.length; k++) length += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1], pts[k][2] - pts[k - 1][2]);
  return length;
}

export interface Take {
  segment: Segment; keyFrame: number; source: string; takeId: string;
  features: Float32Array; descriptor: Matrix; key: KeyPose; pathLength: number;
}

export function bindTake(segment: Segment, keyFrame: number, p: EngineParams, source = 'studio', takeId = ''): Take {
  const features = sequenceFeatures(segment);
  return { segment, keyFrame, source, takeId, features, descriptor: descriptor(features, p.weights), key: keyPose(features, [keyFrame]), pathLength: pathLength(segment) };
}

export interface Tolerances {
  finger: number[]; finger_mean: boolean; spread: number; orientation_deg: number; location: number;
  movement: number; other_hand: number; distance_z: number; lookalike_guard: boolean;
}

export interface SignRef {
  id: string; tier: 'scored' | 'lenient' | 'watch'; takes: Take[]; tol: Tolerances; tau: number;
  lookalikes: string[]; shadowTake: number;
}

/** Distance to the nearest take and its index (the first take wins ties). */
export function closest(sign: SignRef, desc: Matrix, band: number): { distance: number; take: number } {
  let distance = Infinity, take = -1;
  sign.takes.forEach((t, k) => { const d = dtw(desc, t.descriptor, band); if (d < distance) { distance = d; take = k; } });
  return { distance, take };
}

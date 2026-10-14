/** Verify: "was that sign X?", with hint codes (port of verify.py; spec/RUNTIME_SPEC.md §4).
 * The attempt must be prepared (pipeline.prepare). */
import { sequenceFeatures } from './features.ts';
import { DOMINANT } from './joints.ts';
import { descriptor, dtw, dtwPath, FINGER_NAMES, fingerErrors, type Matrix } from './matching.ts';
import type { EngineParams } from './params.ts';
import { wristAt, type Segment } from './segment.ts';
import { closest, keyPose, pathLength, type SignRef } from './signs.ts';

export const KEY_SPAN = 2;
const EPS = 1e-6;
const AXES = [['dominant_side', 'other_side'], ['up', 'down'], ['forward', 'back']] as const;
const LOCATION_WEIGHTS = [1, 1, 0.5];

export type Direction = 'dominant_side' | 'other_side' | 'up' | 'down' | 'forward' | 'back';

export interface Hint { code: string; severity: number; finger?: string; direction?: Direction; signId?: string }

export interface VerifyResult {
  status: 'pass' | 'fail' | 'unscored' | 'no_hands';
  score: number;
  distance: number;
  take: number;
  components: Record<string, number>;
  fingerErrors: number[];
  hints: Hint[];
}

export function directionName(v: number[], weights?: number[]): Direction {
  let axis = 0, best = -1;
  for (let i = 0; i < 3; i++) { const w = Math.abs(v[i]) * (weights ? weights[i] : 1); if (w > best) { best = w; axis = i; } }
  return AXES[axis][v[axis] >= 0 ? 0 : 1];
}

const angleDeg = (a: number[], b: number[]) => {
  const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  return (Math.atan2(Math.hypot(c[0], c[1], c[2]), a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * 180) / Math.PI;
};

function wristTrack(seg: Segment, centre: number[]): Matrix {
  const data = new Float64Array(seg.length * 3);
  for (let t = 0; t < seg.length; t++) for (let c = 0; c < 3; c++) data[t * 3 + c] = seg.wristPos[wristAt(t, DOMINANT) + c] - centre[c];
  return { rows: seg.length, cols: 3, data };
}

const empty = (status: VerifyResult['status'], hints: Hint[] = []): VerifyResult =>
  ({ status, score: 0, distance: Infinity, take: -1, components: {}, fingerErrors: [], hints });

export function verify(attempt: Segment, sign: SignRef, p: EngineParams, toleranceScale = 1, lookalikes: SignRef[] = []): VerifyResult {
  if (sign.tier === 'watch') return empty('unscored');
  let present = 0;
  for (let t = 0; t < attempt.length; t++) if (attempt.present[t * 2 + DOMINANT] > 0.5) present++;
  if (present / attempt.length < p.verify.min_present) return empty('no_hands', [{ code: 'tracking.lost', severity: 1 }]);

  const feats = sequenceFeatures(attempt);
  const desc = descriptor(feats, p.weights);
  let d = Infinity, k = -1, path: [number, number][] = [];
  sign.takes.forEach((take, i) => {
    const r = dtwPath(desc, take.descriptor, p.dtw_band);
    if (k < 0 || r.distance < d) { d = r.distance; k = i; path = r.path; }
  });
  const take = sign.takes[k];
  const aligned = path.filter(([, j]) => j === take.keyFrame).map(([i]) => i);
  const rows: number[] = [];
  for (let r = Math.max(0, aligned[0] - KEY_SPAN); r < Math.min(attempt.length, aligned[aligned.length - 1] + KEY_SPAN + 1); r++) rows.push(r);
  const pose = keyPose(feats, rows), ref = take.key, tol = sign.tol, s = toleranceScale;
  const comps: Record<string, number> = {};
  const hints: Hint[] = [];

  const err = fingerErrors(pose.flexion, ref.flexion);
  const ratio = err.map((e, f) => Math.abs(e) / (tol.finger[f] * s));
  comps.shape = tol.finger_mean ? ratio.reduce((a, b) => a + b, 0) / 5 : Math.max(...ratio);
  if (comps.shape > 1) {
    let worst = [0, 1, 2, 3, 4].filter((f) => ratio[f] > 1).sort((a, b) => ratio[b] - ratio[a]);  // stable sort
    if (!worst.length) worst = [ratio.indexOf(Math.max(...ratio))];
    for (const f of worst) hints.push({ code: err[f] > 0 ? 'finger.bend_more' : 'finger.straighten', severity: ratio[f], finger: FINGER_NAMES[f] });
  }

  if (tol.spread > 0) {
    const ds = ref.spread.map((v, i) => v - pose.spread[i]);
    let worst = 0;
    for (let i = 1; i < 4; i++) if (Math.abs(ds[i]) > Math.abs(ds[worst])) worst = i;
    comps.spread = Math.abs(ds[worst]) / (tol.spread * s);
    // Bending a finger at the knuckle also changes the spread angles: only hint at spread if the shape is right.
    if (comps.spread > 1 && comps.shape <= 1) hints.push({ code: ds[worst] > 0 ? 'spread.more' : 'spread.less', severity: comps.spread });
  }

  const palm = angleDeg(pose.palmNormal, ref.palmNormal), point = angleDeg(pose.direction, ref.direction);
  comps.orientation = Math.max(palm, point) / (tol.orientation_deg * s);
  if (comps.orientation > 1)
    hints.push(palm >= point
      ? { code: 'palm.face', severity: comps.orientation, direction: directionName(ref.palmNormal) }
      : { code: 'hand.point', severity: comps.orientation, direction: directionName(ref.direction) });

  const dw = ref.wrist.map((v, i) => v - pose.wrist[i]);
  comps.location = Math.hypot(...dw.map((v, i) => v * LOCATION_WEIGHTS[i])) / (tol.location * s);
  if (comps.location > 1) hints.push({ code: 'move', severity: comps.location, direction: directionName(dw, LOCATION_WEIGHTS) });

  if (tol.movement > 0) {
    comps.movement = (2 * dtw(wristTrack(attempt, pose.wrist), wristTrack(take.segment, ref.wrist), p.dtw_band)) / (tol.movement * s);
    if (comps.movement > 1) {
      const size = pathLength(attempt) / Math.max(take.pathLength, EPS);
      hints.push({ code: size < 0.6 ? 'movement.bigger' : size > 1.6 ? 'movement.smaller' : 'movement.path', severity: comps.movement });
    }
  }

  if (tol.other_hand > 0 && ref.otherPresent) {
    if (!pose.otherPresent) {
      comps.other_hand = 2;
      hints.push({ code: 'other_hand.missing', severity: 2 });
    } else {
      const diff = ref.otherRel.map((v, i) => v - pose.otherRel[i]);
      comps.other_hand = Math.hypot(diff[0], diff[1], diff[2]) / (tol.other_hand * s);
      if (comps.other_hand > 1) hints.push({ code: 'other_hand.place', severity: comps.other_hand, direction: directionName(diff) });
    }
  }

  comps.overall = d / Math.max(sign.tau * tol.distance_z * s, EPS);
  if (comps.overall > 1) hints.push({ code: 'overall.different', severity: comps.overall });

  if (tol.lookalike_guard && lookalikes.length) {
    let e = -Infinity, other = '';
    for (const l of lookalikes) {  // the first look-alike with the largest error
      const el = (p.verify.lookalike_ratio * d) / Math.max(closest(l, desc, p.dtw_band).distance, EPS) / s;
      if (el > e) { e = el; other = l.id; }
    }
    comps.lookalike = e;
    if (e > 1) hints.push({ code: 'lookalike', severity: e, signId: other });
  }

  const worst = Math.max(...Object.values(comps));
  return {
    status: worst <= 1 ? 'pass' : 'fail', score: Math.min(1, Math.max(0, 1 - worst / 2)), distance: d, take: k,
    components: comps, fingerErrors: err, hints: hints.sort((a, b) => b.severity - a.severity),  // stable, like Python
  };
}

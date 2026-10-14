import { COUNT } from './joints.ts';
import type { V3 } from './math.ts';

/**
 * A canonical sign segment (spec §4): per frame and hand slot, the 21 joints relative to the wrist in
 * reference-frame axes (metres), the wrist position (shoulder widths from the nose) and presence (0/1).
 * Flat row-major arrays: handRel [frames, 2, 21, 3], wristPos [frames, 2, 3], present [frames, 2].
 */
export interface Segment {
  readonly length: number;
  readonly fps: number;
  readonly handRel: Float64Array;
  readonly wristPos: Float64Array;
  readonly present: Float64Array;
}

export function emptySegment(length: number, fps: number): Segment {
  return { length, fps, handRel: new Float64Array(length * 2 * COUNT * 3), wristPos: new Float64Array(length * 6), present: new Float64Array(length * 2) };
}

export const jointAt = (t: number, slot: number, j: number): number => ((t * 2 + slot) * COUNT + j) * 3;
export const wristAt = (t: number, slot: number): number => (t * 2 + slot) * 3;

export function joint(seg: Segment, t: number, slot: number, j: number): V3 {
  const i = jointAt(t, slot, j);
  return [seg.handRel[i], seg.handRel[i + 1], seg.handRel[i + 2]];
}

export function wrist(seg: Segment, t: number, slot: number): V3 {
  const i = wristAt(t, slot);
  return [seg.wristPos[i], seg.wristPos[i + 1], seg.wristPos[i + 2]];
}

export const isPresent = (seg: Segment, t: number, slot: number): boolean => seg.present[t * 2 + slot] > 0.5;

export function hand(seg: Segment, t: number, slot: number): V3[] {
  const out: V3[] = [];
  for (let j = 0; j < COUNT; j++) out.push(joint(seg, t, slot, j));
  return out;
}

export function slice(seg: Segment, a: number, b: number): Segment {
  return {
    length: b - a,
    fps: seg.fps,
    handRel: seg.handRel.slice(a * 2 * COUNT * 3, b * 2 * COUNT * 3),
    wristPos: seg.wristPos.slice(a * 6, b * 6),
    present: seg.present.slice(a * 2, b * 2),
  };
}

export function withFps(seg: Segment, fps: number): Segment {
  return { length: seg.length, fps, handRel: seg.handRel.slice(), wristPos: seg.wristPos.slice(), present: seg.present.slice() };
}

/** From nested arrays ([frames][2][21][3], [frames][2][3], [frames][2]) as in the golden files. */
export function segmentFromNested(d: { fps: number; hand_rel: number[][][][]; wrist_pos: number[][][]; present: number[][] }): Segment {
  const n = d.present.length;
  const s = emptySegment(n, d.fps);
  for (let t = 0; t < n; t++)
    for (let slot = 0; slot < 2; slot++) {
      for (let j = 0; j < COUNT; j++) s.handRel.set(d.hand_rel[t][slot][j], jointAt(t, slot, j));
      s.wristPos.set(d.wrist_pos[t][slot], wristAt(t, slot));
      s.present[t * 2 + slot] = d.present[t][slot];
    }
  return s;
}

/** From flat arrays as in content packs (references.json takes). */
export function segmentFromFlat(d: { fps: number; frames: number; hand_rel: number[]; wrist_pos: number[]; present: number[] }): Segment {
  const n = d.frames;
  if (d.hand_rel.length !== n * 2 * COUNT * 3 || d.wrist_pos.length !== n * 6 || d.present.length !== n * 2)
    throw new Error('reference take arrays have the wrong size');
  return { length: n, fps: d.fps, handRel: Float64Array.from(d.hand_rel), wristPos: Float64Array.from(d.wrist_pos), present: Float64Array.from(d.present) };
}

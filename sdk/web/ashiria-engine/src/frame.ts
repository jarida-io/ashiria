/**
 * Live tracking → one canonical frame (spec/FEATURE_SPEC.md §4.2; port of quest.py / QuestCanonical.cs).
 * Positions are in Unity-style left-handed world space (x right, y up, z forward), metres. WebXR is right-handed
 * (z backwards): convert with `fromWebXRPosition` / `fromWebXRQuaternion` first.
 */
import { COUNT, DOMINANT, OTHER, layoutCount, layoutIndex, WRIST, type JointLayout } from './joints.ts';
import { add, cross, dot, scale, sub, type V3 } from './math.ts';
import type { Frame } from './segmenter.ts';

export const DEFAULT_SHOULDER_WIDTH = 0.38;
/** Nose tip relative to the centre eye in head-yaw axes, metres (average adult): matches the video origin. */
export const NOSE_OFFSET: V3 = [0, -0.045, 0.025];
const UP: V3 = [0, 1, 0];

export type Axes = [V3, V3, V3];

export interface HandInput {
  joints: ArrayLike<number>;  // layoutCount × 3, world space (left-handed)
  layout: JointLayout;
  tracked: boolean; highConfidence: boolean; inferred: boolean; systemGesture: boolean;
}

export const fromWebXRPosition = (x: number, y: number, z: number): V3 => [x, y, -z];
/** A WebXR (right-handed) orientation as the equivalent left-handed quaternion (x, y, z, w). */
export const fromWebXRQuaternion = (x: number, y: number, z: number, w: number): [number, number, number, number] => [-x, -y, z, w];

/** The head's forward vector: the rotation (x, y, z, w) applied to (0, 0, 1). */
export const forwardFromQuaternion = (qx: number, qy: number, qz: number, qw: number): V3 =>
  [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)];

/** Head-yaw axes (rows x, y, z); null when looking almost straight up or down. */
export function headAxes(forward: V3): Axes | null {
  const flat: V3 = [forward[0], 0, forward[2]];
  const n = Math.hypot(flat[0], flat[2]);
  if (n <= 1e-3) return null;
  const z: V3 = [flat[0] / n, 0, flat[2] / n];
  return [cross(UP, z), UP, z];
}

export const usable = (h: HandInput | null): h is HandInput =>
  !!h && h.tracked && h.highConfidence && !h.inferred && !h.systemGesture;

/** Right-handed users: right hand in slot 0. Left-handed users: the (mirrored) left hand in slot 0. */
export const slotFor = (isLeftHand: boolean, leftHanded: boolean): number => (isLeftHand === leftHanded ? DOMINANT : OTHER);

/** One hand into slot `slot` of `out`. */
export function handInto(h: HandInput, head: V3, axes: Axes, shoulderWidth: number, mirror: boolean, out: Frame, slot: number): void {
  const idx = layoutIndex(h.layout);
  if (h.joints.length < layoutCount(h.layout) * 3) throw new Error('not enough joints for the layout');
  const at = (k: number): V3 => [h.joints[k * 3], h.joints[k * 3 + 1], h.joints[k * 3 + 2]];
  const [x, y, z] = axes;
  const wrist = at(idx[WRIST]);
  const sign = mirror ? -1 : 1;
  for (let j = 0; j < COUNT; j++) {
    const p = sub(at(idx[j]), wrist);
    const o = (slot * COUNT + j) * 3;
    out.handRel[o] = sign * dot(x, p); out.handRel[o + 1] = dot(y, p); out.handRel[o + 2] = dot(z, p);
  }
  const origin = add(add(add(head, scale(x, NOSE_OFFSET[0])), scale(y, NOSE_OFFSET[1])), scale(z, NOSE_OFFSET[2]));
  const w = sub(wrist, origin);
  out.wristPos[slot * 3] = (sign * dot(x, w)) / shoulderWidth;
  out.wristPos[slot * 3 + 1] = dot(y, w) / shoulderWidth;
  out.wristPos[slot * 3 + 2] = dot(z, w) / shoulderWidth;
  out.present[slot] = 1;
}

export const emptyFrame = (): Frame => ({ handRel: new Float64Array(2 * COUNT * 3), wristPos: new Float64Array(6), present: new Float64Array(2) });

/** Builds canonical frames tick by tick, keeping the reference axes while the user looks straight up or down. */
export class FrameBuilder {
  axes: Axes | null = null;

  build(head: V3, headForward: V3, right: HandInput | null, left: HandInput | null, leftHanded: boolean,
        shoulderWidth: number, out: Frame = emptyFrame()): Frame {
    const axes = headAxes(headForward) ?? this.axes ?? [[1, 0, 0], UP, [0, 0, 1]];
    this.axes = axes;
    out.handRel.fill(0); out.wristPos.fill(0); out.present.fill(0);
    for (const [isLeft, h] of [[false, right], [true, left]] as const)
      if (usable(h)) handInto(h, head, axes, shoulderWidth, leftHanded, out, slotFor(isLeft, leftHanded));
    return out;
  }
}

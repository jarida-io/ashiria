/** Studio take writer (spec/STUDIO_FORMAT.md): raw world poses plus the app's canonical frames, one JSON per take. */
import { SPEC_VERSION } from './features.ts';
import { layoutCount, type JointLayout } from './joints.ts';
import { RUNTIME_VERSION } from './params.ts';
import type { Frame } from './segmenter.ts';

export const STUDIO_FORMAT = 'ashiria.studio-take';
export const STUDIO_FORMAT_VERSION = 1;

export interface StudioHeader {
  takeId: string; sessionId: string; signId: string; packId: string; packVersion: string; signer: string;
  role: 'reference' | 'test'; handedness: 'right' | 'left'; consentRef: string; shoulderWidthM: number;
  shoulderCalibrated: boolean; restHeight: number; device: string; appVersion: string; tickHz: number; recordedAt: string;
}

/** One hand for one tick: joint positions (layout × 3) and rotations (layout × 4, x y z w), left-handed world space. */
export interface HandSample { pos: ArrayLike<number>; rot: ArrayLike<number>; tracked: boolean; highConf: boolean; inferred: boolean; systemGesture: boolean }

const r = (v: number, d: number) => { const k = 10 ** d; return Math.round(v * k) / k; };

export class StudioTakeWriter {
  readonly header: StudioHeader;
  readonly layout: JointLayout;
  private t: number[] = []; private headPos: number[] = []; private headRot: number[] = [];
  private sides = { right: this.side(), left: this.side() };
  private canon = { hand_rel: [] as number[], wrist_pos: [] as number[], present: [] as number[] };

  constructor(header: StudioHeader, layout: JointLayout = 'webxr25') { this.header = header; this.layout = layout; }

  private side() { return { tracked: [] as number[], high_conf: [] as number[], inferred: [] as number[], system_gesture: [] as number[], pos: [] as number[], rot: [] as number[] }; }

  get frames(): number { return this.t.length; }

  addFrame(time: number, head: ArrayLike<number>, headRot: ArrayLike<number>, right: HandSample, left: HandSample, canonical: Frame): void {
    const n = layoutCount(this.layout);
    this.t.push(r(time, 4));
    for (let i = 0; i < 3; i++) this.headPos.push(r(head[i], 6));
    for (let i = 0; i < 4; i++) this.headRot.push(r(headRot[i], 7));
    for (const [name, h] of [['right', right], ['left', left]] as const) {
      const s = this.sides[name];
      s.tracked.push(+h.tracked); s.high_conf.push(+h.highConf); s.inferred.push(+h.inferred); s.system_gesture.push(+h.systemGesture);
      for (let i = 0; i < n * 3; i++) s.pos.push(r(h.pos[i] ?? 0, 6));
      for (let i = 0; i < n * 4; i++) s.rot.push(r(h.rot[i] ?? 0, 6));
    }
    canonical.handRel.forEach((v) => this.canon.hand_rel.push(r(v, 6)));
    canonical.wristPos.forEach((v) => this.canon.wrist_pos.push(r(v, 6)));
    canonical.present.forEach((v) => this.canon.present.push(v));
  }

  toJSON(trimStart = 0, trimEnd = -1, status: 'keep' | 'reject' = 'keep'): string {
    const h = this.header;
    return JSON.stringify({
      format: STUDIO_FORMAT, format_version: STUDIO_FORMAT_VERSION, take_id: h.takeId, session_id: h.sessionId, sign_id: h.signId,
      pack_id: h.packId, pack_version: h.packVersion, signer: h.signer, role: h.role, handedness: h.handedness, consent_ref: h.consentRef,
      shoulder_width_m: h.shoulderWidthM, shoulder_calibrated: h.shoulderCalibrated, rest_height: h.restHeight, device: h.device,
      app_version: h.appVersion, spec_version: SPEC_VERSION, runtime_version: RUNTIME_VERSION, joint_layout: this.layout, tick_hz: h.tickHz,
      recorded_at: h.recordedAt, trim: { start: trimStart, end: trimEnd < 0 ? this.frames : trimEnd }, status,
      frames: { n: this.frames, t: this.t, head_pos: this.headPos, head_rot: this.headRot, right: this.sides.right, left: this.sides.left, canonical: this.canon },
    });
  }
}

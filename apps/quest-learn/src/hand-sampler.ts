/**
 * Reads WebXR hand tracking every rendered frame and turns it into canonical frames at a fixed 30 Hz
 * (spec/RUNTIME_SPEC.md §1), feeding the attempt segmenter.
 *
 * WebXR is right-handed (z points backwards); the engine and the Studio format use left-handed world space
 * (z forward), so positions become (x, y, -z) and rotations (-x, -y, z, w). Hand data stays in this page:
 * it is processed and dropped, except in Studio (development builds), which saves the team's own recordings.
 */
import {
  AttemptSegmenter, emptyFrame, forwardFromQuaternion, FrameBuilder, TickClock, type Attempt, type Frame, type HandInput, type V3,
} from '@ashiria/engine';
import { createSystem, VisibilityState } from '@iwsdk/core';
import { app, STUDIO_ENABLED } from './app-state.js';

const JOINTS = 25;
const RING = 300; // 10 s of raw samples at 30 Hz, for Studio

/** One tick of raw tracking in left-handed world space. */
export interface RawSample {
  tick: number;
  time: number;
  head: Float64Array;     // 3
  headRot: Float64Array;  // 4 (x, y, z, w)
  hands: { right: RawHand; left: RawHand };
  frame: Frame;           // canonical
}

export interface RawHand { tracked: boolean; pos: Float64Array; rot: Float64Array }

const rawHand = (): RawHand => ({ tracked: false, pos: new Float64Array(JOINTS * 3), rot: new Float64Array(JOINTS * 4) });

/** Rotation part of a column-major 4×4 matrix as a quaternion (x, y, z, w). */
function quatFromMatrix(m: Float32Array, o: number, out: Float64Array, k: number): void {
  const m00 = m[o], m01 = m[o + 4], m02 = m[o + 8], m10 = m[o + 1], m11 = m[o + 5], m12 = m[o + 9], m20 = m[o + 2], m21 = m[o + 6], m22 = m[o + 10];
  const tr = m00 + m11 + m22;
  let x: number, y: number, z: number, w: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  // Right-handed WebXR → left-handed: (-x, -y, z, w).
  out[k] = -x; out[k + 1] = -y; out[k + 2] = z; out[k + 3] = w;
}

type AttemptListener = (attempt: Attempt) => void;

export class HandSamplerSystem extends createSystem({}) {
  private clock = new TickClock(30);
  private builder = new FrameBuilder();
  private segmenter = new AttemptSegmenter(app.params().segmenter);
  private matrices = new Float32Array(JOINTS * 16);
  private spaces = new Map<XRHand, XRJointSpace[]>();
  private latest = { head: new Float64Array(3), headRot: new Float64Array([0, 0, 0, 1]), right: rawHand(), left: rawHand(), valid: false };
  private inputs = { right: this.handInput(), left: this.handInput() };
  private ring: RawSample[] = [];
  private tick = -1;
  private listeners = new Set<AttemptListener>();

  /** The most recent canonical frame (for the debug readout and the shadow hands). */
  lastFrame: Frame = emptyFrame();
  /** Controllers are connected and no hand is tracked (the user should put them down). */
  controllersOnly = false;
  /** Seconds since a hand was last tracked (while in XR). */
  sinceHand = 0;
  /** The current attempt state: 'idle' | 'active' | 'cooldown'. */
  get attemptState() { return this.segmenter.state; }
  private blurred = false;
  get axes() { return this.builder.axes; }
  get headPosition(): Float64Array { return this.latest.head; }

  private handInput(): HandInput {
    return { joints: new Float64Array(JOINTS * 3), layout: 'webxr25', tracked: false, highConfidence: false, inferred: false, systemGesture: false };
  }

  onAttempt(fn: AttemptListener): () => void { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  /** Raw samples for ticks [from, to] still in the buffer (Studio). */
  samples(from: number, to: number): RawSample[] { return this.ring.filter((s) => s.tick >= from && s.tick <= to); }

  /** Start fresh (new segmenter parameters, a mode change, or after a pause). */
  resetSegmenter(): void { this.segmenter = new AttemptSegmenter(app.params().segmenter); this.tick = -1; this.ring = []; }

  init(): void {
    this.cleanupFuncs.push(app.subscribe(() => {
      if (this.segmenter.p.rest_height !== app.settings.restHeight) this.resetSegmenter();
    }));
  }

  update(delta: number): void {
    const frame = this.xrFrame;
    const ref = this.xrManager?.getReferenceSpace();
    if (!frame || !ref) { this.clock.reset(); return; }
    // Paused (system menu, a permission prompt): drop any half-made attempt and wait.
    if (this.world.visibilityState.peek() !== VisibilityState.Visible) {
      if (!this.blurred) { this.blurred = true; this.resetSegmenter(); }
      this.clock.reset();
      return;
    }
    this.blurred = false;
    this.sample(frame, ref);
    const handTracked = this.latest.right.tracked || this.latest.left.tracked;
    this.sinceHand = handTracked ? 0 : this.sinceHand + delta;
    let controllers = false;
    for (const source of frame.session.inputSources) if (source.gamepad && !source.hand) controllers = true;
    this.controllersOnly = controllers && !handTracked;
    const ticks = this.clock.advance(delta);
    for (let i = 0; i < ticks && this.latest.valid; i++) this.step();
  }

  private sample(frame: XRFrame, ref: XRReferenceSpace): void {
    const viewer = frame.getViewerPose(ref);
    this.latest.valid = !!viewer;
    if (!viewer) return;
    const p = viewer.transform.position, q = viewer.transform.orientation;
    this.latest.head.set([p.x, p.y, -p.z]);
    this.latest.headRot.set([-q.x, -q.y, q.z, q.w]);
    this.latest.right.tracked = this.latest.left.tracked = false;
    for (const source of frame.session.inputSources) {
      if (!source.hand || (source.handedness !== 'left' && source.handedness !== 'right')) continue;
      const out = this.latest[source.handedness];
      let spaces = this.spaces.get(source.hand);
      if (!spaces) { spaces = Array.from(source.hand.values()); this.spaces.set(source.hand, spaces); }
      out.tracked = this.readJoints(frame, spaces, ref, out);
    }
  }

  private readJoints(xrFrame: XRFrame, spaces: XRJointSpace[], ref: XRReferenceSpace, out: RawHand): boolean {
    const m = this.matrices;
    const frame = xrFrame as XRFrame & { fillPoses?: (spaces: XRSpace[], base: XRSpace, out: Float32Array) => boolean };
    if (typeof frame.fillPoses === 'function') {
      if (!frame.fillPoses(spaces, ref, m)) return false;
    } else {
      for (let j = 0; j < JOINTS; j++) {
        const pose = frame.getJointPose?.(spaces[j], ref);
        if (!pose) return false;
        m.set(pose.transform.matrix, j * 16);
      }
    }
    for (let j = 0; j < JOINTS; j++) {
      const o = j * 16;
      out.pos[j * 3] = m[o + 12]; out.pos[j * 3 + 1] = m[o + 13]; out.pos[j * 3 + 2] = -m[o + 14];
      quatFromMatrix(m, o, out.rot, j * 4);
    }
    return true;
  }

  private step(): void {
    this.tick++;
    const s = this.latest, settings = app.settings;
    for (const side of ['right', 'left'] as const) {
      const input = this.inputs[side], raw = s[side];
      input.tracked = input.highConfidence = raw.tracked;  // WebXR has no confidence flags: tracked means usable
      (input.joints as Float64Array).set(raw.pos);
    }
    const fwd = forwardFromQuaternion(s.headRot[0], s.headRot[1], s.headRot[2], s.headRot[3]);
    const head: V3 = [s.head[0], s.head[1], s.head[2]];
    const f = this.builder.build(head, fwd, this.inputs.right, this.inputs.left, settings.leftHanded, settings.shoulderWidth, emptyFrame());
    this.lastFrame = f;
    if (STUDIO_ENABLED) this.ring.push({
      tick: this.tick, time: this.tick / this.clock.hz, head: s.head.slice(), headRot: s.headRot.slice(),
      hands: { right: { tracked: s.right.tracked, pos: s.right.pos.slice(), rot: s.right.rot.slice() },
               left: { tracked: s.left.tracked, pos: s.left.pos.slice(), rot: s.left.rot.slice() } },
      frame: f,
    });
    if (this.ring.length > RING) this.ring.shift();  // Studio only: raw samples for saving takes
    const attempt = this.segmenter.push(f);
    if (attempt) this.listeners.forEach((fn) => fn(attempt));
  }
}

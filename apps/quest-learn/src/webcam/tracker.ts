// Steadier webcam tracking on top of MediaPipe: hands kept in the same slot (signer's right, left), jitter smoothed
// with One Euro filters (strong when still, light when moving, so little lag), and a hand that MediaPipe misses for
// a moment held instead of dropped. Produces the engine's frames (body-centred hands). A mirrored camera can't be
// told from the picture (a mirrored person still looks like a person), so flipping is the viewer's choice (Vision).
import { assignSlots, bodyFrame, cameraFrame, type CameraHand, type Frame, type Point } from '@ashiria/engine';
import type { CameraFrame } from './vision.js';

/** One Euro filter (Casiez et al. 2012) over a fixed-length vector of values. */
class OneEuro {
  private x: Float64Array | null = null;
  private dx: Float64Array | null = null;
  private t = 0;
  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;

  constructor(minCutoff: number, beta: number, dCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(values: ArrayLike<number>, tMs: number): Float64Array {
    if (!this.x || this.x.length !== values.length) {
      this.x = Float64Array.from(values);
      this.dx = new Float64Array(values.length);
      this.t = tMs;
      return this.x;
    }
    const dt = Math.max((tMs - this.t) / 1000, 1e-3);
    this.t = tMs;
    const aD = OneEuro.alpha(this.dCutoff, dt);
    // Speed of the whole vector drives the cutoff, so a moving hand isn't smoothed point by point.
    let speed = 0;
    for (let i = 0; i < values.length; i++) {
      const d = (values[i] - this.x[i]) / dt;
      this.dx![i] = aD * d + (1 - aD) * this.dx![i];
      speed = Math.max(speed, Math.abs(this.dx![i]));
    }
    const a = OneEuro.alpha(this.minCutoff + this.beta * speed, dt);
    for (let i = 0; i < values.length; i++) this.x[i] = a * values[i] + (1 - a) * this.x[i];
    return this.x;
  }

  reset(): void { this.x = null; this.dx = null; }
}

// MediaPipe pose joints that set the body frame and place the hands.
const TORSO = [0, 7, 8, 11, 12], WRISTS = [15, 16];

const toPoints = (a: Float64Array): Point[] => Array.from({ length: a.length / 3 }, (_, i) => ({ x: a[i * 3], y: a[i * 3 + 1], z: a[i * 3 + 2] }));
const flat = (pts: Point[]): number[] => pts.flatMap((p) => [p.x, p.y, p.z]);

export interface Tracked { ok: boolean; slots: [CameraHand | null, CameraHand | null]; held: [boolean, boolean] }

export class CameraTracker {
  /** Smoothing: on for showing hands (the emulator), off to read signs exactly as the references were made. */
  smooth = true;
  /** Keep a hand this long (ms) after MediaPipe loses it, instead of dropping it. */
  holdMs = 250;
  private handFilters = [new OneEuro(1.2, 4), new OneEuro(1.2, 4)];
  private torsoFilter = new OneEuro(0.5, 0.5);
  private wristFilter = new OneEuro(1.0, 3);
  private last: [CameraHand | null, CameraHand | null] = [null, null];
  private lastSeen = [-Infinity, -Infinity];
  private body: ReturnType<typeof bodyFrame> = null;

  /** Updates `out` from a camera frame. `ok` is false until a body pose has been seen. */
  update(f: CameraFrame, out: Frame): Tracked {
    const now = f.time;
    let pose = f.pose;
    if (pose && this.smooth) {
      const world = pose.world.map((p) => ({ ...p }));
      const torso = this.torsoFilter.filter(flat(TORSO.map((k) => world[k])), now);
      const wrists = this.wristFilter.filter(flat(WRISTS.map((k) => world[k])), now);
      TORSO.forEach((k, i) => Object.assign(world[k], { x: torso[i * 3], y: torso[i * 3 + 1], z: torso[i * 3 + 2] }));
      WRISTS.forEach((k, i) => Object.assign(world[k], { x: wrists[i * 3], y: wrists[i * 3 + 1], z: wrists[i * 3 + 2] }));
      pose = { image: pose.image, world };
    }
    const body = pose ? bodyFrame(pose.world) : null;
    if (body) this.body = body;

    const [r, l] = assignSlots(f.hands, pose ? pose.image : null);
    const found: [CameraHand | null, CameraHand | null] = [r >= 0 ? f.hands[r] : null, l >= 0 ? f.hands[l] : null];

    const slots: [CameraHand | null, CameraHand | null] = [null, null];
    const held: [boolean, boolean] = [false, false];
    for (let s = 0; s < 2; s++) {
      const h = found[s];
      if (h) {
        const world = this.smooth ? toPoints(this.handFilters[s].filter(flat(h.world), now)) : h.world;
        slots[s] = { ...h, world };
        this.last[s] = slots[s];
        this.lastSeen[s] = now;
      } else if (now - this.lastSeen[s] < this.holdMs && this.last[s]) {
        slots[s] = this.last[s];
        held[s] = true;
      } else {
        this.handFilters[s].reset();
        this.last[s] = null;
      }
    }
    if (!this.body) return { ok: false, slots, held };
    cameraFrame(this.body, [slots[0]?.world ?? null, slots[1]?.world ?? null], out);
    return { ok: true, slots, held };
  }

  reset(): void {
    this.handFilters.forEach((f) => f.reset());
    this.torsoFilter.reset();
    this.wristFilter.reset();
    this.last = [null, null];
    this.lastSeen = [-Infinity, -Infinity];
    this.body = null;
  }
}

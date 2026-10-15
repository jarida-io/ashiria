// Camera input: one frame from MediaPipe's hand and pose landmarkers, read the way the training pipeline reads the
// dictionary videos (training/src/ashiria_train/canonical.py from_mediapipe, extract.py assign_slots). Use the camera's
// raw frames, not a mirrored selfie preview: slot 0 is then the signer's right hand, as on video.
import { COUNT, WRIST } from './joints.ts';
import { add, cross, dot, normalize, scale, sub, type V3 } from './math.ts';
import type { Frame } from './segmenter.ts';

// MediaPipe pose landmark indices.
const NOSE = 0, LEFT_EAR = 7, RIGHT_EAR = 8, LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12, LEFT_WRIST = 15, RIGHT_WRIST = 16;
export const MIN_SHOULDER_WIDTH_M = 0.1;

export interface Point { x: number; y: number; z: number }

/** One detected hand: image landmarks (x, y in 0..1), world landmarks (metres) and MediaPipe's handedness label. */
export interface CameraHand { image: Point[]; world: Point[]; label: string }

/** The body's axes in camera world coordinates (rows: towards the signer's right, up, the way the face points). */
export interface BodyFrame { axes: [V3, V3, V3]; nose: V3; width: number; wrists: [V3, V3] }

const v = (p: Point): V3 => [p.x, p.y, p.z];

/** Which detection goes in each slot ([signer's right, left], -1 for none). With a body pose, each hand goes to the
 * nearer pose wrist; without, MediaPipe's label, which names the hand itself on an unmirrored camera image. */
export function assignSlots(hands: CameraHand[], poseImage: Point[] | null): [number, number] {
  const out: [number, number] = [-1, -1];
  if (!hands.length) return out;
  const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
  if (poseImage) {
    const targets = [poseImage[RIGHT_WRIST], poseImage[LEFT_WRIST]];
    const cost = hands.slice(0, 2).map((h) => targets.map((t) => dist(h.image[WRIST], t)));
    if (cost.length === 1) {
      out[cost[0][0] <= cost[0][1] ? 0 : 1] = 0;
      return out;
    }
    const straight = cost[0][0] + cost[1][1], crossed = cost[0][1] + cost[1][0];
    return straight <= crossed ? [0, 1] : [1, 0];
  }
  if (hands.length >= 2 && hands[0].label === hands[1].label) {
    // The same label twice: the signer's right hand is on the left of the image.
    return hands[0].image[WRIST].x <= hands[1].image[WRIST].x ? [0, 1] : [1, 0];
  }
  hands.slice(0, 2).forEach((h, k) => { out[h.label === 'Right' ? 0 : 1] = k; });
  return out;
}

/** The body frame from pose world landmarks, or null when the shoulders are missing or too close together. */
export function bodyFrame(world: Point[]): BodyFrame | null {
  const ls = v(world[LEFT_SHOULDER]), rs = v(world[RIGHT_SHOULDER]), nose = v(world[NOSE]);
  const width = Math.hypot(...sub(rs, ls));
  if (!(width > MIN_SHOULDER_WIDTH_M)) return null;
  const ears = scale(add(v(world[LEFT_EAR]), v(world[RIGHT_EAR])), 0.5);
  // Up runs from the shoulders to the ears; forward is the way the face points (the nose is in front of the ears).
  const x = normalize(sub(rs, ls));
  const up = sub(ears, scale(add(ls, rs), 0.5));
  const y = normalize(sub(up, scale(x, dot(up, x))));
  let z = cross(y, x);
  if (dot(sub(nose, ears), z) < 0) z = scale(z, -1);
  return { axes: [x, y, z], nose, width, wrists: [v(world[RIGHT_WRIST]), v(world[LEFT_WRIST])] };
}

const rotate = (axes: [V3, V3, V3], p: V3): V3 => [dot(axes[0], p), dot(axes[1], p), dot(axes[2], p)];

/** Fill `out` from a body frame and the hands in each slot (null for none): Segment units, like from_mediapipe. */
export function cameraFrame(body: BodyFrame, slots: [Point[] | null, Point[] | null], out: Frame): void {
  for (let s = 0; s < 2; s++) {
    const w = rotate(body.axes, sub(body.wrists[s], body.nose));
    for (let i = 0; i < 3; i++) out.wristPos[s * 3 + i] = w[i] / body.width;
    const world = slots[s];
    out.present[s] = world ? 1 : 0;
    for (let j = 0; j < COUNT; j++) {
      const p = world ? rotate(body.axes, sub(v(world[j]), v(world[WRIST]))) : [0, 0, 0];
      for (let i = 0; i < 3; i++) out.handRel[(s * COUNT + j) * 3 + i] = p[i];
    }
  }
}

/** Camera frames to engine frames. Frames without a usable body pose borrow the last one that had it (the
 * training pipeline borrows the nearest; live input can only look back). */
export class CameraFrameBuilder {
  private body: BodyFrame | null = null;

  /** Returns false (and leaves `out` alone) until a body pose has been seen. */
  build(hands: CameraHand[], pose: { image: Point[]; world: Point[] } | null, out: Frame): boolean {
    const body = pose ? bodyFrame(pose.world) : null;
    if (body) this.body = body;
    if (!this.body) return false;
    const [r, l] = assignSlots(hands, pose ? pose.image : null);
    cameraFrame(this.body, [r >= 0 ? hands[r].world : null, l >= 0 ? hands[l].world : null], out);
    return true;
  }

  reset(): void { this.body = null; }
}

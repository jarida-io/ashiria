// Development only: drive the headset emulator's hands with your own hands through the webcam, so the app can be
// tried on a desktop before a headset arrives. The camera is read the way the dictionary videos were (engine
// CameraFrameBuilder: body-centred hands), then placed in front of the emulated head where a Quest would report them,
// as WebXR's 25 hand joints. The app's own hand sampler reads them back unchanged.
import { CameraFrameBuilder, NOSE_OFFSET, assignSlots, emptyFrame, type Frame, type V3 } from '@ashiria/engine';
import { app } from '../app-state.js';
import { openVision, type CameraFrame, type Vision } from './vision.js';

const JOINTS = ['wrist', 'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  ...['index-finger', 'middle-finger', 'ring-finger', 'pinky-finger'].flatMap((f) =>
    ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip'].map((b) => `${f}-${b}`))];
// Each WebXR joint's MediaPipe landmark. MediaPipe has no finger metacarpals: those sit a quarter of the way from
// the wrist to the knuckle.
const SOURCE: (number | [number, number])[] = [0, 1, 2, 3, 4, [0, 5], 5, 6, 7, 8, [0, 9], 9, 10, 11, 12,
  [0, 13], 13, 14, 15, 16, [0, 17], 17, 18, 19, 20];
const RADIUS = [0.02, 0.019, 0.012, 0.01, 0.009, ...[0, 1, 2, 3].flatMap(() => [0.018, 0.011, 0.009, 0.008, 0.007])];
// Along each finger: the joint a bone points to (for the joint's orientation); tips point on from their parent.
const NEXT = [10, 2, 3, 4, -1, 6, 7, 8, 9, -1, 11, 12, 13, 14, -1, 16, 17, 18, 19, -1, 21, 22, 23, 24, -1];
// The emulator stores right-hand poses as mirrored left-hand ones; this is its mirror, which undoes itself.
const MIRROR = [1, -1, -1, 0, -1, 1, 1, 0, -1, 1, 1, 0, -1, 1, 1, 1];

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: V3): V3 => mul(a, 1 / Math.max(Math.hypot(...a), 1e-9));

/** The emulated head's level axes in WebXR coordinates: right, up, forward (the app's frame turns with yaw only). */
function headAxes(q: { x: number; y: number; z: number; w: number }): [V3, V3, V3] {
  // Forward is -z rotated by the head's quaternion, flattened onto the floor.
  const { x, y, z, w } = q;
  const f: V3 = [-(2 * (x * z + w * y)), 0, -(1 - 2 * (x * x + y * y))];
  const forward = unit(f);
  return [[-forward[2], 0, forward[0]], [0, 1, 0], forward];
}

/** The 21 hand points of slot `s` in WebXR world coordinates, placed where a Quest would see them. */
function handPoints(frame: Frame, s: number, head: V3, axes: [V3, V3, V3], shoulderWidth: number): V3[] {
  // Body frame (x the signer's right, y up, z forward) to WebXR world.
  const world = (v: V3): V3 => add(add(mul(axes[0], v[0]), mul(axes[1], v[1])), mul(axes[2], v[2]));
  const origin = add(head, world(NOSE_OFFSET));
  const w = frame.wristPos, h = frame.handRel;
  const wrist = add(origin, world(mul([w[s * 3], w[s * 3 + 1], w[s * 3 + 2]], shoulderWidth)));
  return Array.from({ length: 21 }, (_, j) => add(wrist, world([h[(s * 21 + j) * 3], h[(s * 21 + j) * 3 + 1], h[(s * 21 + j) * 3 + 2]])));
}

/** A tracked input's root transform: where the emulator keeps it (its own panel can move it), as a rotation and a
 * position, so joints can be measured from it. */
interface Root { p: V3; r: [V3, V3, V3] } // r: the root's x, y, z axes in world coordinates

function rootOf(hand: any): Root {
  const { x, y, z, w } = hand.quaternion;
  return {
    p: [hand.position.x, hand.position.y, hand.position.z],
    r: [[1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
      [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
      [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]],
  };
}

/** WebXR joint transforms relative to the hand's root: -z along the bone towards the tip, +y out of the back of
 * the hand. */
function jointMatrices(p21: V3[], right: boolean, root: Root): Float32Array[] {
  const local = (v: V3): V3 => [dot(root.r[0], v), dot(root.r[1], v), dot(root.r[2], v)];
  const pts = SOURCE.map((src) => (typeof src === 'number' ? p21[src] : add(p21[src[0]], mul(sub(p21[src[1]], p21[src[0]]), 0.25))));
  // Out of the palm: index knuckle × little-finger knuckle for a right hand; the left hand is its mirror image.
  const palm = mul(unit(cross(sub(p21[5], p21[0]), sub(p21[17], p21[0]))), right ? 1 : -1);
  return pts.map((p, k) => {
    const along = unit(NEXT[k] >= 0 ? sub(pts[NEXT[k]], p) : sub(p, pts[k - 1]));
    const zAxis = mul(along, -1);
    let yAxis = mul(palm, -1);
    yAxis = unit(sub(yAxis, mul(zAxis, dot(yAxis, zAxis))));
    const xAxis = cross(yAxis, zAxis);
    const m = new Float32Array([...local(xAxis), 0, ...local(yAxis), 0, ...local(zAxis), 0, ...local(sub(p, root.p)), 1]);
    if (right) for (let i = 0; i < 16; i++) m[i] *= MIRROR[i];
    return m;
  });
}

function setPose(hand: any, matrices: Float32Array[]): void {
  const key = Object.getOwnPropertySymbols(hand).find((s) => s.description === '@iwer/xr-hand-input');
  if (!key) throw new Error('This version of the headset emulator has no hand poses to set.');
  const store = hand[key];
  const pose = (store.poses.webcam ??= { jointTransforms: {} });
  JOINTS.forEach((name, k) => { pose.jointTransforms[name] = { offsetMatrix: matrices[k], radius: RADIUS[k] }; });
  if (hand.poseId !== 'webcam') hand.poseId = 'webcam';
  hand.setPinchValueImmediate?.(0);
  hand.updateHandPose();
}

/** A small mirrored camera view with the tracked hands, so you can see what the emulator is getting. */
function preview(video: HTMLVideoElement): { draw(f: CameraFrame, slots: [number, number] | null): void; remove(): void } {
  const box = document.createElement('div');
  box.style.cssText = 'position:fixed;right:12px;bottom:12px;width:220px;aspect-ratio:4/3;z-index:2147483000;border-radius:12px;'
    + 'overflow:hidden;background:#000;box-shadow:0 6px 24px rgba(0,0,0,.35);pointer-events:none';
  video.style.cssText = 'width:100%;height:100%;object-fit:cover;transform:scaleX(-1)';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
  const label = document.createElement('div');
  label.style.cssText = 'position:absolute;left:8px;top:6px;font:600 12px system-ui;color:#fff;text-shadow:0 1px 2px #000';
  label.textContent = 'Webcam hands';
  box.append(video, canvas, label);
  document.body.append(box);
  const colours = ['#4fd1e8', '#b49cff'];
  return {
    draw(f, slots) {
      const w = (canvas.width = 220 * devicePixelRatio), h = (canvas.height = 165 * devicePixelRatio);
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, w, h);
      f.hands.forEach((hand, k) => {
        ctx.fillStyle = colours[slots && slots[1] === k ? 1 : 0];
        for (const p of hand.image) { ctx.beginPath(); ctx.arc((1 - p.x) * w, p.y * h, 2.2 * devicePixelRatio, 0, 7); ctx.fill(); }
      });
      label.textContent = f.pose ? `Webcam hands: ${f.hands.length} seen` : 'Webcam hands: step back so your shoulders show';
    },
    remove() { box.remove(); },
  };
}

/** Start driving the emulator's hands from the webcam. Returns a handle that stops it. */
export async function startWebcamHands(onStatus: (text: string) => void = () => {}): Promise<{ stop(): void }> {
  const device = (window as any).IWER_DEVICE;
  if (!device) throw new Error("The headset emulator isn't running (development builds on localhost only).");
  const vision: Vision = await openVision(onStatus);
  device.primaryInputMode = 'hand';
  const view = preview(vision.video);
  const builder = new CameraFrameBuilder();
  const frame = emptyFrame();
  vision.run((f) => {
    view.draw(f, f.pose ? assignSlots(f.hands, f.pose.image) : null);
    const ok = builder.build(f.hands, f.pose, frame);
    (window as { __webcamFrame?: Frame }).__webcamFrame = frame; // for checking what the app reads back
    const head: V3 = [device.position.x, device.position.y, device.position.z];
    const axes = headAxes(device.quaternion);
    [device.hands.right, device.hands.left].forEach((hand: any, s: number) => {
      const seen = ok && frame.present[s] > 0.5;
      if (seen) {
        // The app measures you with your saved shoulder width, so place the hands with the same one.
        const points = handPoints(frame, s, head, axes, app.settings.shoulderWidth);
        setPose(hand, jointMatrices(points, s === 0, rootOf(hand)));
      }
      if (hand.connected !== seen) hand.connected = seen;
    });
  });
  return { stop() { vision.stop(); view.remove(); } };
}

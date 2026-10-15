// Development only: drive the headset emulator's hands with your own hands through the webcam, so the app can be
// tried on a desktop before a headset arrives. The camera is read the way the dictionary videos were (engine
// CameraFrameBuilder: body-centred hands), then placed in front of the emulated head where a Quest would report them,
// as WebXR's 25 hand joints. The app's own hand sampler reads them back unchanged.
import { NOSE_OFFSET, emptyFrame, type CameraHand, type Frame, type V3 } from '@ashiria/engine';
import { app } from '../app-state.js';
import { SignReader, sameSign, type Reading } from './reader.js';
import { CameraTracker } from './tracker.js';
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
    // +y out of the back: the back of the hand for the fingers; for the thumb (joints 1-4), whose nail faces
    // sideways, across the bone and the palm normal, leaning a little to the back (as in the emulator's own poses).
    let yAxis = k >= 1 && k <= 4 ? add(mul(cross(along, palm), right ? 1 : -1), mul(palm, -0.2)) : mul(palm, -1);
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

const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const BONES = FINGERS.flatMap((c) => [[0, c[0]], [c[0], c[1]], [c[1], c[2]], [c[2], c[3]]]).concat([[5, 9], [9, 13], [13, 17]]);
const COLOURS = ['#4fd1e8', '#b49cff']; // signer's right, left: as in the sign browser

/** The camera view in a corner, drawn as a mirror, with each tracked hand's skeleton (faint while held after a
 * dropout), and buttons to flip a mirrored camera and to show the emulator's own controls. */
function preview(vision: Vision, device: any) {
  const video = vision.video;
  const box = document.createElement('div');
  box.style.cssText = 'position:fixed;right:12px;bottom:12px;width:320px;z-index:2147483000;border-radius:14px;overflow:hidden;'
    + 'background:#0d1b22;box-shadow:0 8px 28px rgba(0,0,0,.4);font:13px/1.3 system-ui;color:#fff';
  const stage = document.createElement('div');
  // The view takes the camera's own shape, so landmarks (fractions of the frame) line up with the picture.
  stage.style.cssText = `position:relative;aspect-ratio:${video.videoWidth || 4}/${video.videoHeight || 3};background:#000`;
  video.style.cssText = 'width:100%;height:100%;object-fit:fill;display:block';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
  stage.append(video, canvas);
  const bar = document.createElement('div');
  bar.style.cssText = 'display:flex;gap:6px;align-items:center;padding:8px 10px;flex-wrap:wrap';
  const label = document.createElement('span');
  label.style.cssText = 'flex:1;min-width:120px';
  const button = (text: string, onClick: () => void) => {
    const b = document.createElement('button');
    b.textContent = text;
    b.style.cssText = 'font:inherit;padding:4px 9px;border-radius:8px;border:1px solid #3a5560;background:#16303a;color:#fff;cursor:pointer';
    b.addEventListener('click', onClick);
    return b;
  };
  // The emulator's panels and transform handles get in the way while the webcam drives the hands.
  const devui = device.devui?.devUIContainer as HTMLElement | undefined;
  const handles = device.devui?.devUICanvas as HTMLElement | undefined;
  const setControls = (on: boolean) => {
    for (const el of [devui, handles]) if (el) el.style.visibility = on ? '' : 'hidden';
    controls.textContent = on ? 'Hide emulator controls' : 'Emulator controls';
  };
  const flip = button('Flip camera', () => { vision.setMirrored(!vision.mirrored); onFlip(); });
  const controls = button('', () => setControls(devui?.style.visibility === 'hidden'));
  setControls(false);
  bar.append(label, flip, controls);
  // What the recogniser reads after each sign, over every dictionary sign, and whether it is the lesson's sign.
  const model = document.createElement('div');
  model.style.cssText = 'padding:0 10px 10px;line-height:1.4';
  model.innerHTML = '<span style="opacity:.7">Model: loading…</span>';
  box.append(stage, bar, model);
  document.body.append(box);
  let onFlip = () => {};

  const esc = (t: string) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
  return {
    onFlip(fn: () => void) { onFlip = fn; },
    modelStatus(text: string) { model.innerHTML = `<span style="opacity:.7">${esc(text)}</span>`; },
    reading(r: Reading | null, lesson: { gloss: string; aliases: string[] } | null) {
      if (!r) { model.innerHTML = '<span style="opacity:.7">Model: that was too short to read. Try again.</span>'; return; }
      const [best, ...rest] = r.guesses;
      const match = lesson && r.guesses.slice(0, 3).some((g) => sameSign(g.sign, lesson.gloss, lesson.aliases));
      const head = r.accepted
        ? `<b style="font-size:16px">${esc(best.sign.name)}</b> <span style="opacity:.7">${best.similarity.toFixed(2)}</span>`
        : `<span style="opacity:.8">Not sure. Best guess</span> <b>${esc(best.sign.name)}?</b> <span style="opacity:.7">${best.similarity.toFixed(2)}</span>`;
      const also = rest.slice(0, 2).map((g) => `${esc(g.sign.name)} ${g.similarity.toFixed(2)}`).join(' · ');
      const lessonLine = lesson ? `<div style="color:${match ? '#5fd38d' : '#e0c060'}">${match ? 'Matches' : 'Doesn\'t match'} the lesson sign "${esc(lesson.gloss)}"</div>` : '';
      model.innerHTML = `<div>Model: ${head}</div><div style="opacity:.7;font-size:12px">then ${also}</div>${lessonLine}`;
    },
    draw(f: CameraFrame, slots: [CameraHand | null, CameraHand | null], held: [boolean, boolean]) {
      const w = (canvas.width = canvas.clientWidth * devicePixelRatio), h = (canvas.height = canvas.clientHeight * devicePixelRatio);
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, w, h);
      // Shown as a mirror. Tracking runs on the camera's own frames (flipped back first if "Flip camera" is on), so
      // landmark x is mirrored for drawing.
      video.style.transform = vision.mirrored ? 'none' : 'scaleX(-1)';
      const at = (p: { x: number; y: number }): [number, number] => [(1 - p.x) * w, p.y * h];
      if (f.pose) {
        ctx.strokeStyle = 'rgba(255,255,255,.5)';
        ctx.lineWidth = 2 * devicePixelRatio;
        for (const [a, b] of [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]) {
          ctx.beginPath(); ctx.moveTo(...at(f.pose.image[a])); ctx.lineTo(...at(f.pose.image[b])); ctx.stroke();
        }
      }
      slots.forEach((hand, s) => {
        if (!hand) return;
        ctx.globalAlpha = held[s] ? 0.35 : 1;
        ctx.strokeStyle = ctx.fillStyle = COLOURS[s];
        ctx.lineWidth = 2.2 * devicePixelRatio;
        for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(...at(hand.image[a])); ctx.lineTo(...at(hand.image[b])); ctx.stroke(); }
        for (const p of hand.image) { ctx.beginPath(); ctx.arc(...at(p), 2.6 * devicePixelRatio, 0, 7); ctx.fill(); }
      });
      ctx.globalAlpha = 1;
      const n = slots.filter(Boolean).length;
      label.textContent = !f.pose ? 'Sit back so your head and shoulders show'
        : n === 0 ? 'Raise your hands into view' : `Tracking ${n === 2 ? 'both hands' : slots[0] ? 'your right hand' : 'your left hand'}`;
    },
    remove() { box.remove(); setControls(true); },
  };
}

/** Start driving the emulator's hands from the webcam. Returns a handle that stops it. */
export async function startWebcamHands(onStatus: (text: string) => void = () => {}): Promise<{ stop(): void }> {
  const device = (window as any).IWER_DEVICE;
  if (!device) throw new Error("The headset emulator isn't running (development builds on localhost only).");
  const vision: Vision = await openVision(onStatus);
  device.primaryInputMode = 'hand';
  const view = preview(vision, device);
  const tracker = new CameraTracker();
  // The recogniser reads unsmoothed frames, as the references were made.
  const readTracker = new CameraTracker();
  readTracker.smooth = false;
  readTracker.holdMs = 0;
  const readFrame = emptyFrame();
  view.onFlip(() => { tracker.reset(); readTracker.reset(); });
  // A webcam sees hands less precisely than a headset: start the lesson check at its gentlest.
  app.updateSettings({ strictness: 2 });
  let reader: SignReader | null = null;
  SignReader.load((t) => t && view.modelStatus(`Model: ${t.toLowerCase()}`)).then(
    (r) => { reader = r; view.modelStatus(`Model: ready. Sign something; it knows ${r.signCount} signs.`); },
    (err) => view.modelStatus(`Model unavailable: ${(err as Error).message}`));
  const frame = emptyFrame();
  vision.run((f) => {
    const { ok, slots, held } = tracker.update(f, frame);
    view.draw(f, slots, held);
    if (reader && readTracker.update(f, readFrame).ok) {
      const pending = reader.push(readFrame, f.time);
      if (pending) {
        const sign = app.current;
        view.modelStatus('Model: reading…');
        pending.then((r) => view.reading(r, sign ? { gloss: sign.gloss, aliases: sign.aliases } : null), () => view.modelStatus('Model: could not read that sign.'));
      }
    }
    Object.assign(window, { __webcamFrame: frame, __webcamTracker: tracker }); // for checking what the app reads back
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

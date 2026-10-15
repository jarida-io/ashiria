// Camera input (src/camera.ts) against the Python reference on spec/golden/camera_v1.json: synthetic MediaPipe
// landmarks in a random camera orientation, read by from_mediapipe and assign_slots.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { assignSlots, bodyFrame, cameraFrame, emptyFrame, type CameraHand, type Point } from '../src/index.ts';

const g = JSON.parse(readFileSync(new URL('../../../../spec/golden/camera_v1.json', import.meta.url), 'utf8'));
const pts = (a: number[][]): Point[] => a.map(([x, y, z]) => ({ x, y, z }));
const flat = (a: unknown): number[] => (Array.isArray(a) ? a.flatMap(flat) : [a as number]);

function close(actual: ArrayLike<number>, expected: number[], tol: number, what: string) {
  assert.equal(actual.length, expected.length, `${what}: length`);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  assert.ok(worst <= tol, `${what}: max difference ${worst.toExponential(2)} > ${tol}`);
}

test('camera_v1: body frame and hands match from_mediapipe', () => {
  g.frames.forEach((f: any, t: number) => {
    const body = bodyFrame(pts(f.pose_world));
    assert.ok(body, `frame ${t}: body frame`);
    const out = emptyFrame();
    cameraFrame(body!, [0, 1].map((s) => (f.hand_present[s] ? pts(f.hand_world[s]) : null)) as [Point[] | null, Point[] | null], out);
    close(out.handRel, flat(f.hand_rel), 1e-7, `frame ${t} hand_rel`);
    close(out.wristPos, flat(f.wrist_pos), 1e-7, `frame ${t} wrist_pos`);
    close(out.present, flat(f.present), 0, `frame ${t} present`);
  });
});

test('camera_v1: hands go to the same slots as assign_slots', () => {
  for (const c of g.slots) {
    const hands: CameraHand[] = c.wrists.map(([x, y]: number[], k: number) => ({ image: [{ x, y, z: 0 }], world: [], label: c.labels[k] }));
    let poseImage: Point[] | null = null;
    if (c.pose_wrists) {
      poseImage = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
      poseImage[16] = { x: c.pose_wrists[0][0], y: c.pose_wrists[0][1], z: 0 };
      poseImage[15] = { x: c.pose_wrists[1][0], y: c.pose_wrists[1][1], z: 0 };
    }
    const slots = assignSlots(hands, poseImage);
    const expected: [number, number] = [c.slots['0'] ?? -1, c.slots['1'] ?? -1];
    assert.deepEqual(slots, expected, JSON.stringify(c));
  }
});

test('camera: no body frame when the shoulders are missing', () => {
  const world = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
  assert.equal(bodyFrame(world), null);
});

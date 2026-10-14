import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  emptyFrame, forwardFromQuaternion, fromWebXRPosition, fromWebXRQuaternion, HINT_TEXT, hintText, restHeight, StudioTakeWriter,
  TickClock, WEBXR25, OPENXR26,
} from '../src/index.ts';

test('tick clock samples at a fixed rate and drops stale backlog', () => {
  const clock = new TickClock(30);
  let ticks = 0;
  for (let i = 0; i < 90; i++) ticks += clock.advance(1 / 90);
  assert.equal(ticks, 30);
  assert.equal(clock.advance(5), 3);
  assert.equal(clock.advance(0), 0);
});

test('rest height from resting wrists', () => {
  assert.equal(restHeight(Array(20).fill(-2)), -1.75);
  assert.equal(restHeight(Array(20).fill(-1)), -1.2);
  assert.equal(restHeight(Array(20).fill(-3)), -1.9);
  assert.equal(restHeight([-2], -1.5), -1.5);
});

test('every hint code has words', () => {
  for (const code of Object.keys(HINT_TEXT))
    assert.ok(!hintText({ code, severity: 2, finger: 'ring', direction: 'up', signId: 'letter-s' }, () => 'S').includes('{'), code);
  assert.equal(hintText({ code: 'finger.straighten', severity: 2, finger: 'ring' }), 'Straighten your ring finger.');
  assert.equal(hintText({ code: 'move', severity: 2, direction: 'down' }), 'Move your hand lower.');
});

test('WebXR joints are the OpenXR joints without the palm', () => {
  assert.deepEqual(WEBXR25, OPENXR26.map((i) => i - 1));
  assert.equal(WEBXR25[0], 0);  // WebXR "wrist" is joint 0
});

test('WebXR (right-handed) to left-handed: looking down -z is forward', () => {
  assert.deepEqual(fromWebXRPosition(1, 2, 3), [1, 2, -3]);
  // A WebXR viewer rotated 90° to its left (about +y): it looks towards -x.
  const s = Math.SQRT1_2;
  const [x, y, z, w] = fromWebXRQuaternion(0, s, 0, s);
  const f = forwardFromQuaternion(x, y, z, w);
  assert.ok(Math.abs(f[0] + 1) < 1e-9 && Math.abs(f[2]) < 1e-9, `forward ${f}`);
});

test('studio writer produces the documented format', () => {
  const w = new StudioTakeWriter({
    takeId: 't', sessionId: 's', signId: 'letter-a', packId: 'ksl-core', packVersion: '0.1.0', signer: 'S01', role: 'reference',
    handedness: 'right', consentRef: '', shoulderWidthM: 0.38, shoulderCalibrated: false, restHeight: -1.5, device: 'Quest 3S',
    appVersion: '0.1.0', tickHz: 30, recordedAt: '2026-10-14T00:00:00Z',
  });
  const hand = { pos: new Float64Array(75), rot: new Float64Array(100), tracked: true, highConf: true, inferred: false, systemGesture: false };
  w.addFrame(0, [0, 1.6, 0], [0, 0, 0, 1], hand, { ...hand, tracked: false }, emptyFrame());
  const take = JSON.parse(w.toJSON());
  assert.equal(take.format, 'ashiria.studio-take');
  assert.equal(take.joint_layout, 'webxr25');
  assert.equal(take.frames.right.pos.length, 75);
  assert.equal(take.frames.left.tracked[0], 0);
  assert.deepEqual(take.trim, { start: 0, end: 1 });
});

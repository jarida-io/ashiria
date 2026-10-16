// Parity with the Python reference on spec/golden (the same vectors the C# tests use).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import {
  AttemptSegmenter, activeWindow, defaultParams, descriptor, dtw, FEATURE_DIM, fingerErrors, flexion, FrameBuilder, gapFill,
  hand, holdFrames, lookalikesOf, mirror, paramsFromJson, parsePack, prepare, recognise, resample, segmentFromNested,
  selfCheck, sequenceFeatures, steadiestFrame, trim, verify, emptyFrame, type Segment,
} from '../src/index.ts';

const golden = (name: string) => new URL(`../../../../spec/golden/${name}`, import.meta.url);
const read = (name: string) => readFileSync(golden(name), 'utf8');
const json = (name: string) => JSON.parse(read(name));

function close(actual: ArrayLike<number>, expected: number[], tol: number, what: string) {
  assert.equal(actual.length, expected.length, `${what}: length`);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  assert.ok(worst <= tol, `${what}: max difference ${worst.toExponential(2)} > ${tol}`);
}
const flat = (a: unknown): number[] => (Array.isArray(a) ? a.flatMap(flat) : [a as number]);

test('engine_v1: canonicalisation, features, descriptor, DTW, finger errors, window', () => {
  const g = json('engine_v1.json');
  const p = defaultParams();
  const descs = [];
  const keyFlex: number[][] = [];
  for (const c of g.cases) {
    let seg: Segment = segmentFromNested(c.segment);
    if (c.mirror) seg = mirror(seg);
    const trimmed = trim(gapFill(seg, g.gap_fill_frames), g.rest_height, g.trim_margin_frames)!;
    assert.equal(trimmed.length, c.trimmed_length, `${c.name}: trimmed length`);
    const res = resample(trimmed, g.seq_len);
    assert.ok(Math.abs(res.fps - c.resampled_fps) < 1e-3, `${c.name}: fps`);
    assert.deepEqual(Array.from(res.present), flat(c.resampled_present), `${c.name}: present`);
    const feats = sequenceFeatures(res);
    close(feats, flat(c.features), g.tolerance.features, `${c.name} features`);
    const d = descriptor(feats, p.weights);
    close(d.data, flat(c.descriptor), g.tolerance.descriptor, `${c.name} descriptor`);
    descs.push(d);
    keyFlex.push(flexion(hand(res, c.key_frame, 0)));
    const win = activeWindow(trimmed);
    assert.equal(win.length, c.window_length, `${c.name}: window length`);
    assert.deepEqual(holdFrames(win), c.window_hold, `${c.name}: hold frames`);
    assert.equal(steadiestFrame(resample(win, g.seq_len)), c.window_steadiest, `${c.name}: steadiest frame`);
    assert.equal(feats.length, g.seq_len * FEATURE_DIM);
  }
  for (const pair of g.dtw) assert.ok(Math.abs(dtw(descs[pair.a], descs[pair.b]) - pair.distance) <= g.tolerance.dtw, `dtw ${pair.a}-${pair.b}`);
  for (const e of g.finger_errors) close(fingerErrors(keyFlex[e.attempt], keyFlex[e.reference]), e.errors, g.tolerance.finger_errors, 'finger errors');
});

async function loadPack(dir: string) {
  return parsePack(read(`${dir}/manifest.json`), read(`${dir}/catalogue.json`), read(`${dir}/references.json`));
}

test('runtime_v1: pack loading, self-check and damage detection', async () => {
  const g = json('runtime_v1.json');
  const pack = await loadPack(g.pack);
  assert.equal(pack.signs.size, pack.catalogue.length);
  const check = selfCheck(pack);
  assert.ok(check.ok, `self-check ${check.actual} vs ${check.expected}`);
  const damaged = read(`${g.pack}/references.json`).replace('"tau":', '"tau": ');
  await assert.rejects(parsePack(read(`${g.pack}/manifest.json`), read(`${g.pack}/catalogue.json`), damaged), /does not match/);
});

test('pack: the avatar coach is optional, hash-checked when listed', async () => {
  const g = json('runtime_v1.json');
  const [cat, refs] = [read(`${g.pack}/catalogue.json`), read(`${g.pack}/references.json`)];
  const id = JSON.parse(cat).signs[0].id;
  const coach = JSON.stringify({ signs: { [id]: { fps: 24, frames: 1, present: [[1, 0]], learn: { hands: ['right hand'], steps: [] } } } });
  const hash = createHash('sha256').update(coach).digest('hex');
  const manifest = JSON.parse(read(`${g.pack}/manifest.json`));
  manifest.files.coach = { path: 'coach.json', sha256: hash };
  const pack = await parsePack(JSON.stringify(manifest), cat, refs, true, coach);
  assert.equal(pack.coach.get(id)?.fps, 24);
  assert.equal((await loadPack(g.pack)).coach.size, 0);
  await assert.rejects(parsePack(JSON.stringify(manifest), cat, refs, true, coach.replace('24', '25')), /coach.json does not match/);
});

test('runtime_v1: prepare, Quest frames, segmenter', async () => {
  const g = json('runtime_v1.json');
  const p = (await loadPack(g.pack)).params;
  for (const c of g.prepare) {
    const out = prepare(segmentFromNested(c.segment), p, c.rest_height)!;
    assert.ok(Math.abs(out.fps - c.fps) < 1e-3, `${c.name}: fps`);
    assert.deepEqual(Array.from(out.present), flat(c.present), `${c.name}: present`);
    close(descriptor(sequenceFeatures(out), p.weights).data, flat(c.descriptor), g.tolerance.descriptor, `${c.name} descriptor`);
  }
  for (const c of g.quest) {
    const builder = new FrameBuilder();
    builder.axes = c.last_axes;
    const input = (side: 'right' | 'left') => ({
      joints: flat(c.hands[side].joints), layout: 'openxr26' as const, tracked: c.hands[side].tracked,
      highConfidence: true, inferred: false, systemGesture: false,
    });
    const f = builder.build(c.head_pos, c.head_forward, input('right'), input('left'), c.left_handed, c.shoulder_width);
    assert.deepEqual(Array.from(f.present), c.present);
    close(f.wristPos, flat(c.wrist_pos), g.tolerance.quest, 'quest wrist');
    close(f.handRel, flat(c.hand_rel), g.tolerance.quest, 'quest joints');
  }
  for (const c of g.segmenter) {
    const seg = segmentFromNested(c.segment);
    const s = new AttemptSegmenter(p.segmenter);
    const found: string[] = [];
    for (let t = 0; t < seg.length; t++) {
      const a = s.push({
        handRel: seg.handRel.subarray(t * 126, (t + 1) * 126), wristPos: seg.wristPos.subarray(t * 6, t * 6 + 6),
        present: seg.present.subarray(t * 2, t * 2 + 2),
      });
      if (a) found.push(`${a.start}-${a.end} ${a.reason}`);
    }
    assert.deepEqual(found, c.attempts.map((a: any) => `${a.start}-${a.end} ${a.reason}`), c.name);
  }
});

test('runtime_v1: Verify and Recognise', async () => {
  const g = json('runtime_v1.json');
  const pack = await loadPack(g.pack);
  const tol = g.tolerance.verify;
  for (const c of g.verify) {
    const r = verify(segmentFromNested(c.attempt), pack.signs.get(c.sign)!, pack.params, c.tolerance_scale, lookalikesOf(pack, c.sign));
    assert.equal(r.status, c.status, c.name);
    assert.equal(r.take, c.take, `${c.name}: take`);
    assert.ok(Math.abs(r.score - c.score) <= tol, `${c.name}: score`);
    assert.deepEqual(Object.keys(r.components).sort(), Object.keys(c.components).sort(), `${c.name}: components`);
    for (const [k, v] of Object.entries(c.components)) assert.ok(Math.abs(r.components[k] - (v as number)) <= tol, `${c.name}: ${k}`);
    close(r.fingerErrors, c.finger_errors, tol, `${c.name}: finger errors`);
    const hints = (hs: any[]) => hs.map((h) => `${h.code}|${h.finger ?? ''}|${h.direction ?? ''}|${h.sign_id ?? h.signId ?? ''}`);
    assert.deepEqual(hints(r.hints), hints(c.hints), `${c.name}: hints`);
  }
  const candidates = pack.catalogue.filter((s) => s.recognise).map((s) => pack.signs.get(s.id)!);
  for (const c of g.recognise) {
    const p = paramsFromJson({ ...pack.params, recognise: { ...pack.params.recognise, accept_z: c.accept_z, min_margin_z: c.min_margin_z } });
    const r = recognise(segmentFromNested(c.attempt), candidates, p);
    assert.equal(r.status, c.status, c.name);
    assert.equal(r.best, c.best);
    assert.equal(r.runnerUp, c.runner_up);
    assert.ok(Math.abs(r.bestZ - c.best_z) <= g.tolerance.recognise, `${c.name}: best z`);
    assert.ok(Math.abs(r.runnerUpZ - c.runner_up_z) <= g.tolerance.recognise, `${c.name}: runner-up z`);
    assert.ok(Math.abs(r.confidence - c.confidence) <= g.tolerance.recognise, `${c.name}: confidence`);
    assert.deepEqual(r.ranking.map((x) => x.id), c.ranking.map((x: any) => x[0]), `${c.name}: ranking`);
  }
});

test('studio take: canonical frames from raw poses match Python', () => {
  const take = json('studio_take_v1.json');
  const f = take.frames;
  const builder = new FrameBuilder();
  const out = emptyFrame();
  for (let t = 0; t < f.n; t++) {
    const q = f.head_rot.slice(t * 4, t * 4 + 4);
    const fwd: [number, number, number] = [2 * (q[0] * q[2] + q[3] * q[1]), 2 * (q[1] * q[2] - q[3] * q[0]), 1 - 2 * (q[0] * q[0] + q[1] * q[1])];
    const input = (side: 'right' | 'left') => ({
      joints: f[side].pos.slice(t * 78, t * 78 + 78), layout: 'openxr26' as const, tracked: f[side].tracked[t] === 1,
      highConfidence: f[side].high_conf[t] === 1, inferred: f[side].inferred[t] === 1, systemGesture: f[side].system_gesture[t] === 1,
    });
    builder.build(f.head_pos.slice(t * 3, t * 3 + 3), fwd, input('right'), input('left'), take.handedness === 'left', take.shoulder_width_m, out);
    assert.deepEqual(Array.from(out.present), f.canonical.present.slice(t * 2, t * 2 + 2));
    for (let s = 0; s < 2; s++) {
      if (out.present[s] < 0.5) continue;
      close(out.wristPos.subarray(s * 3, s * 3 + 3), f.canonical.wrist_pos.slice((t * 2 + s) * 3, (t * 2 + s) * 3 + 3), 1e-4, `wrist ${t}`);
      close(out.handRel.subarray(s * 63, s * 63 + 63), f.canonical.hand_rel.slice((t * 2 + s) * 63, (t * 2 + s) * 63 + 63), 1e-4, `joints ${t}`);
    }
  }
});

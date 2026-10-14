/**
 * Studio (development builds, ?studio): every attempt the segmenter cuts is saved as a take
 * (spec/STUDIO_FORMAT.md) to training/data/studio/<date>/ on this Mac through the dev server.
 * These are the team's own recordings; nothing is sent anywhere else.
 */
import { StudioTakeWriter, type Attempt } from '@ashiria/engine';
import { app } from './app-state.js';
import type { HandSamplerSystem } from './hand-sampler.js';

const MARGIN = 10;  // ticks kept either side of the attempt
const session = new Date().toISOString().slice(0, 10);

export async function saveTake(sampler: HandSamplerSystem, attempt: Attempt, signId: string): Promise<number> {
  if (!import.meta.env.DEV) throw new Error('Studio is only available in development builds');
  const samples = sampler.samples(attempt.start - MARGIN, attempt.end + MARGIN);
  if (!samples.length) throw new Error('no samples for this attempt');
  const s = app.settings;
  const writer = new StudioTakeWriter({
    takeId: crypto.randomUUID(), sessionId: session, signId, packId: app.pack?.packId ?? '', packVersion: app.pack?.version ?? '',
    signer: 'S01', role: 'reference', handedness: s.leftHanded ? 'left' : 'right', consentRef: '', shoulderWidthM: s.shoulderWidth,
    shoulderCalibrated: false, restHeight: s.restHeight, device: /OculusBrowser|Quest/.test(navigator.userAgent) ? 'Meta Quest' : 'emulator',
    appVersion: '0.1.0', tickHz: 30, recordedAt: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  }, 'webxr25');
  const t0 = samples[0].time;
  for (const x of samples) {
    const hand = (h: typeof x.hands.right) => ({ pos: h.pos, rot: h.rot, tracked: h.tracked, highConf: h.tracked, inferred: false, systemGesture: false });
    writer.addFrame(x.time - t0, x.head, x.headRot, hand(x.hands.right), hand(x.hands.left), x.frame);
  }
  const first = samples[0].tick;
  const res = await fetch('/dev/studio', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: writer.toJSON(attempt.start - first, attempt.end - first + 1),
  });
  if (!res.ok) throw new Error(`dev server: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { take: number }).take;
}

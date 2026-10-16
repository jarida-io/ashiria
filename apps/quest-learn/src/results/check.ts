// Reads a dictionary clip the way the web app reads a camera: every frame through MediaPipe, the same tracker and the
// exported recogniser (webcam/reader.ts). Frames are visited by seeking, so none are skipped however slow tracking
// is, and a few are kept as thumbnails with the landmarks MediaPipe found. Development only: the clips stay on this Mac.
import { emptyFrame, type Point } from '@ashiria/engine';
import type { Reading, SignReader } from '../webcam/reader.js';
import { CameraTracker } from '../webcam/tracker.js';
import type { CameraFrame, Landmarkers } from '../webcam/vision.js';

const RATE = 25;           // frames read per second of video
const CLOSE_S = 1.2;       // hands down after the clip, so the last sign's attempt ends
const BONES = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17]];

export interface ClipReading {
  reading: Reading | null;
  /** Every attempt the reader cut from the clip (a clip can hold more than one). */
  readings: Reading[];
  thumbs: HTMLCanvasElement[];
  frames: number;
  /** Frames where MediaPipe found at least one hand. */
  handFrames: number;
}

let clock = 0; // MediaPipe's timestamps must keep increasing across clips
const nextMs = () => (clock += 1000 / RATE);

async function open(url: string): Promise<HTMLVideoElement> {
  const video = document.createElement('video');
  Object.assign(video, { src: url, muted: true, playsInline: true, preload: 'auto' });
  await new Promise<void>((ok, fail) => { video.onloadeddata = () => ok(); video.onerror = () => fail(new Error(`cannot load ${url}`)); });
  return video;
}

async function seek(video: HTMLVideoElement, t: number): Promise<void> {
  if (Math.abs(video.currentTime - t) < 1e-3) return;
  await new Promise<void>((ok) => { video.onseeked = () => ok(); video.currentTime = t; });
}

function thumb(source: CanvasImageSource, w: number, h: number, f: CameraFrame | null): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = 176;
  c.height = Math.round((176 * h) / w);
  const g = c.getContext('2d')!;
  g.drawImage(source, 0, 0, c.width, c.height);
  for (const hand of f?.hands ?? []) {
    g.strokeStyle = hand.label === 'Right' ? '#ffb347' : '#5ec8ff';
    g.lineWidth = 1.5;
    const p = (q: Point) => [q.x * c.width, q.y * c.height] as const;
    for (const chain of BONES) {
      g.beginPath();
      chain.forEach((k, i) => (i ? g.lineTo(...p(hand.image[k])) : g.moveTo(...p(hand.image[k]))));
      g.stroke();
    }
  }
  return c;
}

/** Read one clip. `mirror` flips every frame first (a mirrored camera, or a left-handed signer). */
export async function readClip(url: string, marks: Landmarkers, reader: SignReader, mirror = false,
  thumbAt: number[] = [0.2, 0.4, 0.6, 0.8]): Promise<ClipReading> {
  const video = await open(url);
  const w = video.videoWidth, h = video.videoHeight;
  const flip = document.createElement('canvas');
  Object.assign(flip, { width: w, height: h });
  const fg = flip.getContext('2d')!;
  fg.setTransform(-1, 0, 0, 1, w, 0);
  const tracker = new CameraTracker();
  tracker.smooth = false; // as the references were made (webcam/emulate.ts reads the same way)
  tracker.holdMs = 0;
  reader.reset();
  const out = emptyFrame();
  const readings: Reading[] = [];
  const pending: Promise<Reading | null>[] = [];
  const thumbs: HTMLCanvasElement[] = [];
  let frames = 0, handFrames = 0, lastPose: CameraFrame['pose'] = null;
  const times = Array.from({ length: Math.max(1, Math.floor(video.duration * RATE)) }, (_, k) => k / RATE);
  const thumbFrames = new Set(thumbAt.map((x) => Math.min(times.length - 1, Math.round(x * (times.length - 1)))));
  for (let k = 0; k < times.length; k++) {
    await seek(video, times[k]);
    let source: HTMLVideoElement | HTMLCanvasElement = video;
    if (mirror) { fg.drawImage(video, 0, 0); source = flip; }
    const f = { ...marks.detect(source, nextMs()), time: times[k] * 1000 };
    frames++;
    if (f.hands.length) handFrames++;
    if (f.pose) lastPose = f.pose;
    if (thumbFrames.has(k)) thumbs.push(thumb(source, w, h, f));
    if (tracker.update(f, out).ok) {
      const p = reader.push(out, f.time);
      if (p) pending.push(p);
    }
  }
  const end = times.length / RATE;
  for (let k = 1; k <= CLOSE_S * RATE; k++) { // the signer's hands at rest: no hands in view
    const f: CameraFrame = { hands: [], pose: lastPose, time: (end + k / RATE) * 1000 };
    if (tracker.update(f, out).ok) {
      const p = reader.push(out, f.time);
      if (p) pending.push(p);
    }
  }
  for (const r of await Promise.all(pending)) if (r) readings.push(r);
  video.removeAttribute('src');
  video.load();
  // The clip's sign is its clearest attempt (a hand settling at the end can make a short extra one).
  const reading = readings.reduce<Reading | null>((a, r) => (!a || r.guesses[0].similarity > a.guesses[0].similarity ? r : a), null);
  return { reading, readings, thumbs, frames, handFrames };
}

const stills = new Map<string, Promise<HTMLCanvasElement[]>>();

/** Two frames from the middle of a clip, for showing what a guess looks like. */
export function clipStills(url: string): Promise<HTMLCanvasElement[]> {
  if (!stills.has(url)) {
    stills.set(url, (async () => {
      const video = await open(url);
      const out = [];
      for (const x of [0.4, 0.6]) {
        await seek(video, x * video.duration);
        out.push(thumb(video, video.videoWidth, video.videoHeight, null));
      }
      video.removeAttribute('src');
      video.load();
      return out;
    })());
  }
  return stills.get(url)!;
}

/** Shared canonicalisation (spec §5): mirror, gap fill, trim, active window, resample. Port of canonical.py. */
import { COUNT, DOMINANT } from './joints.ts';
import { median, roundHalfEven } from './math.ts';
import { emptySegment, isPresent, jointAt, slice, withFps, wristAt, type Segment } from './segment.ts';

/** For each frame, the nearest frame where `valid` is true (ties go to the earlier frame); -1 if none. */
export function nearestValid(valid: ArrayLike<boolean>): Int32Array {
  const n = valid.length;
  const prev = new Int32Array(n), next = new Int32Array(n), out = new Int32Array(n);
  let last = -1;
  for (let t = 0; t < n; t++) { if (valid[t]) last = t; prev[t] = last; }
  last = -1;
  for (let t = n - 1; t >= 0; t--) { if (valid[t]) last = t; next[t] = last; }
  for (let t = 0; t < n; t++) {
    const l = prev[t], r = next[t];
    out[t] = l < 0 ? r : r < 0 ? l : t - l <= r - t ? l : r;
  }
  return out;
}

/** Left-right mirror: negate x and swap the hand slots (makes a left-handed signer right-handed). */
export function mirror(seg: Segment): Segment {
  const s = emptySegment(seg.length, seg.fps);
  for (let t = 0; t < seg.length; t++)
    for (let slot = 0; slot < 2; slot++) {
      const from = 1 - slot;
      for (let j = 0; j < COUNT; j++) {
        const a = jointAt(t, slot, j), b = jointAt(t, from, j);
        s.handRel[a] = -seg.handRel[b]; s.handRel[a + 1] = seg.handRel[b + 1]; s.handRel[a + 2] = seg.handRel[b + 2];
      }
      const a = wristAt(t, slot), b = wristAt(t, from);
      s.wristPos[a] = -seg.wristPos[b]; s.wristPos[a + 1] = seg.wristPos[b + 1]; s.wristPos[a + 2] = seg.wristPos[b + 2];
      s.present[t * 2 + slot] = seg.present[t * 2 + from];
    }
  return s;
}

/** Linearly fill dropouts of at most `maxGap` frames between two tracked frames. */
export function gapFill(seg: Segment, maxGap: number): Segment {
  const s = withFps(seg, seg.fps);
  for (let slot = 0; slot < 2; slot++) {
    let a = -1;
    for (let b = 0; b < seg.length; b++) {
      if (!isPresent(seg, b, slot)) continue;
      const gap = b - a - 1;
      if (a >= 0 && gap > 0 && gap <= maxGap) {
        for (let k = 1; k <= gap; k++) {
          const w = k / (gap + 1), t = a + k;
          for (let c = 0; c < 3; c++) {
            s.wristPos[wristAt(t, slot) + c] = seg.wristPos[wristAt(a, slot) + c] * (1 - w) + seg.wristPos[wristAt(b, slot) + c] * w;
            for (let j = 0; j < COUNT; j++)
              s.handRel[jointAt(t, slot, j) + c] = seg.handRel[jointAt(a, slot, j) + c] * (1 - w) + seg.handRel[jointAt(b, slot, j) + c] * w;
          }
          s.present[t * 2 + slot] = 1;
        }
      }
      a = b;
    }
  }
  return s;
}

/** Keep the frames from the first to the last moment a hand is up in signing space, plus a margin.
 * Null if no hand is ever present. */
export function trim(seg: Segment, restHeight: number, margin: number): Segment | null {
  const n = seg.length;
  const active: boolean[] = [], any: boolean[] = [];
  let anyActive = false;
  for (let t = 0; t < n; t++) {
    let height = -Infinity, has = false;
    for (let slot = 0; slot < 2; slot++)
      if (isPresent(seg, t, slot)) { has = true; height = Math.max(height, seg.wristPos[wristAt(t, slot) + 1]); }
    any.push(has);
    active.push(has && height >= restHeight);
    anyActive ||= has && height >= restHeight;
  }
  const use = anyActive ? active : any;
  const first = use.indexOf(true), last = use.lastIndexOf(true);
  if (first < 0) return null;
  return slice(seg, Math.max(0, first - margin), Math.min(n, last + margin + 1));
}

/** Cut the rise from rest and the drop back: keep frames where the dominant wrist is above `fraction` of its own
 * height range. Unchanged if that range is under `minRange` shoulder widths. */
export function activeWindow(seg: Segment, fraction = 0.3, margin = 2, minFrames = 6, minRange = 0.3): Segment {
  const n = seg.length;
  let count = 0, lo = Infinity, hi = -Infinity;
  for (let t = 0; t < n; t++) {
    if (!isPresent(seg, t, DOMINANT)) continue;
    count++;
    const y = seg.wristPos[wristAt(t, DOMINANT) + 1];
    lo = Math.min(lo, y); hi = Math.max(hi, y);
  }
  if (count < 3 || hi - lo < minRange) return seg;
  const threshold = lo + fraction * (hi - lo);
  let first = -1, last = -1;
  for (let t = 0; t < n; t++)
    if (isPresent(seg, t, DOMINANT) && seg.wristPos[wristAt(t, DOMINANT) + 1] >= threshold) { if (first < 0) first = t; last = t; }
  const a = Math.max(0, first - margin), b = Math.min(n, last + margin + 1);
  return b - a >= minFrames ? slice(seg, a, b) : seg;
}

export const HOLD_RADIUS = 0.12;

/** Frames where the wrist, after a centred 5-frame moving average (zero-padded at the ends), stays within
 * HOLD_RADIUS of its median over the present frames. */
export function holdFrames(seg: Segment, slot = DOMINANT, smooth = 5): boolean[] {
  const n = seg.length;
  const hold = new Array<boolean>(n).fill(false);
  const ok: number[] = [];
  for (let t = 0; t < n; t++) if (isPresent(seg, t, slot)) ok.push(t);
  if (ok.length < 3) return hold;
  let p: number[][] = [];
  for (let t = 0; t < n; t++) p.push([0, 1, 2].map((c) => seg.wristPos[wristAt(t, slot) + c]));
  if (n >= smooth) {
    const half = (smooth - 1) >> 1;
    p = p.map((_, t) => [0, 1, 2].map((c) => {
      let sum = 0;
      for (let m = t - half; m <= t + half; m++) if (m >= 0 && m < n) sum += p[m][c] / smooth;
      return sum;
    }));
  }
  const centre = [0, 1, 2].map((c) => median(ok.map((t) => p[t][c])));
  for (const t of ok) hold[t] = Math.hypot(p[t][0] - centre[0], p[t][1] - centre[1], p[t][2] - centre[2]) < HOLD_RADIUS;
  return hold;
}

/** The frame in the middle 60% where the dominant wrist moves least (first if tied). */
export function steadiestFrame(seg: Segment): number {
  const n = seg.length;
  const lo = Math.floor(n * 0.2), hi = Math.max(Math.floor(n * 0.8), Math.floor(n * 0.2) + 1);
  let best = -1, bestSpeed = Infinity;
  for (let t = Math.max(lo, 1); t < Math.min(hi, n); t++) {
    if (!isPresent(seg, t, DOMINANT)) continue;
    const a = wristAt(t, DOMINANT), b = wristAt(t - 1, DOMINANT);
    const speed = Math.hypot(seg.wristPos[a] - seg.wristPos[b], seg.wristPos[a + 1] - seg.wristPos[b + 1], seg.wristPos[a + 2] - seg.wristPos[b + 2]);
    if (speed < bestSpeed) { bestSpeed = speed; best = t; }
  }
  return best >= 0 ? best : n >> 1;
}

/** Resample to exactly n frames: positions interpolate linearly (untracked frames hold the nearest tracked pose
 * first); presence takes the nearest frame (round half to even). */
export function resample(seg: Segment, n: number): Segment {
  const L = seg.length;
  const held = withFps(seg, seg.fps);
  for (let slot = 0; slot < 2; slot++) {
    const ok: boolean[] = [];
    for (let t = 0; t < L; t++) ok.push(isPresent(seg, t, slot));
    if (!ok.includes(true)) continue;
    const near = nearestValid(ok);
    for (let t = 0; t < L; t++) {
      held.wristPos.set(seg.wristPos.subarray(wristAt(near[t], slot), wristAt(near[t], slot) + 3), wristAt(t, slot));
      held.handRel.set(seg.handRel.subarray(jointAt(near[t], slot, 0), jointAt(near[t], slot, 0) + COUNT * 3), jointAt(t, slot, 0));
    }
  }
  const duration = Math.max(L - 1, 1) / seg.fps;
  const s = emptySegment(n, (n - 1) / duration);
  const step = n > 1 ? (L - 1) / (n - 1) : 0;
  for (let k = 0; k < n; k++) {
    const src = k === n - 1 && n > 1 ? L - 1 : k * step;
    const i0 = Math.floor(src), i1 = Math.min(i0 + 1, L - 1), f = src - i0;
    const nearest = roundHalfEven(src);
    for (let slot = 0; slot < 2; slot++) {
      for (let c = 0; c < 3; c++) {
        s.wristPos[wristAt(k, slot) + c] = held.wristPos[wristAt(i0, slot) + c] * (1 - f) + held.wristPos[wristAt(i1, slot) + c] * f;
        for (let j = 0; j < COUNT; j++)
          s.handRel[jointAt(k, slot, j) + c] = held.handRel[jointAt(i0, slot, j) + c] * (1 - f) + held.handRel[jointAt(i1, slot, j) + c] * f;
      }
      s.present[k * 2 + slot] = seg.present[nearest * 2 + slot];
    }
  }
  return s;
}

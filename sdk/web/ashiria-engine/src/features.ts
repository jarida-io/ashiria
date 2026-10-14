/** Per-frame features (spec §6, 197 floats per frame). Port of features.py. Output is float32, as in Python. */
import { COUNT, FINGERS, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST } from './joints.ts';
import { angle, cross, dot, EPS, norm, normalize, sub, type V3 } from './math.ts';
import { hand, isPresent, wristAt, type Segment } from './segment.ts';

export const SPEC_VERSION = 1;
export const HAND_DIM = 97;
export const FEATURE_DIM = 2 * HAND_DIM + 3;
const TIP_PAIRS = [[4, 8], [4, 12], [4, 16], [4, 20], [8, 12]] as const;

/** Hand axes: y = wrist to middle knuckle, z = palm normal, x = y × z; plus the palm length. */
export function handAxes(h: V3[]): { x: V3; y: V3; z: V3; palm: number } {
  const toMiddle = sub(h[MIDDLE_MCP], h[WRIST]);
  const palm = norm(toMiddle);
  const y = normalize(toMiddle);
  const n = cross(sub(h[INDEX_MCP], h[WRIST]), sub(h[PINKY_MCP], h[WRIST]));
  const x = normalize(cross(y, n));
  return { x, y, z: cross(x, y), palm };
}

/** 15 bend angles in radians, three per finger, thumb first (0 = straight). */
export function flexion(h: V3[]): number[] {
  const out: number[] = [];
  for (const chain of FINGERS) {
    const pts = [WRIST, ...chain];
    for (let k = 1; k <= 3; k++) out.push(angle(sub(h[pts[k]], h[pts[k - 1]]), sub(h[pts[k + 1]], h[pts[k]])));
  }
  return out;
}

function handBlock(h: V3[], w: V3, vel: V3, f: Float32Array, o: number): void {
  const { x, y, z, palm } = handAxes(h);
  const inv = 1 / Math.max(palm, EPS);
  let k = o;
  for (let j = 1; j < COUNT; j++) { f[k++] = dot(x, h[j]) * inv; f[k++] = dot(y, h[j]) * inv; f[k++] = dot(z, h[j]) * inv; }
  for (const b of flexion(h)) f[k++] = b;
  for (let i = 0; i < 4; i++) f[k++] = angle(sub(h[FINGERS[i][1]], h[FINGERS[i][0]]), sub(h[FINGERS[i + 1][1]], h[FINGERS[i + 1][0]]));
  for (const [a, b] of TIP_PAIRS) f[k++] = norm(sub(h[a], h[b])) * inv;
  f.set(z, k); k += 3;
  f.set(y, k); k += 3;
  f.set(w, k); k += 3;
  f.set(vel, k); k += 3;
  f[k++] = 1;
  if (k !== o + HAND_DIM) throw new Error('hand block size mismatch');
}

/** Features for a resampled canonical segment: [frames × 197] float32, row-major. Absent hands are all zeros. */
export function sequenceFeatures(seg: Segment): Float32Array {
  const n = seg.length, dt = 1 / seg.fps;
  const f = new Float32Array(n * FEATURE_DIM);
  for (let slot = 0; slot < 2; slot++)
    for (let t = 0; t < n; t++) {
      if (!isPresent(seg, t, slot)) continue;
      const w = wristAt(t, slot);
      const pos: V3 = [seg.wristPos[w], seg.wristPos[w + 1], seg.wristPos[w + 2]];
      let vel: V3 = [0, 0, 0];
      if (t > 0 && isPresent(seg, t - 1, slot)) {
        const p = wristAt(t - 1, slot);
        vel = [(pos[0] - seg.wristPos[p]) / dt, (pos[1] - seg.wristPos[p + 1]) / dt, (pos[2] - seg.wristPos[p + 2]) / dt];
      }
      handBlock(hand(seg, t, slot), pos, vel, f, t * FEATURE_DIM + slot * HAND_DIM);
    }
  for (let t = 0; t < n; t++) {
    if (!isPresent(seg, t, 0) || !isPresent(seg, t, 1)) continue;
    for (let c = 0; c < 3; c++) f[t * FEATURE_DIM + 2 * HAND_DIM + c] = seg.wristPos[wristAt(t, 1) + c] - seg.wristPos[wristAt(t, 0) + c];
  }
  return f;
}

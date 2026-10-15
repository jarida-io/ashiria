// The avatar: a plain figure that signs from a clip's landmarks (`ashiria study-pose`), shared by the sign browser
// and the headset. It is a Group facing +z with the nose at its origin; put it where it should stand.
// - Arms: two-bone IK with fixed bone lengths, bending towards MediaPipe's elbow, so the forearm always meets the hand.
// - Hands: each finger bone keeps its median length for the clip (MediaPipe's lengths wobble frame to frame); nails
//   mark the backs of the fingers and the palm side is paler, so orientation reads at a glance.
// - Head: turns, nods and tilts with the face points, relative to the clip's own resting pose.
// - Teaching: the moving hand can glow, and the path of the current step can be drawn as a trail.
import * as THREE from 'three';

const SW = 0.36; // metres per shoulder width (the dictionary signer's median is 0.34, the headset default 0.38)
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const CHAIN = FINGERS.flatMap((c) => [[0, c[0]], [c[0], c[1]], [c[1], c[2]], [c[2], c[3]]]); // wrist outwards
const SEGMENTS = [[1, 2], [2, 3], [3, 4], ...FINGERS.slice(1).flatMap((c) => [[c[0], c[1]], [c[1], c[2]], [c[2], c[3]]])];
const PALM_RING = [0, 1, 5, 9, 13, 17];
const TIPS = [4, 8, 12, 16, 20];
const RADIUS = (() => {
  const r = new Array(21).fill(0.0095);
  r[0] = 0.016;
  [0.0125, 0.0108, 0.0097, 0.0088].forEach((v, k) => (r[1 + k] = v));
  FINGERS.slice(1).forEach((c, f) => {
    const scale = [1, 1.02, 0.97, 0.86][f];
    [0.0102, 0.009, 0.0081, 0.0074].forEach((v, k) => (r[c[k]] = v * scale));
  });
  return r;
})();
const PALM_HALF = 0.0105;
// Hand colours match the browser's landmark colours (signer's right cyan, left violet); the palm side is paler.
export const HAND_COLOURS = [{ back: 0x2fb7d1, palm: 0xc8f2fa }, { back: 0x9a83ee, palm: 0xe2d9ff }];
const MAX_TURN = THREE.MathUtils.degToRad(40);

type V = THREE.Vector3;
const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Vector3();
const v3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

/** One clip's body and hands in the body frame (see training/src/ashiria_train/study.py clip_pose). */
export interface AvatarPose {
  fps: number; frames: number; present: number[][]; hand_rel: number[][][][]; wrist_pos: number[][][];
  shoulders: number[][][]; elbows: number[][][]; hips: number[][][]; face?: number[][][];
  learn?: { hands: string[]; dominant?: string; steps: Step[] };
}
export interface Step { kind: 'start' | 'move' | 'end'; from: number; to: number; key: number; text: string }

function limb(r0: number, r1: number, material: THREE.Material, radial = 16) {
  return new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, radial, 1, true), material);
}
function ball(radius: number, material: THREE.Material, w = 18, h = 14) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius, w, h), material);
}
function place(mesh: THREE.Object3D, a: V, b: V) {
  const d = tmp.subVectors(b, a);
  const len = d.length();
  mesh.visible = len > 1e-5;
  if (!mesh.visible) return;
  mesh.position.copy(a).addScaledVector(d, 0.5);
  mesh.quaternion.setFromUnitVectors(UP, d.multiplyScalar(1 / len));
  mesh.scale.set(1, len, 1);
}
function solveElbow(S: V, W: V, l1: number, l2: number, pole: V, out: V) {
  const sw = v3().subVectors(W, S);
  const d = Math.min(Math.max(sw.length(), 1e-4), (l1 + l2) * 0.999);
  const u = sw.normalize();
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(l1 * l1 - a * a, 0));
  const v = v3().subVectors(pole, S);
  v.addScaledVector(u, -v.dot(u));
  if (v.lengthSq() < 1e-8) v.set(0, -1, 0).addScaledVector(u, -u.y);
  v.normalize();
  return out.copy(S).addScaledVector(u, a).addScaledVector(v, h);
}
function median(xs: number[]) {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}
function torsoGeometry() {
  const profile = [[0, 0.085], [0.17, 0.085], [0.31, 0.07], [0.43, 0.035], [0.5, -0.05], [0.51, -0.22], [0.48, -0.5],
    [0.42, -0.78], [0.44, -0.95], [0.4, -1.02], [0, -1.02]].reverse().map(([x, y]) => new THREE.Vector2(x, y));
  return new THREE.LatheGeometry(profile, 40);
}

export interface Figure {
  group: THREE.Group;
  setPose(pose: AvatarPose | null): void;
  setFrame(frame: number): void;
  /** Mirror: a true mirror image, as if the signer were your reflection. Default: facing you (signer's right on
   * your left). */
  setMirror(on: boolean): void;
  /** First person: hide the head, neck and chest (a camera in the head looks at the hands). */
  setFirstPerson(on: boolean): void;
  /** Make a hand glow (0 the signer's right, 1 left), or none. */
  setHighlight(slots: number[]): void;
  /** Draw the path of frames [from, to] of the given hands as a trail, or clear it. */
  setTrail(from: number, to: number, slots: number[]): void;
  /** A hand's middle knuckle in the group's coordinates at the current frame (for cameras to follow). */
  handCentre(slot: number): V | null;
  readonly pose: AvatarPose | null;
}

export function createFigure(): Figure {
  const skin = new THREE.MeshStandardMaterial({ color: 0x8d99a4, roughness: 0.62 });
  const shirt = new THREE.MeshStandardMaterial({ color: 0x4b5964, roughness: 0.85 });
  const feature = new THREE.MeshStandardMaterial({ color: 0x1a252e, roughness: 0.5 });
  const nail = new THREE.MeshStandardMaterial({ color: 0xf6f1ea, roughness: 0.3 });

  const group = new THREE.Group();
  group.name = 'Avatar';
  const torso = new THREE.Mesh(torsoGeometry(), shirt);
  torso.matrixAutoUpdate = false;
  const neck = limb(0.052, 0.046, skin);
  group.add(torso, neck);

  const headPivot = new THREE.Group();
  const head = new THREE.Group();
  head.position.set(0, 0.085, 0);
  const skull = ball(0.1, skin, 32, 24);
  skull.scale.set(0.86, 1.1, 0.96);
  const brows = [-1, 1].map((s) => {
    const b = limb(0.0045, 0.0045, feature, 8);
    b.position.set(s * 0.035, 0.045, 0.084);
    b.rotation.z = Math.PI / 2 - s * 0.06;
    b.scale.y = 0.03;
    return b;
  });
  const eyes = [-1, 1].map((s) => {
    const e = ball(0.0105, feature);
    e.position.set(s * 0.034, 0.02, 0.088);
    e.scale.set(1, 1.15, 0.6);
    return e;
  });
  const noseTip = ball(0.012, skin);
  noseTip.position.set(0, -0.008, 0.1);
  noseTip.scale.set(0.8, 1.2, 1);
  const mouth = limb(0.0035, 0.0035, feature, 8);
  mouth.position.set(0, -0.05, 0.088);
  mouth.rotation.z = Math.PI / 2;
  mouth.scale.y = 0.036;
  const ears = [-1, 1].map((s) => {
    const e = ball(0.022, skin);
    e.position.set(s * 0.087, 0.008, -0.004);
    e.scale.set(0.45, 1.25, 0.8);
    return e;
  });
  head.add(skull, ...brows, ...eyes, noseTip, mouth, ...ears);
  headPivot.add(head);
  group.add(headPivot);

  const arms = [0, 1].map((slot) => {
    const back = new THREE.MeshStandardMaterial({ color: HAND_COLOURS[slot].back, roughness: 0.5, emissive: HAND_COLOURS[slot].back, emissiveIntensity: 0 });
    const palm = new THREE.MeshStandardMaterial({ color: HAND_COLOURS[slot].palm, roughness: 0.55, side: THREE.DoubleSide });
    const backLayer = new THREE.MeshStandardMaterial({ color: HAND_COLOURS[slot].back, roughness: 0.5, side: THREE.DoubleSide });
    const arm = {
      shoulder: ball(0.05, shirt), upper: limb(0.047, 0.04, shirt), elbow: ball(0.039, skin), fore: limb(0.037, 0.029, skin),
      wrist: ball(0.029, skin), hand: new THREE.Group(), joints: [] as THREE.Mesh[], segments: [] as THREE.Mesh[],
      nails: [] as THREE.Mesh[], ring: [] as THREE.Mesh[], back,
      palmGeo: new THREE.BufferGeometry(), backGeo: new THREE.BufferGeometry(), lengths: [] as number[],
    };
    for (const g of [arm.palmGeo, arm.backGeo]) g.setAttribute('position', new THREE.BufferAttribute(new Float32Array((PALM_RING.length - 2) * 9), 3));
    arm.hand.add(new THREE.Mesh(arm.palmGeo, palm), new THREE.Mesh(arm.backGeo, backLayer));
    for (let j = 0; j < 21; j++) arm.joints.push(ball(j === 0 ? 0.018 : RADIUS[j], back, 14, 10));
    for (const [a, b] of SEGMENTS) arm.segments.push(limb(RADIUS[a], RADIUS[b], back, 12));
    for (let k = 0; k < PALM_RING.length; k++) arm.ring.push(limb(PALM_HALF, PALM_HALF, back, 12));
    for (let k = 0; k < TIPS.length; k++) arm.nails.push(ball(1, nail, 12, 8));
    arm.hand.add(...arm.joints, ...arm.segments, ...arm.ring, ...arm.nails);
    group.add(arm.shoulder, arm.upper, arm.elbow, arm.fore, arm.wrist, arm.hand);
    return arm;
  });

  const trail = new THREE.Group();
  group.add(trail);

  let pose: AvatarPose | null = null, mirror = false, firstPerson = false, frame = 0;
  let armLengths: { upper: number; fore: number }[] = [];
  let headRest: THREE.Matrix4 | null = null;
  let glow: number[] = [];
  const hand21 = () => [...Array(21)].map(() => v3());
  const P = { S: [v3(), v3()], E: [v3(), v3()], W: [v3(), v3()], H: [v3(), v3()], raw: [hand21(), hand21()],
    hand: [hand21(), hand21()], face: [...Array(7)].map(() => v3()) };

  const toScene = (p: number[], scale: number, out: V) => out.set((mirror ? 1 : -1) * p[0] * scale, p[1] * scale, p[2] * scale);
  const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const handPoint = (t: number, s: number, j: number, out: V) => {
    const w = pose!.wrist_pos[t][s], rel = pose!.hand_rel[t][s][j];
    return toScene([w[0] * SW + rel[0], w[1] * SW + rel[1], w[2] * SW + rel[2]], 1, out);
  };

  function headBasis(t: number, out: THREE.Matrix4) {
    const f = pose!.face![t].map((p, k) => toScene(p, SW, P.face[k]));
    const right = v3().subVectors(f[4], f[3]).normalize();
    const up = v3().addVectors(f[1], f[2]).sub(v3().addVectors(f[5], f[6]));
    up.addScaledVector(right, -up.dot(right)).normalize();
    return out.makeBasis(right, up, v3().crossVectors(right, up));
  }

  function measure() {
    const p = pose!;
    armLengths = [0, 1].map((s) => ({
      upper: median(p.shoulders.map((sh, t) => dist(sh[s], p.elbows[t][s]))) || 0.71,
      fore: median(p.elbows.map((el, t) => dist(el[s], p.wrist_pos[t][s]))) || 0.68,
    }));
    arms.forEach((arm, s) => {
      const seen = p.hand_rel.filter((_, t) => p.present[t][s] > 0);
      arm.lengths = CHAIN.map(([a, b]) => median(seen.map((h) => dist(h[s][a], h[s][b]))));
    });
    headRest = null;
    if (p.face) {
      const m = new THREE.Matrix4(), x = v3(), y = v3(), z = v3(), xs = v3(), ys = v3();
      for (let t = 0; t < p.frames; t++) { headBasis(t, m).extractBasis(x, y, z); xs.add(x); ys.add(y); }
      xs.normalize();
      ys.addScaledVector(xs, -ys.dot(xs)).normalize();
      headRest = new THREE.Matrix4().makeBasis(xs, ys, v3().crossVectors(xs, ys)).invert();
    }
  }

  const headTurn = new THREE.Quaternion(), mNow = new THREE.Matrix4(), still = new THREE.Quaternion();

  function update() {
    if (!pose) return;
    const t = Math.min(Math.max(frame, 0), pose.frames - 1);
    for (let s = 0; s < 2; s++) {
      toScene(pose.shoulders[t][s], SW, P.S[s]);
      toScene(pose.wrist_pos[t][s], SW, P.W[s]);
      toScene(pose.elbows[t][s], SW, P.E[s]);
      toScene(pose.hips[t][s], SW, P.H[s]);
      solveElbow(P.S[s], P.W[s], armLengths[s].upper * SW, armLengths[s].fore * SW, P.E[s], P.E[s]);
      for (let j = 0; j < 21; j++) handPoint(t, s, j, P.raw[s][j]);
      const q = P.hand[s], raw = P.raw[s], L = arms[s].lengths;
      q[0].copy(raw[0]);
      CHAIN.forEach(([a, b], k) => {
        const d = tmp.subVectors(raw[b], raw[a]);
        const len = d.length();
        q[b].copy(q[a]);
        if (len > 1e-6 && Number.isFinite(L[k])) q[b].addScaledVector(d, L[k] / len);
        else q[b].add(d);
      });
    }

    const shoulderMid = v3().addVectors(P.S[0], P.S[1]).multiplyScalar(0.5);
    const hipMid = v3().addVectors(P.H[0], P.H[1]).multiplyScalar(0.5);
    const up = v3().subVectors(shoulderMid, hipMid), height = up.length();
    const across = v3().subVectors(P.S[0], P.S[1]);
    const width = across.length();
    up.normalize();
    across.addScaledVector(up, -across.dot(up)).normalize();
    const fwd = v3().crossVectors(across, up).normalize();
    const top = v3().copy(shoulderMid).addScaledVector(up, 0.045).addScaledVector(fwd, -0.012);
    torso.matrix.makeBasis(across.clone().multiplyScalar(width * 1.04), up.clone().multiplyScalar(height + 0.045),
      fwd.clone().multiplyScalar(0.42)).setPosition(top);
    torso.matrixWorldNeedsUpdate = true;

    headPivot.position.copy(toScene([0, -0.012 / SW, -0.09 / SW], SW, v3())).add(v3(0, -0.085, 0));
    headPivot.quaternion.identity();
    if (headRest && pose.face) {
      headTurn.setFromRotationMatrix(headBasis(t, mNow).multiply(headRest));
      const angle = 2 * Math.acos(Math.min(1, Math.abs(headTurn.w)));
      if (angle > MAX_TURN) headTurn.slerp(still, 1 - MAX_TURN / angle);
      headPivot.quaternion.copy(headTurn);
    }
    headPivot.visible = torso.visible = !firstPerson;
    place(neck, top, headPivot.position);
    neck.visible = neck.visible && !firstPerson;

    for (let s = 0; s < 2; s++) {
      const arm = arms[s];
      arm.shoulder.position.copy(P.S[s]);
      arm.elbow.position.copy(P.E[s]);
      arm.wrist.position.copy(P.W[s]);
      place(arm.upper, P.S[s], P.E[s]);
      place(arm.fore, P.E[s], P.W[s]);
      const present = pose.present[t][s] > 0;
      arm.hand.visible = present;
      if (!present) continue;
      const q = P.hand[s];
      q.forEach((p, j) => arm.joints[j].position.copy(p));
      SEGMENTS.forEach(([a, b], k) => place(arm.segments[k], q[a], q[b]));
      const n = v3().crossVectors(v3().subVectors(q[5], q[0]), v3().subVectors(q[17], q[0])).normalize();
      if ((s === 0) === mirror) n.negate();
      for (const [geo, side] of [[arm.palmGeo, 1], [arm.backGeo, -1]] as const) {
        const pos = geo.attributes.position as THREE.BufferAttribute;
        for (let k = 0; k < PALM_RING.length - 2; k++) {
          [PALM_RING[0], PALM_RING[k + 1], PALM_RING[k + 2]].forEach((j, i) => {
            tmp.copy(q[j]).addScaledVector(n, side * PALM_HALF);
            pos.setXYZ(k * 3 + i, tmp.x, tmp.y, tmp.z);
          });
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();
        geo.computeBoundingSphere();
      }
      PALM_RING.forEach((j, k) => place(arm.ring[k], q[j], q[PALM_RING[(k + 1) % PALM_RING.length]]));
      TIPS.forEach((tip, k) => {
        const dir = v3().subVectors(q[tip], q[tip - 1]).normalize();
        const back = v3().copy(n).negate();
        back.addScaledVector(dir, -back.dot(dir)).normalize();
        const r = RADIUS[tip];
        const nl = arm.nails[k];
        nl.position.copy(q[tip]).addScaledVector(dir, -r * 0.45).addScaledVector(back, r * 0.8);
        nl.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(v3().crossVectors(dir, back), dir, back));
        nl.scale.set(r * 0.78, r * 1.05, r * 0.32);
      });
    }
  }

  function clearTrail() {
    trail.children.forEach((m) => (m as THREE.Mesh).geometry.dispose());
    trail.clear();
  }

  return {
    group,
    get pose() { return pose; },
    setPose(p) {
      pose = p;
      frame = 0;
      group.visible = !!p;
      clearTrail();
      if (p) { measure(); update(); }
    },
    setFrame(f) { if (f !== frame) { frame = f; update(); } },
    setMirror(on) { mirror = on; if (pose) { measure(); update(); } },
    setFirstPerson(on) { firstPerson = on; update(); },
    setHighlight(slots) {
      glow = slots;
      arms.forEach((arm, s) => { arm.back.emissiveIntensity = glow.includes(s) ? 0.55 : 0; });
    },
    setTrail(from, to, slots) {
      clearTrail();
      if (!pose || to - from < 3) return;
      for (const s of slots) {
        const pts: V[] = [];
        for (let t = Math.max(0, from); t <= Math.min(to, pose.frames - 1); t++) if (pose.present[t][s] > 0) pts.push(handPoint(t, s, 9, v3()));
        if (pts.length < 4) continue;
        const material = new THREE.MeshBasicMaterial({ color: HAND_COLOURS[s].back, transparent: true, opacity: 0.5, depthWrite: false });
        trail.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), pts.length * 3, 0.004, 6), material));
        // An arrowhead at the end shows which way the hand goes.
        const end = pts[pts.length - 1], before = pts[Math.max(0, pts.length - 4)];
        const cone = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.03, 12), material);
        cone.position.copy(end);
        cone.quaternion.setFromUnitVectors(UP, v3().subVectors(end, before).normalize());
        trail.add(cone);
      }
    },
    handCentre(slot) {
      if (!pose || !(pose.present[Math.min(frame, pose.frames - 1)][slot] > 0)) return null;
      return P.hand[slot][9].clone();
    },
  };
}

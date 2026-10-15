// The sign browser's avatar (dev only): a plain figure that signs a dictionary clip from its MediaPipe landmarks.
// Input is `ashiria study-pose`: the hands as Ashiria sees them (hand_rel in metres from the wrist, wrist_pos in
// shoulder widths from the nose, slot 0 the signer's right), and the shoulders, elbows, hips and face points in the
// same body frame (x towards the signer's right, y up, z the way the face points).
// - Arms: two-bone IK with fixed bone lengths, bending towards MediaPipe's elbow, so the forearm always meets the hand.
// - Hands: each finger bone keeps its median length for the clip (MediaPipe's lengths wobble frame to frame); nails
//   mark the backs of the fingers and the palm side is paler, so orientation reads at a glance.
// - Head: turns, nods and tilts with the face points, relative to the clip's own resting pose.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SW = 0.36; // metres per shoulder width (the dictionary signer's median is 0.34, the headset default 0.38)
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const CHAIN = FINGERS.flatMap((c) => [[0, c[0]], [c[0], c[1]], [c[1], c[2]], [c[2], c[3]]]); // wrist outwards
const SEGMENTS = [[1, 2], [2, 3], [3, 4], ...FINGERS.slice(1).flatMap((c) => [[c[0], c[1]], [c[1], c[2]], [c[2], c[3]]])];
const PALM_RING = [0, 1, 5, 9, 13, 17];
const TIPS = [4, 8, 12, 16, 20];
// Joint radii in metres: the thumb, then index to pinky (knuckle, middle, last joint, tip); fingers taper to the tip.
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
const PALM_HALF = 0.0105; // half the palm's thickness
// Hand colours match the browser's landmark colours (signer's right cyan, left violet); the palm side is paler.
const HAND = [{ back: 0x2fb7d1, palm: 0xc8f2fa }, { back: 0x9a83ee, palm: 0xe2d9ff }];
const VIEWS = {
  front: { pos: [0, -0.2, 1.75], target: [0, -0.33, 0.12], fov: 32 },
  side: { pos: [-1.55, -0.2, 0.55], target: [0, -0.33, 0.12], fov: 32 },
  // Close on the hands, following them: for reading handshapes and fingerspelling.
  hands: { pos: [0, -0.3, 0.85], target: [0, -0.35, 0.3], fov: 32, follow: true },
  // From the signer's own eyes, looking at their hands: the view a learner copies from. About a person's field of view.
  yours: { pos: [0, 0.0, -0.06], target: [0, -0.32, 0.6], fov: 75 },
};
const MAX_TURN = THREE.MathUtils.degToRad(40); // the head turns at most this far from its resting pose

const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Vector3();
const v3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

/** A tapered limb from radius `r0` at its base to `r1` at its end, placed with `place`. */
function limb(r0, r1, material, radial = 16) {
  return new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, radial, 1, true), material);
}

function ball(radius, material, w = 18, h = 14) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius, w, h), material);
}

function place(mesh, a, b) {
  const d = tmp.subVectors(b, a);
  const len = d.length();
  mesh.visible = len > 1e-5;
  if (!mesh.visible) return;
  mesh.position.copy(a).addScaledVector(d, 0.5);
  mesh.quaternion.setFromUnitVectors(UP, d.multiplyScalar(1 / len));
  mesh.scale.set(1, len, 1);
}

/** Elbow from shoulder S and wrist W with bone lengths l1, l2, bending towards `pole`. */
function solveElbow(S, W, l1, l2, pole, out) {
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

function median(xs) {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}

/** Torso: a lathed body (hips, waist, chest, rounded shoulders, neck base), one unit across at the shoulders, from
 * y = -1 at the hips up to y = 0 at the shoulder line. Listed from the bottom up, so the surface faces outwards.
 * Scaled and turned each frame. */
function torsoGeometry() {
  const profile = [[0, 0.085], [0.17, 0.085], [0.31, 0.07], [0.43, 0.035], [0.5, -0.05], [0.51, -0.22], [0.48, -0.5],
    [0.42, -0.78], [0.44, -0.95], [0.4, -1.02], [0, -1.02]].reverse().map(([x, y]) => new THREE.Vector2(x, y));
  return new THREE.LatheGeometry(profile, 40);
}

export function createAvatar() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.style.cssText = 'width:100%;height:100%;display:block;touch-action:none';

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1b22);
  scene.add(new THREE.HemisphereLight(0xf4f8ff, 0x1b2a33, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(0.7, 1.3, 1.7);
  const fill = new THREE.DirectionalLight(0xbfd8ff, 0.55);
  fill.position.set(-1.3, 0.2, 0.6);
  const rim = new THREE.DirectionalLight(0x9fd4ff, 0.9); // outlines the figure against the dark background
  rim.position.set(0, 0.8, -1.6);
  scene.add(key, fill, rim);

  const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.01, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.05;
  controls.maxDistance = 4;

  const skin = new THREE.MeshStandardMaterial({ color: 0x8d99a4, roughness: 0.62 });
  const shirt = new THREE.MeshStandardMaterial({ color: 0x4b5964, roughness: 0.85 });
  const feature = new THREE.MeshStandardMaterial({ color: 0x1a252e, roughness: 0.5 });
  const nail = new THREE.MeshStandardMaterial({ color: 0xf6f1ea, roughness: 0.3 });

  const figure = new THREE.Group();
  scene.add(figure);

  // Torso and neck.
  const torso = new THREE.Mesh(torsoGeometry(), shirt);
  torso.matrixAutoUpdate = false;
  const neck = limb(0.052, 0.046, skin);
  figure.add(torso, neck);

  // Head: a pivot at the base of the skull, so turns and nods swing the face. In the head's own frame x is the
  // figure's left as you face it, y up, z towards you; the features are symmetric, so mirroring needs nothing here.
  const headPivot = new THREE.Group();
  const head = new THREE.Group();
  head.position.set(0, 0.085, 0);
  const skull = ball(0.1, skin, 32, 24);
  skull.scale.set(0.86, 1.1, 0.96);
  const brows = [-1, 1].map((s) => {
    const b = limb(0.0045, 0.0045, feature, 8);
    b.position.set(s * 0.035, 0.045, 0.084);
    b.rotation.z = Math.PI / 2 - s * 0.06; // a touch lower at the outer ends: a calm, neutral face
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
  figure.add(headPivot);

  const arms = [0, 1].map((slot) => {
    const back = new THREE.MeshStandardMaterial({ color: HAND[slot].back, roughness: 0.5 });
    const palm = new THREE.MeshStandardMaterial({ color: HAND[slot].palm, roughness: 0.55, side: THREE.DoubleSide });
    const backLayer = new THREE.MeshStandardMaterial({ color: HAND[slot].back, roughness: 0.5, side: THREE.DoubleSide });
    const arm = {
      shoulder: ball(0.05, shirt), upper: limb(0.047, 0.04, shirt), elbow: ball(0.039, skin), fore: limb(0.037, 0.029, skin),
      wrist: ball(0.029, skin), hand: new THREE.Group(), joints: [], segments: [], nails: [], ring: [],
      palmGeo: new THREE.BufferGeometry(), backGeo: new THREE.BufferGeometry(), lengths: null,
    };
    const fan = PALM_RING.length - 2;
    for (const g of [arm.palmGeo, arm.backGeo]) g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(fan * 9), 3));
    arm.hand.add(new THREE.Mesh(arm.palmGeo, palm), new THREE.Mesh(arm.backGeo, backLayer));
    for (let j = 0; j < 21; j++) arm.joints.push(ball(j === 0 ? 0.018 : RADIUS[j], back, 14, 10));
    for (const [a, b] of SEGMENTS) arm.segments.push(limb(RADIUS[a], RADIUS[b], back, 12));
    for (let k = 0; k < PALM_RING.length; k++) arm.ring.push(limb(PALM_HALF, PALM_HALF, back, 12));
    for (let k = 0; k < TIPS.length; k++) arm.nails.push(ball(1, nail, 12, 8));
    arm.hand.add(...arm.joints, ...arm.segments, ...arm.ring, ...arm.nails);
    figure.add(arm.shoulder, arm.upper, arm.elbow, arm.fore, arm.wrist, arm.hand);
    return arm;
  });

  // The paths the hands take through the sign (a teaching aid, off by default).
  const paths = new THREE.Group();
  scene.add(paths);

  let pose = null, mirror = false, view = 'front', frame = 0, armLengths = null, showPath = false, headRest = null;
  const hand21 = () => [...Array(21)].map(() => v3());
  const P = { S: [v3(), v3()], E: [v3(), v3()], W: [v3(), v3()], H: [v3(), v3()], raw: [hand21(), hand21()],
    hand: [hand21(), hand21()], face: [...Array(7)].map(() => v3()) };

  /** A point in the body frame (units given by `scale`) as a scene position. Facing you: the signer's right is on
   * your left (x flipped, a proper rotation). Mirror: a true mirror image, as if the signer were your reflection.
   * "Your view" is never mirrored: it is the signer's own eyes. */
  const mirrored = () => mirror && view !== 'yours';
  const toScene = (p, scale, out) => out.set((mirrored() ? 1 : -1) * p[0] * scale, p[1] * scale, p[2] * scale);
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  /** Head orientation at frame t from the face points (nose, eyes, ears, mouth corners): ear to ear, mouth to eyes. */
  function headBasis(t, out) {
    const f = pose.face[t].map((p, k) => toScene(p, SW, P.face[k]));
    const right = v3().subVectors(f[4], f[3]).normalize();
    const up = v3().addVectors(f[1], f[2]).sub(v3().addVectors(f[5], f[6]));
    up.addScaledVector(right, -up.dot(right)).normalize();
    return out.makeBasis(right, up, v3().crossVectors(right, up));
  }

  function measure() {
    armLengths = [0, 1].map((s) => ({
      upper: median(pose.shoulders.map((sh, t) => dist(sh[s], pose.elbows[t][s]))) || 0.71,
      fore: median(pose.elbows.map((el, t) => dist(el[s], pose.wrist_pos[t][s]))) || 0.68,
    }));
    // Each finger bone's median length in this clip, over the frames that see the hand.
    arms.forEach((arm, s) => {
      const seen = pose.hand_rel.filter((_, t) => pose.present[t][s] > 0);
      arm.lengths = CHAIN.map(([a, b]) => median(seen.map((h) => dist(h[s][a], h[s][b]))));
    });
    // The head's resting orientation: the clip's average, so only its movement shows.
    headRest = null;
    if (pose.face) {
      const m = new THREE.Matrix4(), x = v3(), y = v3(), z = v3(), xs = v3(), ys = v3();
      for (let t = 0; t < pose.frames; t++) {
        headBasis(t, m).extractBasis(x, y, z);
        xs.add(x);
        ys.add(y);
      }
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
      toScene(pose.elbows[t][s], SW, P.E[s]); // MediaPipe's elbow: only the bend direction
      toScene(pose.hips[t][s], SW, P.H[s]);
      solveElbow(P.S[s], P.W[s], armLengths[s].upper * SW, armLengths[s].fore * SW, P.E[s], P.E[s]);
      const w = pose.wrist_pos[t][s], rel = pose.hand_rel[t][s];
      for (let j = 0; j < 21; j++) toScene([w[0] * SW + rel[j][0], w[1] * SW + rel[j][1], w[2] * SW + rel[j][2]], 1, P.raw[s][j]);
      // Rebuild the hand from the wrist outwards with each bone at its median length.
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

    // Torso: from just above the shoulder joints (MediaPipe's shoulders are the joints, below the top of the
    // shoulders) down to the hips, turned with the shoulders.
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

    // Head: the nose is the origin, level with the head's centre, which sits behind it; the pivot is below the
    // centre. Turns and nods are relative to the clip's resting pose, at most 40 degrees.
    headPivot.position.copy(toScene([0, -0.012 / SW, -0.09 / SW], SW, v3())).add(v3(0, -0.085, 0));
    headPivot.quaternion.identity();
    if (headRest) {
      headTurn.setFromRotationMatrix(headBasis(t, mNow).multiply(headRest));
      const angle = 2 * Math.acos(Math.min(1, Math.abs(headTurn.w)));
      if (angle > MAX_TURN) headTurn.slerp(still, 1 - MAX_TURN / angle);
      headPivot.quaternion.copy(headTurn);
    }
    headPivot.visible = view !== 'yours';
    torso.visible = headPivot.visible; // from the signer's own eyes the chest would fill the bottom of the view
    place(neck, top, headPivot.position);
    neck.visible = neck.visible && headPivot.visible; // in "your view" the camera sits where the head and neck are

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

      // Palm normal: index knuckle × little-finger knuckle points out of the palm for the right hand facing you;
      // mirroring or the left hand flips it.
      const n = v3().crossVectors(v3().subVectors(q[5], q[0]), v3().subVectors(q[17], q[0])).normalize();
      if ((s === 0) === mirrored()) n.negate();
      // Palm: two layers (palm side pale, back coloured) joined by a rounded rim.
      for (const [geo, side] of [[arm.palmGeo, 1], [arm.backGeo, -1]]) {
        const pos = geo.attributes.position;
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

      // Nails on the backs of the fingertips.
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

  // Close-up: keep the camera's offset and glide its target towards the signing hand: the higher of the visible
  // hands (a resting hand hangs low), or between them when both are up together.
  const aim = v3();
  function follow() {
    const t = Math.min(frame, pose.frames - 1);
    const seen = [0, 1].filter((s) => pose.present[t][s] > 0).map((s) => P.hand[s][9]);
    if (!seen.length) return;
    if (seen.length === 2 && Math.abs(seen[0].y - seen[1].y) < 0.12) aim.addVectors(seen[0], seen[1]).multiplyScalar(0.5);
    else aim.copy(seen.reduce((a, b) => (b.y > a.y ? b : a)));
    const step = v3().subVectors(aim, controls.target).multiplyScalar(0.08);
    controls.target.add(step);
    camera.position.add(step);
  }

  /** Thin tubes along each hand's path (its middle knuckle) through the whole clip, one per stretch it is seen. */
  function buildPaths() {
    paths.children.forEach((m) => m.geometry.dispose());
    paths.clear();
    paths.visible = showPath && !!pose;
    if (!paths.visible) return;
    for (let s = 0; s < 2; s++) {
      const material = new THREE.MeshBasicMaterial({ color: HAND[s].back, transparent: true, opacity: 0.45, depthWrite: false });
      let run = [];
      const flush = () => {
        if (run.length > 3) paths.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(run), run.length * 3, 0.0035, 6), material));
        run = [];
      };
      for (let t = 0; t < pose.frames; t++) {
        if (pose.present[t][s] > 0) {
          const w = pose.wrist_pos[t][s], rel = pose.hand_rel[t][s];
          run.push(toScene([w[0] * SW + rel[9][0], w[1] * SW + rel[9][1], w[2] * SW + rel[9][2]], 1, v3()));
        } else flush();
      }
      flush();
    }
  }

  /** After anything that changes how body-frame points map into the scene (pose, mirror, view). */
  function remap() {
    if (pose) { measure(); buildPaths(); }
    update();
  }

  function setView(name) {
    view = name;
    const v = VIEWS[name];
    camera.position.set(...v.pos);
    controls.target.set(...v.target);
    camera.fov = v.fov;
    camera.updateProjectionMatrix();
    controls.update();
    remap();
  }

  let host = null, raf = 0;
  const resize = () => {
    if (!host) return;
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  const loop = () => {
    if (pose && VIEWS[view].follow) follow(); // every frame, so the camera settles even while paused
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(loop);
  };

  setView('front');
  return {
    /** Put the avatar's canvas into `el` (one WebGL context, moved between signs). */
    mount(el) {
      if (host) observer.unobserve(host);
      host = el;
      el.appendChild(canvas);
      observer.observe(el);
      resize();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    },
    unmount() { cancelAnimationFrame(raf); if (host) observer.unobserve(host); canvas.remove(); host = null; },
    setPose(p) { pose = p; frame = 0; figure.visible = !!p; remap(); },
    setFrame(f) { if (f !== frame) { frame = f; update(); } },
    setMirror(m) { mirror = m; remap(); },
    setPath(on) { showPath = on; buildPaths(); },
    setView,
  };
}

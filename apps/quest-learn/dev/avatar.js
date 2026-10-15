// The sign browser's avatar (dev only): a plain figure that signs a dictionary clip from its MediaPipe landmarks.
// Input is `ashiria study-pose`: the hands as Ashiria sees them (hand_rel in metres from the wrist, wrist_pos in
// shoulder widths from the nose, slot 0 the signer's right) and the shoulders, elbows and hips in the same body frame
// (x towards the signer's right, y up, z the way the face points). Elbows come from two-bone IK with fixed bone
// lengths, using MediaPipe's elbow as the bend direction, so the forearm always meets the hand.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SW = 0.36; // metres per shoulder width (the dictionary signer's median is 0.34, the headset default 0.38)
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const HAND_BONES = FINGERS.flatMap((c) => [[0, c[0]], [c[0], c[1]], [c[1], c[2]], [c[2], c[3]]]).concat([[5, 9], [9, 13], [13, 17], [0, 17]]);
const PALM_FAN = [[0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17]];
// Hand colours match the browser's landmark colours (signer's right cyan, left violet); the palm side is paler, so
// palm orientation reads at a glance.
const HAND = [{ back: 0x2fb7d1, palm: 0xc8f2fa }, { back: 0x9a83ee, palm: 0xe2d9ff }];
const VIEWS = {
  front: { pos: [0, -0.2, 1.75], target: [0, -0.33, 0.12], fov: 32 },
  side: { pos: [-1.55, -0.2, 0.55], target: [0, -0.33, 0.12], fov: 32 },
  // Close on the hands, following them: for reading handshapes and fingerspelling.
  hands: { pos: [0, -0.3, 0.95], target: [0, -0.35, 0.3], fov: 32, follow: true },
  // From the signer's own eyes, looking at their hands: the view a learner copies from. About a person's field of view.
  yours: { pos: [0, 0.0, -0.06], target: [0, -0.45, 0.55], fov: 75 },
};

const UP = new THREE.Vector3(0, 1, 0);
const tmp = new THREE.Vector3();

function bone(radius, material) {
  return new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 12), material);
}

function ball(radius, material) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), material);
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
  const sw = new THREE.Vector3().subVectors(W, S);
  const d = Math.min(Math.max(sw.length(), 1e-4), (l1 + l2) * 0.999);
  const u = sw.normalize();
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(l1 * l1 - a * a, 0));
  const v = new THREE.Vector3().subVectors(pole, S);
  v.addScaledVector(u, -v.dot(u));
  if (v.lengthSq() < 1e-8) v.set(0, -1, 0).addScaledVector(u, -u.y);
  v.normalize();
  return out.copy(S).addScaledVector(u, a).addScaledVector(v, h);
}

function median(xs) {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}

export function createAvatar() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.style.cssText = 'width:100%;height:100%;display:block;touch-action:none';

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1b22);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x1b2a33, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(0.6, 1.2, 1.6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xbfd8ff, 0.6);
  fill.position.set(-1.2, 0.3, 0.4);
  scene.add(fill);

  const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.01, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.05;
  controls.maxDistance = 4;

  const skin = new THREE.MeshStandardMaterial({ color: 0x7d8a95, roughness: 0.7 });
  const shirt = new THREE.MeshStandardMaterial({ color: 0x52606b, roughness: 0.85 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x14202a, roughness: 0.4 });

  const figure = new THREE.Group();
  scene.add(figure);
  const head = ball(0.1, skin);
  head.scale.set(0.9, 1.13, 1);
  const eyes = [ball(0.011, dark), ball(0.011, dark)];
  const neck = bone(0.045, skin);
  // Torso: a tapered, flattened cylinder (shoulders wider than the waist), unit size, scaled each frame.
  const torsoGeo = new THREE.CylinderGeometry(0.5, 0.4, 1, 32, 1);
  torsoGeo.translate(0, -0.5, 0); // origin at the top, so it hangs from the shoulder line
  const torso = new THREE.Mesh(torsoGeo, shirt);
  figure.add(head, ...eyes, neck, torso);

  const arms = [0, 1].map((slot) => {
    const handMats = { back: new THREE.MeshStandardMaterial({ color: HAND[slot].back, roughness: 0.55 }),
      palm: new THREE.MeshStandardMaterial({ color: HAND[slot].palm, roughness: 0.6 }) };
    const arm = {
      shoulder: ball(0.052, shirt), upper: bone(0.043, shirt), elbow: ball(0.04, shirt), fore: bone(0.034, skin),
      wrist: ball(0.03, handMats.back), joints: [], bones: [], mats: handMats,
      palmGeo: new THREE.BufferGeometry(), hand: new THREE.Group(),
    };
    arm.palmGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PALM_FAN.length * 9), 3));
    arm.palmFront = new THREE.Mesh(arm.palmGeo, handMats.palm.clone());
    arm.palmBack = new THREE.Mesh(arm.palmGeo, handMats.back.clone());
    arm.palmFront.material.side = THREE.FrontSide;
    arm.palmBack.material.side = THREE.BackSide;
    for (let j = 0; j < 21; j++) arm.joints.push(ball(j === 0 ? 0.014 : j % 4 === 0 ? 0.0085 : 0.0095, handMats.back));
    for (const [a] of HAND_BONES) arm.bones.push(bone(a === 0 || FINGERS[0].includes(a) ? 0.0095 : 0.0082, handMats.back));
    arm.hand.add(arm.palmFront, arm.palmBack, ...arm.joints, ...arm.bones);
    figure.add(arm.shoulder, arm.upper, arm.elbow, arm.fore, arm.wrist, arm.hand);
    return arm;
  });

  let pose = null, mirror = false, view = 'front', frame = 0, lengths = null;
  const v3 = () => new THREE.Vector3();
  const P = { S: [v3(), v3()], E: [v3(), v3()], W: [v3(), v3()], H: [v3(), v3()], hand: [[...Array(21)].map(v3), [...Array(21)].map(v3)] };

  /** A point in the body frame (units given by `scale`) as a scene position. Facing you: the signer's right is on
   * your left (x flipped, a proper rotation). Mirror: a true mirror image, as if the signer were your reflection.
   * "Your view" is never mirrored: it is the signer's own eyes. */
  const mirrored = () => mirror && view !== 'yours';
  const toScene = (p, scale, out) => out.set((mirrored() ? 1 : -1) * p[0] * scale, p[1] * scale, p[2] * scale);

  function measure() {
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    lengths = [0, 1].map((s) => ({
      upper: median(pose.shoulders.map((sh, t) => dist(sh[s], pose.elbows[t][s]))) || 0.71,
      fore: median(pose.elbows.map((el, t) => dist(el[s], pose.wrist_pos[t][s]))) || 0.68,
    }));
  }

  function update() {
    if (!pose) return;
    const t = Math.min(Math.max(frame, 0), pose.frames - 1);
    for (let s = 0; s < 2; s++) {
      toScene(pose.shoulders[t][s], SW, P.S[s]);
      toScene(pose.wrist_pos[t][s], SW, P.W[s]);
      toScene(pose.elbows[t][s], SW, P.E[s]); // MediaPipe's elbow: only the bend direction
      toScene(pose.hips[t][s], SW, P.H[s]);
      const L = lengths[s];
      solveElbow(P.S[s], P.W[s], L.upper * SW, L.fore * SW, P.E[s], P.E[s]);
      const w = pose.wrist_pos[t][s], rel = pose.hand_rel[t][s];
      for (let j = 0; j < 21; j++) toScene([w[0] * SW + rel[j][0], w[1] * SW + rel[j][1], w[2] * SW + rel[j][2]], 1, P.hand[s][j]);
    }
    const shoulderMid = v3().addVectors(P.S[0], P.S[1]).multiplyScalar(0.5);
    const hipMid = v3().addVectors(P.H[0], P.H[1]).multiplyScalar(0.5);
    // Torso: hangs from a little above the shoulder joints (MediaPipe's shoulders are the joints, below the top of
    // the shoulders) to the hips, turned with the shoulders.
    const up = v3().subVectors(shoulderMid, hipMid), height = up.length();
    const across = v3().subVectors(P.S[0], P.S[1]);
    const width = across.length();
    up.normalize();
    across.addScaledVector(up, -across.dot(up)).normalize();
    const fwd = v3().crossVectors(across, up).normalize();
    const top = v3().copy(shoulderMid).addScaledVector(up, 0.05).addScaledVector(fwd, -0.01);
    torso.matrixAutoUpdate = false;
    torso.matrix.makeBasis(across.clone().multiplyScalar(width * 1.05), up.clone().multiplyScalar(height + 0.05),
      fwd.clone().multiplyScalar(0.21)).setPosition(top);
    // Head: the nose is the origin, about level with the head's centre, which sits behind it. The figure doesn't
    // turn its head (the body frame has none), so the eyes only show which way it faces.
    const headPos = toScene([0, 0.005 / SW, -0.09 / SW], SW, v3());
    head.position.copy(headPos);
    head.visible = view !== 'yours';
    eyes.forEach((e, i) => { toScene([(i ? -1 : 1) * 0.034 / SW, 0.03 / SW, -0.012 / SW], SW, e.position); e.visible = head.visible; });
    place(neck, top, v3().copy(headPos).add(v3(0, -0.06, 0)));
    neck.visible = neck.visible && head.visible; // in "your view" the camera sits where the head and neck are

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
      const pts = P.hand[s];
      pts.forEach((p, j) => arm.joints[j].position.copy(p));
      HAND_BONES.forEach(([a, b], k) => place(arm.bones[k], pts[a], pts[b]));
      const pos = arm.palmGeo.attributes.position;
      PALM_FAN.forEach((tri, k) => tri.forEach((j, i) => pos.setXYZ(k * 3 + i, pts[j].x, pts[j].y, pts[j].z)));
      pos.needsUpdate = true;
      arm.palmGeo.computeVertexNormals();
      arm.palmGeo.computeBoundingSphere();
      // The fan's front face is the palm for the right hand seen facing you; mirroring or the left hand flips it.
      const palmFront = (s === 0) !== mirrored();
      arm.palmFront.material.color.setHex(palmFront ? HAND[s].palm : HAND[s].back);
      arm.palmBack.material.color.setHex(palmFront ? HAND[s].back : HAND[s].palm);
    }
    if (VIEWS[view].follow) follow();
  }

  // Close-up: keep the camera's offset and glide its target towards the middle of the visible hands.
  const aim = v3();
  function follow() {
    let n = 0;
    aim.set(0, 0, 0);
    for (let s = 0; s < 2; s++) if (pose.present[Math.min(frame, pose.frames - 1)][s] > 0) { aim.add(P.hand[s][9]); n++; }
    if (!n) return;
    aim.multiplyScalar(1 / n);
    const step = v3().subVectors(aim, controls.target).multiplyScalar(0.15);
    controls.target.add(step);
    camera.position.add(step);
  }

  function setView(name) {
    view = name;
    const v = VIEWS[name];
    camera.position.set(...v.pos);
    controls.target.set(...v.target);
    camera.fov = v.fov;
    camera.updateProjectionMatrix();
    controls.update();
    update();
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
  const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };

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
    setPose(p) { pose = p; frame = 0; if (p) measure(); figure.visible = !!p; update(); },
    setFrame(f) { if (f !== frame) { frame = f; update(); } },
    setMirror(m) { mirror = m; update(); },
    setView,
  };
}

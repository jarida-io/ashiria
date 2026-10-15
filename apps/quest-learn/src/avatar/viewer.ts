// The avatar on a web page (the sign browser): renderer, camera views and, in Learn mode, a camera that follows the
// moving hand and moves in close while a step is held, with the step's caption, glow and path.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { createFigure, type AvatarPose } from './figure.js';
import { LearnPlayer, type LearnState } from './learn.js';

type ViewName = 'front' | 'side' | 'hands' | 'yours';
const VIEWS: Record<ViewName, { pos: [number, number, number]; target: [number, number, number]; fov: number; follow?: boolean }> = {
  front: { pos: [0, -0.2, 1.75], target: [0, -0.33, 0.12], fov: 32 },
  side: { pos: [-1.55, -0.2, 0.55], target: [0, -0.33, 0.12], fov: 32 },
  hands: { pos: [0, -0.3, 0.85], target: [0, -0.35, 0.3], fov: 32, follow: true },
  yours: { pos: [0, 0.0, -0.06], target: [0, -0.32, 0.6], fov: 75 },
};

export interface AvatarView {
  mount(el: HTMLElement): void;
  unmount(): void;
  setPose(pose: AvatarPose | null): void;
  /** Show this frame of the clip (ignored while learning, which keeps its own time). */
  setFrame(frame: number): void;
  setMirror(on: boolean): void;
  setView(name: ViewName): void;
  /** Draw each hand's path through the whole clip. */
  setPath(on: boolean): void;
  /** Learn mode: steps one at a time, slowly, the camera following the moving hand. `onStep` hears each change. */
  setLearning(on: boolean, onStep?: (s: LearnState, steps: string[]) => void): void;
  readonly learning: boolean;
}

export function createAvatar(): AvatarView {
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
  const rim = new THREE.DirectionalLight(0x9fd4ff, 0.9);
  rim.position.set(0, 0.8, -1.6);
  scene.add(key, fill, rim);

  const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.01, 20);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.05;
  controls.maxDistance = 4;

  const figure = createFigure();
  scene.add(figure.group);

  let view: ViewName = 'front', mirror = false, showPath = false, learning = false;
  let learn: LearnPlayer | null = null, learnTime = 0, lastStep = -2, lastPhase = '';
  let onStep: ((s: LearnState, steps: string[]) => void) | undefined;

  const applyMirror = () => figure.setMirror(mirror && view !== 'yours');
  const refreshPath = () => {
    const p = figure.pose;
    if (!p || learning) return;
    figure.setTrail(0, showPath ? p.frames - 1 : 0, showPath ? [0, 1] : []);
  };

  // Follow a point: keep the camera's offset from its target and glide both towards it; `distance` optionally pulls
  // the camera in or out along its line of sight.
  const follow = (aim: THREE.Vector3, rate: number, distance?: number) => {
    const step = new THREE.Vector3().subVectors(aim, controls.target).multiplyScalar(rate);
    controls.target.add(step);
    camera.position.add(step);
    if (distance) {
      const off = new THREE.Vector3().subVectors(camera.position, controls.target);
      off.setLength(off.length() + (distance - off.length()) * rate);
      camera.position.copy(controls.target).add(off);
    }
  };
  const signingHandCentre = (hands: number[]): THREE.Vector3 | null => {
    const pts = hands.map((s) => figure.handCentre(s)).filter((p): p is THREE.Vector3 => !!p);
    if (!pts.length) return null;
    return pts.reduce((a, b) => a.add(b), new THREE.Vector3()).multiplyScalar(1 / pts.length);
  };

  let last = performance.now();
  function tick(now: number) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (learning && learn) {
      learnTime += dt;
      const s = learn.at(learnTime);
      figure.setFrame(s.frame);
      figure.setHighlight(s.phase === 'move' ? s.hands : []);
      if (s.step !== lastStep || s.phase !== lastPhase) {
        if (s.step !== lastStep) figure.setTrail(s.from, s.to, s.step >= 0 ? s.hands : []);
        lastStep = s.step;
        lastPhase = s.phase;
        onStep?.(s, learn.steps);
      }
      // Watch the signing hand: from a little way off while it moves, close in while a step is held.
      const aim = signingHandCentre(s.hands);
      if (aim && view !== 'yours') follow(aim, 0.06, s.phase === 'hold' ? 0.6 : s.phase === 'move' ? 0.95 : 1.4);
    } else if (VIEWS[view].follow) {
      const p = figure.pose;
      const seen = [0, 1].map((s) => figure.handCentre(s)).filter((x): x is THREE.Vector3 => !!x);
      if (p && seen.length) follow(seen.length === 2 && Math.abs(seen[0].y - seen[1].y) < 0.12 ? seen[0].add(seen[1]).multiplyScalar(0.5) : seen.reduce((a, b) => (b.y > a.y ? b : a)), 0.08);
    }
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }

  let host: HTMLElement | null = null, raf = 0;
  const resize = () => {
    if (!host) return;
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);

  const api: AvatarView = {
    mount(el) {
      if (host) observer.unobserve(host);
      host = el;
      el.appendChild(canvas);
      observer.observe(el);
      resize();
      cancelAnimationFrame(raf);
      last = performance.now();
      raf = requestAnimationFrame(tick);
    },
    unmount() { cancelAnimationFrame(raf); if (host) observer.unobserve(host); canvas.remove(); host = null; },
    setPose(p) {
      figure.setPose(p);
      applyMirror();
      learn = p ? new LearnPlayer(p) : null;
      learnTime = 0;
      lastStep = -2;
      refreshPath();
    },
    setFrame(f) { if (!learning) figure.setFrame(f); },
    setMirror(on) { mirror = on; applyMirror(); refreshPath(); },
    setView(name) {
      view = name;
      const v = VIEWS[name];
      camera.position.set(...v.pos);
      controls.target.set(...v.target);
      camera.fov = v.fov;
      camera.updateProjectionMatrix();
      controls.update();
      figure.setFirstPerson(name === 'yours');
      applyMirror();
    },
    setPath(on) { showPath = on; refreshPath(); },
    setLearning(on, cb) {
      learning = on;
      onStep = cb;
      learnTime = 0;
      lastStep = -2;
      figure.setHighlight([]);
      if (!on) refreshPath();
    },
    get learning() { return learning; },
  };
  api.setView('front');
  return api;
}

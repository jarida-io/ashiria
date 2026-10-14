/**
 * Shadow hands: the target sign's reference take, played back where the user's own hands should be.
 * Canonical joints (head-yaw frame, nose origin, shoulder widths) are mapped back to world space with the
 * user's current head frame, the inverse of the canonicalisation in FrameBuilder.
 */
import { DOMINANT, FINGERS, NOSE_OFFSET, type Take } from '@ashiria/engine';
import {
  Color, createSystem, CylinderGeometry, InstancedMesh, Matrix4, MeshBasicMaterial, Quaternion, SphereGeometry, Vector3,
} from '@iwsdk/core';
import { app } from './app-state.js';
import { HandSamplerSystem } from './hand-sampler.js';

const JOINTS = 21;
const BONES: [number, number][] = [
  ...FINGERS.flatMap((c) => [[0, c[0]], [c[0], c[1]], [c[1], c[2]], [c[2], c[3]]] as [number, number][]),
  [5, 9], [9, 13], [13, 17],
];
const HOLD_AT_END_S = 0.8;
const COLORS = { dominant: new Color('#4fd1e8'), other: new Color('#b49cff'), wrong: new Color('#ff9f43') };
/** Bones belonging to each finger (thumb..pinky), for highlighting. */
const FINGER_BONES = FINGERS.map((_, f) => [f * 4, f * 4 + 1, f * 4 + 2, f * 4 + 3]);

export class ShadowHandsSystem extends createSystem({}) {
  private joints!: InstancedMesh;
  private bones!: InstancedMesh;
  private m = new Matrix4();
  private q = new Quaternion();
  private v = new Vector3();
  private a = new Vector3();
  private b = new Vector3();
  private up = new Vector3(0, 1, 0);
  private scaleV = new Vector3();
  private pts = new Float64Array(2 * JOINTS * 3);
  private present = [false, false];
  private clock = 0;
  /** Fingers to highlight (thumb..pinky) after a Verify result, until this time (seconds). */
  highlight: boolean[] = [false, false, false, false, false];
  highlightUntil = 0;

  init(): void {
    const material = new MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false });
    this.joints = new InstancedMesh(new SphereGeometry(0.0055, 10, 6), material, 2 * JOINTS);
    this.bones = new InstancedMesh(new CylinderGeometry(0.0042, 0.0042, 1, 8, 1), material, 2 * BONES.length);
    this.joints.frustumCulled = this.bones.frustumCulled = false;
    this.joints.name = 'ShadowJoints';
    this.bones.name = 'ShadowBones';
    for (let i = 0; i < 2 * JOINTS; i++) this.joints.setColorAt(i, i < JOINTS ? COLORS.dominant : COLORS.other);
    for (let i = 0; i < 2 * BONES.length; i++) this.bones.setColorAt(i, i < BONES.length ? COLORS.dominant : COLORS.other);
    this.pts.fill(0);
    this.world.createTransformEntity(this.joints);
    this.world.createTransformEntity(this.bones);
  }

  private take(): Take | null {
    const info = app.current;
    if (!info || app.mode === 'free') return null;
    const sign = app.pack?.signs.get(info.id);
    return sign ? sign.takes[sign.shadowTake] ?? sign.takes[0] : null;
  }

  update(delta: number, time: number): void {
    const take = this.take();
    this.joints.visible = this.bones.visible = !!take;
    if (!take) return;
    const seg = take.segment;
    const period = (seg.length - 1) / seg.fps + HOLD_AT_END_S;
    this.clock = (this.clock + delta) % period;
    const t = Math.min(seg.length - 1, Math.floor(this.clock * seg.fps));
    this.toWorld(take, t);
    this.draw(time);
  }

  /** Canonical frame t of the take → world-space joints (three.js, right-handed) in this.pts. */
  private toWorld(take: Take, t: number): void {
    const sampler = this.sampler();
    const s = app.settings;
    const sign = s.leftHanded ? -1 : 1;
    let x: number[], y: number[], z: number[], head: number[];
    if (sampler?.axes) {
      [x, y, z] = sampler.axes; head = Array.from(sampler.headPosition);
    } else {  // outside XR: in front of the camera (left-handed: flip z)
      this.camera.getWorldPosition(this.v);
      head = [this.v.x, this.v.y, -this.v.z];
      x = [1, 0, 0]; y = [0, 1, 0]; z = [0, 0, 1];
      this.camera.getWorldDirection(this.a);
      const flat = Math.hypot(this.a.x, this.a.z);
      if (flat > 1e-3) { z = [this.a.x / flat, 0, -this.a.z / flat]; x = [z[2], 0, -z[0]]; }
    }
    const origin = [0, 1, 2].map((c) => head[c] + x[c] * NOSE_OFFSET[0] + y[c] * NOSE_OFFSET[1] + z[c] * NOSE_OFFSET[2]);
    const seg = take.segment;
    const twoHanded = app.current?.tags.hands === 'two';
    for (let slot = 0; slot < 2; slot++) {
      // One-handed signs: the other hand is just resting, so don't draw it.
      this.present[slot] = seg.present[t * 2 + slot] > 0.5 && (slot === DOMINANT || twoHanded);
      const wo = (t * 2 + slot) * 3;
      const wx = sign * seg.wristPos[wo] * s.shoulderWidth, wy = seg.wristPos[wo + 1] * s.shoulderWidth, wz = seg.wristPos[wo + 2] * s.shoulderWidth;
      for (let j = 0; j < JOINTS; j++) {
        const jo = ((t * 2 + slot) * JOINTS + j) * 3;
        const rx = wx + sign * seg.handRel[jo], ry = wy + seg.handRel[jo + 1], rz = wz + seg.handRel[jo + 2];
        const k = (slot * JOINTS + j) * 3;
        for (let c = 0; c < 3; c++) this.pts[k + c] = origin[c] + x[c] * rx + y[c] * ry + z[c] * rz;
        this.pts[k + 2] = -this.pts[k + 2];  // left-handed → three.js
      }
    }
  }

  private sampler(): HandSamplerSystem | undefined {
    return this.world.getSystem(HandSamplerSystem) as HandSamplerSystem | undefined;
  }

  private draw(time: number): void {
    const hl = time < this.highlightUntil;
    for (let slot = 0; slot < 2; slot++) {
      const scale = this.present[slot] ? 1 : 0;
      for (let j = 0; j < JOINTS; j++) {
        const k = (slot * JOINTS + j) * 3;
        this.m.makeScale(scale, scale, scale).setPosition(this.pts[k], this.pts[k + 1], this.pts[k + 2]);
        this.joints.setMatrixAt(slot * JOINTS + j, this.m);
      }
      BONES.forEach(([i, j], bi) => {
        const ka = (slot * JOINTS + i) * 3, kb = (slot * JOINTS + j) * 3;
        this.a.set(this.pts[ka], this.pts[ka + 1], this.pts[ka + 2]);
        this.b.set(this.pts[kb], this.pts[kb + 1], this.pts[kb + 2]);
        this.v.subVectors(this.b, this.a);
        const len = this.v.length();
        this.q.setFromUnitVectors(this.up, len > 1e-6 ? this.v.multiplyScalar(1 / len) : this.up);
        this.scaleV.set(scale, scale * len, scale);
        this.m.compose(this.a.add(this.b).multiplyScalar(0.5), this.q, this.scaleV);
        this.bones.setMatrixAt(slot * BONES.length + bi, this.m);
        if (slot === DOMINANT) {
          const finger = FINGER_BONES.findIndex((bs) => bs.includes(bi));
          this.bones.setColorAt(bi, hl && finger >= 0 && this.highlight[finger] ? COLORS.wrong : COLORS.dominant);
        }
      });
    }
    this.joints.instanceMatrix.needsUpdate = this.bones.instanceMatrix.needsUpdate = true;
    if (this.bones.instanceColor) this.bones.instanceColor.needsUpdate = true;
  }
}

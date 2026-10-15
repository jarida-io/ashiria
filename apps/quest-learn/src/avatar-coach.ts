/**
 * The avatar coach: a figure standing in front of the learner, facing them, that signs the current sign one step at
 * a time (start, each movement, end), slowly, pausing on each key moment; the moving hand glows and the step's path
 * is drawn. The step's caption is shown on the lesson panel (LessonSystem reads `caption`). Hidden while signing
 * from memory (recall) and in Free Sign, like the shadow hands.
 *
 * Development only for now: each sign's avatar (pose and steps) comes from the dev server (/dev/avatar/<sign>).
 * Shipping it means adding the avatar data to the content pack.
 */
import { createSystem, DirectionalLight, HemisphereLight, Vector3, VisibilityState } from '@iwsdk/core';
import { app } from './app-state.js';
import { createFigure, type AvatarPose } from './avatar/figure.js';
import { LearnPlayer } from './avatar/learn.js';

const DISTANCE_M = 1.3; // in front of the learner
const ASIDE_M = 0.3;    // to the right, clear of the lesson panel (which floats low on the left)

export class AvatarCoachSystem extends createSystem({}) {
  private figure = createFigure();
  private learner: LearnPlayer | null = null;
  private signId: string | null = null;
  private loading = false;
  private clock = 0;
  private lastStep = -2;
  private placed = false;
  private placedAt = new Vector3();
  private immersiveFor = 0;
  /** The current step, for the lesson panel ('' when there is none). */
  caption = '';

  init(): void {
    const g = this.figure.group;
    // The scene has no lights of its own (the panel and shadow hands don't need them); the figure brings soft ones.
    const sky = new HemisphereLight(0xf4f8ff, 0x3a4650, 1.4);
    const key = new DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, 1.2, 1.5);
    g.add(sky, key);
    g.visible = false;
    this.world.createTransformEntity(g);
    // Stand in front of the learner each time a headset session starts (before it, the camera is a desktop view).
    this.world.visibilityState.subscribe((v) => { if (v === VisibilityState.NonImmersive) { this.placed = false; this.immersiveFor = 0; } });
  }

  private async load(id: string): Promise<void> {
    this.loading = true;
    this.signId = id;
    try {
      const res = await fetch(`/dev/avatar/${encodeURIComponent(id)}`);
      const pose: AvatarPose | null = res.ok ? await res.json() : null;
      if (this.signId !== id) return;
      this.figure.setPose(pose);
      this.learner = pose ? new LearnPlayer(pose) : null;
      this.clock = 0;
      this.lastStep = -2;
    } catch {
      this.learner = null;
    } finally {
      this.loading = false;
    }
  }

  /** Stand in front of the learner's head, facing them (the figure faces +z, its nose at the group's origin). */
  private place(): void {
    const cam = this.world.camera;
    const head = cam.getWorldPosition(new Vector3());
    const fwd = cam.getWorldDirection(new Vector3());
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    const right = new Vector3(-fwd.z, 0, fwd.x);
    const g = this.figure.group;
    g.position.copy(head).addScaledVector(fwd, DISTANCE_M).addScaledVector(right, ASIDE_M);
    g.position.y = head.y - 0.05;
    g.rotation.set(0, Math.atan2(-fwd.x, -fwd.z), 0);
    this.placedAt.copy(head);
    this.placed = true;
  }

  update(delta: number): void {
    const info = app.current;
    if (import.meta.env.DEV && info && info.id !== this.signId && !this.loading) void this.load(info.id);
    const show = !!info && !!this.learner && app.mode !== 'free' && !(app.mode === 'practice' && app.phase === 'recall');
    const g = this.figure.group;
    g.visible = show;
    if (!show) { this.caption = ''; return; }
    // Place once the headset's pose has arrived (half a second into the session), and again if the learner moves
    // well away from where it was placed (stands up, walks).
    if (this.world.visibilityState.value !== VisibilityState.NonImmersive) this.immersiveFor += delta;
    const head = this.world.camera.getWorldPosition(new Vector3());
    if (this.placed && head.distanceTo(this.placedAt) > 0.6) this.placed = false;
    if (!this.placed && this.immersiveFor > 0.5) this.place();
    this.clock += delta;
    const s = this.learner!.at(this.clock);
    this.figure.setFrame(s.frame);
    this.figure.setHighlight(s.phase === 'move' ? s.hands : []);
    if (s.step !== this.lastStep) {
      this.figure.setTrail(s.from, s.to, s.step >= 0 ? s.hands : []);
      this.lastStep = s.step;
    }
    this.caption = s.step >= 0 && s.text ? `Step ${s.step + 1} of ${this.learner!.steps.length}: ${s.text}` : '';
  }

  /** Stand in front of the learner again (after they turn or move). */
  recentre(): void { this.placed = false; }
}

/**
 * The lesson loop: practise the target sign (Verify), Free Sign (Recognise among learned signs), or Studio
 * (development: save the team's takes). Drives the lesson panel and the shadow-hand highlights.
 */
import { hintText, lookalikesOf, prepare, recognise, restHeight, verify, type Attempt } from '@ashiria/engine';
import { createSystem, FollowBehavior, Follower, RayInteractable, VisibilityState, type UIKitMLAsset } from '@iwsdk/core';
import { app, STUDIO_ENABLED } from './app-state.js';
import { currentStreak, masteryText, unitSummary } from './flow.js';
import { HandSamplerSystem } from './hand-sampler.js';
import { ShadowHandsSystem } from './shadow-hands.js';
import { saveTake } from './studio.js';

const UI_GUARD_MS = 700;        // an attempt that starts this soon after a panel click is the click, not a sign
const REST_WAIT_S = 1.0, REST_SAMPLE_S = 2.0;
const STRICTNESS = [[0.85, 'Strict'], [1, 'Normal'], [1.25, 'Relaxed'], [1.5, 'Easy']] as const;

type Element = { setProperties(p: Record<string, unknown>): void; addEventListener(t: string, f: () => void): void; removeEventListener(t: string, f: () => void): void };

export class LessonSystem extends createSystem({}) {
  private el: Record<string, Element | null> = {};
  private lastClick = -Infinity;
  private now = 0;
  private infoTimer = 0;
  private calibrating: { start: number; samples: number[] } | null = null;
  private deleteArmedUntil = 0;
  private fps = 0;
  private controllerHint = false;

  init(): void {
    const panel = this.world.getSceneObject<UIKitMLAsset>('lesson-panel');
    if (!panel) return;
    // Keep the panel to the left of the user's view, within reach for poking, whether they sit or stand.
    const panelEntity = this.world.getSceneEntity('lesson-panel');
    panelEntity?.addComponent(Follower, {
      target: this.player.head, offsetPosition: [-0.3, -0.28, -0.48], behavior: FollowBehavior.PivotY,
    });
    for (const id of ['unit', 'gloss', 'mastery', 'mode', 'status', 'hint', 'info', 'mode-label', 'hand-label', 'strict-label',
      'delete-label', 'preview-label', 'rest-label', 'btn-prev', 'btn-next', 'btn-mode', 'btn-rest', 'btn-hand', 'btn-strict',
      'btn-delete', 'btn-preview', 'xr-button', 'exit-button'])
      this.el[id] = panel.getElementById(id) as unknown as Element | null;

    this.on('btn-prev', () => app.step(-1));
    this.on('btn-next', () => app.step(1));
    this.on('btn-mode', () => {
      app.mode = app.mode === 'free' ? (STUDIO_ENABLED ? 'studio' : 'practice') : 'free';
      this.say(app.mode === 'free' ? 'Sign any sign you have learned' : 'Copy the shadow hands', '');
      app.changed();
    });
    this.on('btn-hand', () => app.updateSettings({ leftHanded: !app.settings.leftHanded }));
    this.on('btn-strict', () => {
      const i = STRICTNESS.findIndex(([v]) => v === app.settings.strictness);
      app.updateSettings({ strictness: STRICTNESS[(i + 1) % STRICTNESS.length][0] });
    });
    this.on('btn-rest', () => {
      this.calibrating = { start: this.now + REST_WAIT_S, samples: [] };
      this.say('Rest your hands', 'Put your hands where they rest between signs, in your lap or at your sides.');
    });
    this.on('btn-preview', () => {
      app.preview = !app.preview;
      this.say(app.preview ? 'Week preview (simulated)' : 'Your own progress',
        app.preview ? 'Shows what a week of practice looks like: learned and mastered signs and a streak. Nothing is saved.' : '');
      app.changed();
    });
    this.on('btn-delete', () => {
      if (this.now < this.deleteArmedUntil) {
        app.deleteMyData();
        this.deleteArmedUntil = 0;
        this.say('Deleted', 'Your progress and settings were deleted from this headset.');
      } else {
        this.deleteArmedUntil = this.now + 3;
        this.set('delete-label', 'Tap again to delete');
      }
    });
    if (this.world.xrEnabled) {
      this.on('xr-button', () => this.world.launchXR());
      this.on('exit-button', () => this.world.exitXR());
      this.cleanupFuncs.push(this.world.visibilityState.subscribe((v) => {
        const flat = v === VisibilityState.NonImmersive;
        this.el['xr-button']?.setProperties({ display: flat ? 'flex' : 'none' });
        this.el['exit-button']?.setProperties({ display: flat ? 'none' : 'flex' });
        // In the headset the panel is poke-only: a pinch-shaped sign (O, F, D, 9) must never click it.
        // The desktop view keeps mouse clicks (RayInteractable also carries canvas pointer input).
        if (!panelEntity) return;
        if (flat && !panelEntity.hasComponent(RayInteractable)) panelEntity.addComponent(RayInteractable);
        if (!flat && panelEntity.hasComponent(RayInteractable)) panelEntity.removeComponent(RayInteractable);
      }));
    } else {
      this.el['xr-button']?.setProperties({ display: 'none' });
    }

    const sampler = this.world.getSystem(HandSamplerSystem) as HandSamplerSystem | undefined;
    if (sampler) this.cleanupFuncs.push(sampler.onAttempt((a) => this.onAttempt(a, sampler)));
    this.cleanupFuncs.push(app.subscribe(() => this.render()));
    this.render();
  }

  private on(id: string, fn: () => void): void {
    const el = this.el[id];
    if (!el) return;
    const handler = () => { this.lastClick = performance.now(); fn(); };
    el.addEventListener('click', handler);
    this.cleanupFuncs.push(() => el.removeEventListener('click', handler));
  }

  private set(id: string, text: string): void { this.el[id]?.setProperties({ text }); }
  private say(status: string, hint: string): void { this.set('status', status); this.set('hint', hint); }
  private gloss = (id: string) => app.pack?.catalogue.find((s) => s.id === id)?.gloss;

  private render(): void {
    if (!app.pack) {
      this.set('unit', app.packError ? 'Could not load signs' : 'Loading signs');
      this.set('gloss', 'Ashiria');
      if (app.packError) this.say('Something went wrong', app.packError);
      return;
    }
    const sign = app.current;
    const unit = app.pack.units.find((u) => u.id === sign?.unit);
    const unitIds = app.signs.filter((s) => s.unit === sign?.unit).map((s) => s.id);
    const progress = app.progressView;
    const sum = unitSummary(progress, unitIds);
    const where = unitIds.indexOf(sign?.id ?? '') + 1;
    this.set('unit', sign ? `${unit?.title ?? ''}   ${where} of ${sum.total}   (${sum.learned} learned, ${sum.mastered} mastered)` : 'No signs');
    if (app.mode === 'free') {
      this.set('gloss', 'Free Sign');
      this.set('mastery', `${app.learnedIds.length} signs to recognise`);
      this.set('mode', 'Sign any sign you have learned');
    } else {
      this.set('gloss', sign?.gloss ?? '');
      this.set('mastery', masteryText(progress.signs[sign?.id ?? '']?.level ?? 0));
      const gentle = sign?.tier === 'lenient' ? ' (gentle check)' : '';
      const phase = sign?.tier === 'watch' ? 'Watch and copy (not scored)'
        : app.phase === 'recall' ? 'From memory: the shadow is hidden' : `Copy the shadow hands${gentle}`;
      this.set('mode', app.mode === 'studio' ? 'Studio: every attempt is saved as a take' : phase);
    }
    this.set('mode-label', app.mode === 'free' ? 'Practice' : 'Free Sign');
    this.set('hand-label', app.settings.leftHanded ? 'Left hand' : 'Right hand');
    this.set('strict-label', STRICTNESS.find(([v]) => v === app.settings.strictness)?.[1] ?? 'Normal');
    this.set('delete-label', 'Delete my data');
    this.set('preview-label', app.preview ? 'Preview on' : 'Preview');
  }

  update(delta: number, time: number): void {
    this.now = time;
    const sampler = this.world.getSystem(HandSamplerSystem) as HandSamplerSystem | undefined;
    if (this.calibrating && sampler && time >= this.calibrating.start) {
      const f = sampler.lastFrame;
      for (let s = 0; s < 2; s++) if (f.present[s] > 0.5) this.calibrating.samples.push(f.wristPos[s * 3 + 1]);
      if (time >= this.calibrating.start + REST_SAMPLE_S) {
        const samples = this.calibrating.samples;
        this.calibrating = null;
        if (samples.length < 10) this.say('I could not see your hands', 'Keep your hands visible to the headset and try Set rest again.');
        else {
          app.updateSettings({ restHeight: restHeight(samples) });
          this.say('Rest height set', 'Now raise your hand into the shadow hands to try a sign.');
        }
      }
    }
    if (this.deleteArmedUntil && time > this.deleteArmedUntil) { this.deleteArmedUntil = 0; this.set('delete-label', 'Delete my data'); }
    if (delta > 0) this.fps = this.fps ? this.fps * 0.95 + (1 / delta) * 0.05 : 1 / delta;
    if (sampler) {
      const needHint = sampler.controllersOnly && sampler.sinceHand > 3;
      if (needHint && !this.controllerHint) this.say('Put your controllers down', 'Ashiria uses your hands. Set the controllers aside and hold up your hands.');
      if (!needHint && this.controllerHint) this.say('Hands found', 'Raise your hand into the shadow hands to try a sign.');
      this.controllerHint = needHint;
    }
    this.infoTimer -= delta;
    if (this.infoTimer <= 0 && sampler) {
      this.infoTimer = 0.25;
      const f = sampler.lastFrame;
      const hands = `${f.present[0] > 0.5 ? 'signing hand' : '-'} / ${f.present[1] > 0.5 ? 'other hand' : '-'}`;
      const dev = import.meta.env.DEV ? `   ${Math.round(this.fps)} fps${app.pack?.distribution === 'internal' ? '   Development pack' : ''}` : '';
      const streak = currentStreak(app.progressView, app.today);
      const days = streak ? `   Streak: ${streak} day${streak === 1 ? '' : 's'}${app.preview ? ' (simulated)' : ''}` : '';
      this.set('info', `Hands: ${hands}${days}${dev}`);
    }
  }

  private onAttempt(attempt: Attempt, sampler: HandSamplerSystem): void {
    if (!app.pack || this.calibrating) return;
    const started = performance.now() - ((attempt.end - attempt.start + 1) / 30) * 1000;
    if (started < this.lastClick + UI_GUARD_MS) return;
    const p = app.params();
    const prepared = prepare(attempt.segment, p, app.settings.restHeight);
    const info = app.current;
    if (!prepared) return;

    if (app.mode === 'free') {
      const candidates = app.learnedIds.map((id) => app.pack!.signs.get(id)!).filter(Boolean);
      const r = recognise(prepared, candidates, p);
      const g = (id: string | null) => (id ? this.gloss(id) ?? id : '');
      if (r.status === 'match') this.say(`That was ${g(r.best)}`, `Confidence ${(r.confidence * 100).toFixed(0)}%`);
      else if (r.status === 'ambiguous') this.say(`${g(r.best)} or ${g(r.runnerUp)}?`, 'They look alike. Try again with a clear handshape.');
      else if (r.status === 'not_sure') this.say('Not sure', 'That did not look like a sign you have learned. Try again.');
      else this.say('I lost sight of your hand', 'Keep your hand in front of you.');
      return;
    }

    if (!info) return;
    const sign = app.pack.signs.get(info.id);
    if (!sign) return;
    const r = verify(prepared, sign, p, app.settings.strictness, lookalikesOf(app.pack, info.id));

    if (app.mode === 'studio') {
      void saveTake(sampler, attempt, info.id).then(
        (n) => this.say(`Saved take ${n} for ${info.gloss}`, `Draft check: ${r.status}${r.hints[0] ? `, ${hintText(r.hints[0], this.gloss)}` : ''}`),
        (err) => this.say('Could not save the take', String(err)));
      return;
    }

    if (r.status === 'unscored') { this.say('Nice', 'This sign is shown for watching and copying. It is not scored.'); return; }
    if (r.status === 'no_hands') { this.say('I lost sight of your hand', hintText(r.hints[0])); return; }
    const shadow = this.world.getSystem(ShadowHandsSystem) as ShadowHandsSystem | undefined;
    const wasRecall = app.phase === 'recall';
    const out = app.result(info.id, r.status === 'pass');
    if (r.status === 'pass') {
      if (shadow) shadow.highlightUntil = 0;
      if (out.event === 'matched') this.say(`Yes! That is ${info.gloss}`, 'Now sign it from memory: the shadow hands are hidden.');
      else if (out.event === 'mastered') this.say(`Mastered ${info.gloss}!`, 'Recalled on three different days. Tap Next for the next sign.');
      else this.say(`Well done: ${info.gloss} from memory`, 'Come back tomorrow to master it. Tap Next for the next sign.');
      return;
    }
    const fingers = [0, 1, 2, 3, 4].map((f) => r.hints.some((h) => h.finger === ['thumb', 'index', 'middle', 'ring', 'pinky'][f]));
    if (shadow) { shadow.highlight = fingers; shadow.highlightUntil = this.now + 2.5; }
    const hint = r.hints[0] ? hintText(r.hints[0], this.gloss) : 'Try again.';
    const own = app.misses >= 2 && info.hint ? ` ${info.hint}` : '';
    if (out.event === 'back_to_match') this.say("Let's look again", `The shadow hands are back. ${hint}`);
    else this.say(wasRecall ? 'Not quite, try once more from memory' : 'Not quite', `${hint}${own}`);
  }
}

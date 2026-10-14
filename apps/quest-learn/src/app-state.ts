/**
 * Shared app state: the loaded content pack, the current mode and sign, and the user's settings and progress.
 * Settings and progress live only on this device (localStorage); hand data is never stored or sent anywhere.
 */
import { defaultParams, type ContentPack, type EngineParams } from '@ashiria/engine';
import { applyResult, dayOf, emptyProgress, learned, seen, simulatedWeek, type Outcome, type Phase, type Progress } from './flow.js';

export type Mode = 'practice' | 'free' | 'studio';

export interface Settings {
  leftHanded: boolean;
  strictness: number;     // tolerance scale: > 1 is more forgiving
  restHeight: number;     // shoulder widths; measured at onboarding
  shoulderWidth: number;  // metres
}

const SETTINGS_KEY = 'ashiria.settings.v1';
const PROGRESS_KEY = 'ashiria.progress.v2';
/** Everything this app keeps on the device (Delete my data / Start over remove all of it). */
const STORAGE_KEYS = [SETTINGS_KEY, PROGRESS_KEY, 'ashiria.progress.v1', 'ashiria.name.v1', 'ashiria.wall.v1', 'iwsdk_scene_anchor_uuid'];

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: keep in memory */ }
}

/** Studio is a development tool: only in dev builds, switched on with ?studio in the URL (remembered on this
 * device until ?studio=off). */
function studioSwitch(): boolean {
  if (!import.meta.env.DEV) return false;
  const q = new URLSearchParams(location.search).get('studio');
  try {
    if (q !== null) localStorage.setItem('ashiria.studio', q === 'off' ? '0' : '1');
    return localStorage.getItem('ashiria.studio') === '1';
  } catch {
    return q !== null && q !== 'off';
  }
}
export const STUDIO_ENABLED = studioSwitch();

class AppState {
  pack: ContentPack | null = null;
  packError = '';
  mode: Mode = STUDIO_ENABLED ? 'studio' : 'practice';
  signIndex = 0;
  settings: Settings = load(SETTINGS_KEY, { leftHanded: false, strictness: 1, restHeight: -1.5, shoulderWidth: 0.38 });
  progress: Progress = load(PROGRESS_KEY, emptyProgress());
  /** Practice phase for the current sign, and misses in that phase. */
  phase: Phase = 'match';
  misses = 0;
  /** Judge preview: show a simulated week of progress (never saved). */
  preview = false;
  private listeners = new Set<() => void>();

  /** Signs in teaching order (units, then order within the unit). */
  get signs() {
    const p = this.pack;
    if (!p) return [];
    const unitOrder = new Map(p.units.map((u) => [u.id, u.order]));
    return [...p.catalogue].sort((a, b) => (unitOrder.get(a.unit) ?? 0) - (unitOrder.get(b.unit) ?? 0) || a.order - b.order);
  }

  get current() { return this.signs[this.signIndex] ?? null; }

  get today(): string { return dayOf(new Date()); }

  /** The progress to show: the user's own, or a simulated week in judge preview. */
  get progressView(): Progress {
    return this.preview ? simulatedWeek(this.signs.map((s) => s.id), this.today) : this.progress;
  }

  /** Signs matched or better: Free Sign recognises among these (all recognisable signs until two are learned). */
  get learnedIds(): string[] {
    const recognisable = this.signs.filter((s) => s.recognise).map((s) => s.id);
    const ids = learned(this.progressView, recognisable);
    return ids.length >= 2 ? ids : recognisable;
  }

  /** Engine parameters: the pack's, with the user's own rest height. */
  params(): EngineParams {
    const base = this.pack?.params ?? defaultParams();
    return { ...base, segmenter: { ...base.segmenter, rest_height: this.settings.restHeight } };
  }

  subscribe(fn: () => void): () => void { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  changed(): void { this.listeners.forEach((fn) => fn()); }

  updateSettings(patch: Partial<Settings>): void { this.settings = { ...this.settings, ...patch }; save(SETTINGS_KEY, this.settings); this.changed(); }

  /** Apply a Verify result to the current sign: updates progress, phase and misses, and saves. */
  result(signId: string, passed: boolean): Outcome {
    const out = applyResult(this.progress, signId, this.phase, passed, this.misses, this.today);
    this.misses = out.phase !== this.phase || passed ? 0 : this.misses + 1;
    this.phase = out.phase;
    save(PROGRESS_KEY, this.progress);
    this.changed();
    return out;
  }

  /** Delete my data / Start over: erase everything this app stored on the device. */
  deleteMyData(): void {
    try { STORAGE_KEYS.forEach((k) => localStorage.removeItem(k)); } catch { /* ignore */ }
    this.progress = emptyProgress();
    this.settings = { leftHanded: false, strictness: 1, restHeight: -1.5, shoulderWidth: 0.38 };
    this.preview = false;
    this.goTo(0);
  }

  /** Show sign i: back to the match phase, and mark it seen. */
  goTo(i: number): void {
    const n = this.signs.length;
    if (!n) { this.changed(); return; }
    this.signIndex = ((i % n) + n) % n;
    this.phase = 'match';
    this.misses = 0;
    seen(this.progress, this.signs[this.signIndex].id);
    save(PROGRESS_KEY, this.progress);
    this.changed();
  }

  step(by: number): void { this.goTo(this.signIndex + by); }
}

export const app = new AppState();

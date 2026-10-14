/**
 * Shared app state: the loaded content pack, the current mode and sign, and the user's settings and progress.
 * Settings and progress live only on this device (localStorage); hand data is never stored or sent anywhere.
 */
import { defaultParams, type ContentPack, type EngineParams } from '@ashiria/engine';

export type Mode = 'practice' | 'free' | 'studio';

export interface Settings {
  leftHanded: boolean;
  strictness: number;     // tolerance scale: > 1 is more forgiving
  restHeight: number;     // shoulder widths; measured at onboarding
  shoulderWidth: number;  // metres
}

export interface Progress { passed: Record<string, number>; attempts: Record<string, number> }

const SETTINGS_KEY = 'ashiria.settings.v1';
const PROGRESS_KEY = 'ashiria.progress.v1';

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
  progress: Progress = load(PROGRESS_KEY, { passed: {}, attempts: {} });
  private listeners = new Set<() => void>();

  /** Signs in teaching order (units, then order within the unit). */
  get signs() {
    const p = this.pack;
    if (!p) return [];
    const unitOrder = new Map(p.units.map((u) => [u.id, u.order]));
    return [...p.catalogue].sort((a, b) => (unitOrder.get(a.unit) ?? 0) - (unitOrder.get(b.unit) ?? 0) || a.order - b.order);
  }

  get current() { return this.signs[this.signIndex] ?? null; }

  /** Signs the user has passed at least once: Free Sign recognises among these (all recognisable signs until then). */
  get learnedIds(): string[] {
    const recognisable = this.signs.filter((s) => s.recognise).map((s) => s.id);
    const learned = recognisable.filter((id) => (this.progress.passed[id] ?? 0) > 0);
    return learned.length >= 2 ? learned : recognisable;
  }

  /** Engine parameters: the pack's, with the user's own rest height. */
  params(): EngineParams {
    const base = this.pack?.params ?? defaultParams();
    return { ...base, segmenter: { ...base.segmenter, rest_height: this.settings.restHeight } };
  }

  subscribe(fn: () => void): () => void { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  changed(): void { this.listeners.forEach((fn) => fn()); }

  updateSettings(patch: Partial<Settings>): void { this.settings = { ...this.settings, ...patch }; save(SETTINGS_KEY, this.settings); this.changed(); }

  record(signId: string, passed: boolean): void {
    this.progress.attempts[signId] = (this.progress.attempts[signId] ?? 0) + 1;
    if (passed) this.progress.passed[signId] = (this.progress.passed[signId] ?? 0) + 1;
    save(PROGRESS_KEY, this.progress);
  }

  /** Settings → Delete my data: erase progress and settings from this device. */
  deleteMyData(): void {
    try { localStorage.removeItem(SETTINGS_KEY); localStorage.removeItem(PROGRESS_KEY); } catch { /* ignore */ }
    this.progress = { passed: {}, attempts: {} };
    this.settings = { leftHanded: false, strictness: 1, restHeight: -1.5, shoulderWidth: 0.38 };
    this.changed();
  }

  step(by: number): void {
    const n = this.signs.length;
    if (n) { this.signIndex = (this.signIndex + by + n) % n; this.changed(); }
  }
}

export const app = new AppState();

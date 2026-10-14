/** Every engine parameter (port of params.py). Content packs carry them (manifest "engine"). */

export const RUNTIME_VERSION = 1;

export interface Weights { flexion: number; spread: number; tips: number; palm_normal: number; direction: number; wrist: number; present: number }

export interface SegmenterParams {
  tick_hz: number; rest_height: number; start_s: number; pre_roll_s: number; drop_s: number; hold_s: number;
  hold_radius: number; hold_shape: number; min_attempt_s: number; max_attempt_s: number; rearm_move: number;
}

export interface TierTolerances {
  finger: number; finger_mean: boolean; spread: number; orientation_deg: number; location: number; movement: number;
  other_hand: number; distance_z: number; lookalike_guard: boolean;
}

export interface VerifyParams {
  min_present: number; thumb_scale: number; dictionary_scale: number; take_spread_k: number; take_spread_cap: number;
  lookalike_ratio: number; scored: TierTolerances; lenient: TierTolerances;
}

export interface RecogniseParams { min_present: number; accept_z: number; min_margin_z: number; temperature: number; top: number }

export interface EngineParams {
  seq_len: number; gap_fill_frames: number; trim_margin_frames: number; active_fraction: number; active_min_range: number;
  active_margin: number; active_min_frames: number; dtw_band: number; weights: Weights;
  segmenter: SegmenterParams; verify: VerifyParams; recognise: RecogniseParams;
}

export function defaultParams(): EngineParams {
  return {
    seq_len: 32, gap_fill_frames: 5, trim_margin_frames: 2, active_fraction: 0.3, active_min_range: 0.3,
    active_margin: 2, active_min_frames: 6, dtw_band: 8,
    weights: { flexion: 1, spread: 1, tips: 0.5, palm_normal: 1, direction: 1, wrist: 3, present: 1 },
    segmenter: {
      tick_hz: 30, rest_height: -1.5, start_s: 0.1, pre_roll_s: 0.2, drop_s: 0.13, hold_s: 0.6, hold_radius: 0.12,
      hold_shape: 0.35, min_attempt_s: 0.35, max_attempt_s: 4.0, rearm_move: 0.3,
    },
    verify: {
      min_present: 0.6, thumb_scale: 1.4, dictionary_scale: 1.25, take_spread_k: 2.5, take_spread_cap: 2.0, lookalike_ratio: 0.9,
      scored: { finger: 0.35, finger_mean: false, spread: 0.3, orientation_deg: 35, location: 0.35, movement: 0.15, other_hand: 0.5, distance_z: 1.3, lookalike_guard: true },
      lenient: { finger: 0.5, finger_mean: false, spread: 0.45, orientation_deg: 50, location: 0.5, movement: 0.22, other_hand: 0.75, distance_z: 1.6, lookalike_guard: true },
    },
    recognise: { min_present: 0.5, accept_z: 1.25, min_margin_z: 0.15, temperature: 0.5, top: 5 },
  };
}

type Json = { [k: string]: unknown };

function merge<T>(base: T, over: unknown): T {
  if (over === null || typeof over !== 'object' || Array.isArray(over)) return (over === undefined ? base : over) as T;
  const out: Json = { ...(base as Json) };
  for (const [k, v] of Object.entries(over as Json)) {
    if (k === 'runtime_version') continue;
    if (!(k in out)) throw new Error(`unknown engine parameter ${k}`);
    out[k] = typeof out[k] === 'object' && out[k] !== null ? merge(out[k], v) : v;
  }
  return out as T;
}

/** Parameters from a pack manifest's "engine" object (missing keys keep the defaults). */
export function paramsFromJson(engine: unknown): EngineParams {
  return merge(defaultParams(), engine);
}

/** Seconds to whole ticks, at least one (Python's round-half-to-even). */
export function frames(p: SegmenterParams, seconds: number): number {
  const x = seconds * p.tick_hz;
  const r = Math.round(x);
  return Math.max(1, Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r);
}

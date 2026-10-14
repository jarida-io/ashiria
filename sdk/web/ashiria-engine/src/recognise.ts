/** Recognise: "which sign was that?" for Free Sign (port of recognise.py; spec/RUNTIME_SPEC.md §5). */
import { sequenceFeatures } from './features.ts';
import { DOMINANT } from './joints.ts';
import { descriptor } from './matching.ts';
import type { EngineParams } from './params.ts';
import type { Segment } from './segment.ts';
import { closest, type SignRef } from './signs.ts';

export interface RecogniseResult {
  status: 'match' | 'ambiguous' | 'not_sure' | 'no_hands';
  best: string | null; runnerUp: string | null; bestZ: number; runnerUpZ: number; margin: number; confidence: number;
  ranking: { id: string; distance: number; z: number }[];
}

export function recognise(attempt: Segment, candidates: SignRef[], p: EngineParams): RecogniseResult {
  const rp = p.recognise;
  const none: RecogniseResult = { status: 'no_hands', best: null, runnerUp: null, bestZ: Infinity, runnerUpZ: Infinity, margin: Infinity, confidence: 0, ranking: [] };
  let present = 0;
  for (let t = 0; t < attempt.length; t++) if (attempt.present[t * 2 + DOMINANT] > 0.5) present++;
  if (!candidates.length || present / attempt.length < rp.min_present) return none;
  const desc = descriptor(sequenceFeatures(attempt), p.weights);
  const dist = candidates.map((c) => closest(c, desc, p.dtw_band).distance);
  const z = dist.map((d, i) => d / candidates[i].tau);
  const order = z.map((_, i) => i).sort((a, b) => z[a] - z[b]);  // stable: ties keep the candidates' order
  const logits = z.map((v) => -v / rp.temperature);
  const maxLogit = Math.max(...logits);
  const sum = logits.reduce((a, l) => a + Math.exp(l - maxLogit), 0);
  const b = order[0];
  const r: RecogniseResult = {
    status: 'match', best: candidates[b].id, runnerUp: null, bestZ: z[b], runnerUpZ: Infinity, margin: Infinity,
    confidence: Math.exp(logits[b] - maxLogit) / sum,
    ranking: order.slice(0, rp.top).map((i) => ({ id: candidates[i].id, distance: dist[i], z: z[i] })),
  };
  if (order.length > 1) {
    const ru = order[1];
    r.runnerUp = candidates[ru].id; r.runnerUpZ = z[ru]; r.margin = r.runnerUpZ - r.bestZ;
  }
  if (r.bestZ > rp.accept_z) r.status = 'not_sure';
  else if (r.margin < rp.min_margin_z && r.runnerUpZ <= rp.accept_z) r.status = 'ambiguous';
  return r;
}

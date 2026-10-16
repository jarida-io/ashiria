/**
 * The practice loop and progress, as pure functions (no rendering, no storage), so they are unit-tested
 * (`node --test src/flow.test.ts`).
 *
 * Per sign: match (copy the shadow hands) → recall (from memory, shadow hidden). Two misses in recall bring the
 * shadow back. Mastery: seen → matched → recalled → mastered (recalled on 3 different days). A phrase goes the same
 * way, signed part by part: each sign it is made of in turn, the phrase's result coming with its last sign.
 */

export type Phase = 'match' | 'recall';
export type Level = 0 | 1 | 2 | 3 | 4;
export const LEVEL_NAMES = ['new', 'seen', 'matched', 'recalled', 'mastered'] as const;
export const MASTERY_DAYS = 3;
export const RECALL_MISSES = 2;

export interface SignProgress { level: Level; recalledDays: string[]; attempts: number; passes: number }
export interface Progress { signs: Record<string, SignProgress>; streak: { lastDay: string | null; count: number } }

export const emptyProgress = (): Progress => ({ signs: {}, streak: { lastDay: null, count: 0 } });
const sign = (p: Progress, id: string): SignProgress =>
  (p.signs[id] ??= { level: 0, recalledDays: [], attempts: 0, passes: 0 });

/** Local calendar day, e.g. 2026-10-14. */
export const dayOf = (d: Date): string =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

function previousDay(day: string): string {
  const [y, m, d] = day.split('-').map(Number);
  return dayOf(new Date(y, m - 1, d - 1));
}

/** A sign was shown to the user. */
export function seen(p: Progress, id: string): void {
  const s = sign(p, id);
  if (s.level < 1) s.level = 1;
}

/** Count a practice day for the streak (call on any pass). */
export function touchStreak(p: Progress, today: string): void {
  const st = p.streak;
  if (st.lastDay === today) return;
  st.count = st.lastDay === previousDay(today) ? st.count + 1 : 1;
  st.lastDay = today;
}

/** The streak as it stands today: it is broken if the last practice day was before yesterday. */
export function currentStreak(p: Progress, today: string): number {
  const last = p.streak.lastDay;
  return last === today || last === previousDay(today) ? p.streak.count : 0;
}

export interface Outcome { phase: Phase; event: 'matched' | 'recalled' | 'mastered' | 'miss' | 'back_to_match' | 'part' }

/**
 * Apply a Verify result for sign `id` in `phase`. `misses` counts misses in the current phase (the caller keeps
 * it per sign and resets it when the phase changes). Returns the next phase and what happened.
 */
export function applyResult(p: Progress, id: string, phase: Phase, passed: boolean, misses: number, today: string): Outcome {
  const s = sign(p, id);
  s.attempts++;
  if (!passed) {
    if (phase === 'recall' && misses + 1 >= RECALL_MISSES) return { phase: 'match', event: 'back_to_match' };
    return { phase, event: 'miss' };
  }
  s.passes++;
  touchStreak(p, today);
  if (phase === 'match') {
    if (s.level < 2) s.level = 2;
    return { phase: 'recall', event: 'matched' };
  }
  if (!s.recalledDays.includes(today)) s.recalledDays.push(today);
  if (s.recalledDays.length >= MASTERY_DAYS) {
    s.level = 4;
    return { phase: 'recall', event: 'mastered' };
  }
  if (s.level < 3) s.level = 3;
  return { phase: 'recall', event: 'recalled' };
}

/**
 * A Verify result for part `part` of phrase `id` (made of `parts` signs). A passed sign that isn't the last moves
 * on to the next part without touching progress; the last one, or any miss, counts for the phrase (applyResult).
 * Returns the outcome and the part to sign next.
 */
export function applyPhraseResult(p: Progress, id: string, phase: Phase, part: number, parts: number, passed: boolean,
  misses: number, today: string): { out: Outcome; part: number } {
  if (passed && part + 1 < parts) return { out: { phase, event: 'part' }, part: part + 1 };
  const out = applyResult(p, id, phase, passed, misses, today);
  return { out, part: passed ? 0 : part };
}

/** Signs the user can sign without the shadow (matched or better): Free Sign searches these. */
export const learned = (p: Progress, ids: string[]): string[] => ids.filter((id) => (p.signs[id]?.level ?? 0) >= 2);

/** Mastery in words for the panel (plain ASCII: the panel font may lack symbol glyphs), e.g. "Matched (2 of 4)". */
export function masteryText(level: number): string {
  const n = Math.min(4, Math.max(0, level));
  const name = LEVEL_NAMES[n];
  return `${name[0].toUpperCase()}${name.slice(1)} (${n} of 4)`;
}

/**
 * Judge preview: a simulated week of progress, so judges see growth in one sitting. Never saved; the panel labels
 * it "simulated". The first signs are mastered, the next recalled or matched, and the streak is five days.
 */
export function simulatedWeek(ids: string[], today: string): Progress {
  const p = emptyProgress();
  const days = [4, 3, 2, 1, 0].map((back) => {
    const [y, m, d] = today.split('-').map(Number);
    return dayOf(new Date(y, m - 1, d - back));
  });
  ids.slice(0, 18).forEach((id, i) => {
    const level: Level = i < 6 ? 4 : i < 12 ? 3 : 2;
    p.signs[id] = { level, recalledDays: level >= 3 ? days.slice(0, level === 4 ? 3 : 1) : [], attempts: 6, passes: 4 };
  });
  p.streak = { lastDay: today, count: 5 };
  return p;
}

/** Counts per unit for the panel: how many signs reached each level. */
export function unitSummary(p: Progress, ids: string[]): { mastered: number; learned: number; total: number } {
  const levels = ids.map((id) => p.signs[id]?.level ?? 0);
  return { mastered: levels.filter((l) => l >= 4).length, learned: levels.filter((l) => l >= 2).length, total: ids.length };
}

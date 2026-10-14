import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  applyResult, currentStreak, dayOf, emptyProgress, learned, masteryText, seen, simulatedWeek, touchStreak, unitSummary,
} from './flow.ts';

test('match then recall, two recall misses go back to match', () => {
  const p = emptyProgress();
  seen(p, 'a');
  assert.equal(p.signs.a.level, 1);
  assert.deepEqual(applyResult(p, 'a', 'match', false, 0, '2026-10-14'), { phase: 'match', event: 'miss' });
  assert.deepEqual(applyResult(p, 'a', 'match', true, 1, '2026-10-14'), { phase: 'recall', event: 'matched' });
  assert.equal(p.signs.a.level, 2);
  assert.deepEqual(applyResult(p, 'a', 'recall', false, 0, '2026-10-14'), { phase: 'recall', event: 'miss' });
  assert.deepEqual(applyResult(p, 'a', 'recall', false, 1, '2026-10-14'), { phase: 'match', event: 'back_to_match' });
  assert.deepEqual(applyResult(p, 'a', 'recall', true, 0, '2026-10-14'), { phase: 'recall', event: 'recalled' });
  assert.equal(p.signs.a.level, 3);
  assert.equal(p.signs.a.attempts, 5);
});

test('mastered after recalls on three different days', () => {
  const p = emptyProgress();
  applyResult(p, 'b', 'match', true, 0, '2026-10-14');
  assert.equal(applyResult(p, 'b', 'recall', true, 0, '2026-10-14').event, 'recalled');
  assert.equal(applyResult(p, 'b', 'recall', true, 0, '2026-10-14').event, 'recalled');  // same day: no progress
  assert.equal(applyResult(p, 'b', 'recall', true, 0, '2026-10-15').event, 'recalled');
  assert.equal(applyResult(p, 'b', 'recall', true, 0, '2026-10-17').event, 'mastered');
  assert.equal(p.signs.b.level, 4);
});

test('streak counts consecutive practice days and breaks after a gap', () => {
  const p = emptyProgress();
  touchStreak(p, '2026-10-30'); touchStreak(p, '2026-10-31'); touchStreak(p, '2026-11-01'); touchStreak(p, '2026-11-01');
  assert.equal(p.streak.count, 3);  // across a month boundary
  assert.equal(currentStreak(p, '2026-11-02'), 3);
  assert.equal(currentStreak(p, '2026-11-03'), 0);
  touchStreak(p, '2026-11-04');
  assert.equal(p.streak.count, 1);
});

test('learned signs, summaries, mastery text and the simulated week', () => {
  const ids = Array.from({ length: 30 }, (_, i) => `s${i}`);
  const p = simulatedWeek(ids, '2026-10-14');
  assert.equal(learned(p, ids).length, 18);
  assert.deepEqual(unitSummary(p, ids), { mastered: 6, learned: 18, total: 30 });
  assert.equal(currentStreak(p, '2026-10-14'), 5);
  assert.equal(masteryText(2), 'Matched (2 of 4)');
  assert.equal(masteryText(0), 'New (0 of 4)');
  assert.equal(dayOf(new Date(2026, 0, 5)), '2026-01-05');
});

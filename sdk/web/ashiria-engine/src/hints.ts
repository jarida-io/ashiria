/** Default English wording for Verify hints (port of HintText.cs). A sign's own catalogue hint is shown after two misses. */
import type { Hint } from './verify.ts';

export const HINT_TEXT: Record<string, string> = {
  'tracking.lost': 'I lost sight of your hand. Keep it in front of you.',
  'finger.bend_more': 'Bend your {finger} more.',
  'finger.straighten': 'Straighten your {finger}.',
  'spread.more': 'Spread your fingers apart.',
  'spread.less': 'Bring your fingers closer together.',
  'palm.face': 'Turn your palm to face {direction}.',
  'hand.point': 'Point your fingers {direction}.',
  move: 'Move your hand {direction}.',
  'movement.bigger': 'Make the movement bigger.',
  'movement.smaller': 'Make the movement smaller.',
  'movement.path': "Follow the shadow hand's movement.",
  'other_hand.missing': 'This sign uses both hands.',
  'other_hand.place': 'Move your other hand {direction}.',
  'overall.different': 'Not quite. Watch the shadow hand and try again.',
  lookalike: 'That looks like {sign}. Check the handshape.',
};

const FINGERS: Record<string, string> = { thumb: 'thumb', index: 'index finger', middle: 'middle finger', ring: 'ring finger', pinky: 'little finger' };
const FACING: Record<string, string> = { up: 'up', down: 'down', forward: 'away from you', back: 'towards you', dominant_side: 'out to the side', other_side: 'across your body' };
const PLACE: Record<string, string> = { up: 'higher', down: 'lower', forward: 'further from you', back: 'closer to you', dominant_side: 'out to the side', other_side: 'across your body' };

export function hintText(h: Hint, gloss?: (id: string) => string | undefined): string {
  const text = HINT_TEXT[h.code] ?? HINT_TEXT['overall.different'];
  const dirs = h.code === 'move' || h.code === 'other_hand.place' ? PLACE : FACING;
  return text
    .replace('{finger}', (h.finger && FINGERS[h.finger]) || 'fingers')
    .replace('{direction}', (h.direction && dirs[h.direction]) || '')
    .replace('{sign}', (h.signId && (gloss?.(h.signId) ?? h.signId)) || 'another sign');
}

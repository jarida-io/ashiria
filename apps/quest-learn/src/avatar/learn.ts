// Learning playback for the avatar: a sign as a sequence of steps (`ashiria` steps.py: start, movements, end),
// each movement played slowly and then held on its key moment with its caption, so a learner can see one part at a
// time. Without steps, the whole clip plays at normal speed.
import type { AvatarPose } from './figure.js';

export type Phase = 'move' | 'hold' | 'rest';
export interface LearnState {
  frame: number;
  /** Index into pose.learn.steps, or -1 before the first. */
  step: number;
  phase: Phase;
  text: string;
  /** Hands that sign (0 the signer's right, 1 left): glow while they move. */
  hands: number[];
  /** Frames of the movement leading to this step's key moment, for drawing its path. */
  from: number;
  to: number;
}

interface Piece { phase: Phase; seconds: number; f0: number; f1: number; step: number; text: string }

const RISE_SPEED = 0.75, MOVE_SPEED = 0.5, END_SPEED = 0.6, REST_S = 0.9;
/** Long enough to read the step: about a third of a second a word, between 2 and 4.5 seconds. */
const holdFor = (text: string) => Math.min(4.5, Math.max(2, 0.8 + 0.3 * text.split(/\s+/).length));

export class LearnPlayer {
  readonly pieces: Piece[] = [];
  readonly duration: number;
  readonly hands: number[];
  readonly steps: string[];

  constructor(private pose: AvatarPose) {
    const fps = pose.fps, last = pose.frames - 1;
    const steps = pose.learn?.steps ?? [];
    this.steps = steps.map((s) => s.text);
    const names = pose.learn?.hands ?? ['right hand'];
    this.hands = names.map((n) => (n.startsWith('left') ? 1 : 0));
    const play = (f0: number, f1: number, speed: number, phase: Phase, step: number, text: string) => {
      if (f1 > f0) this.pieces.push({ phase, seconds: (f1 - f0) / (fps * speed), f0, f1, step, text });
    };
    if (!steps.length) {
      play(0, last, 1, 'move', -1, '');
      this.pieces.push({ phase: 'rest', seconds: REST_S, f0: last, f1: last, step: -1, text: '' });
    } else {
      let at = 0;
      steps.forEach((s, i) => {
        const speed = s.kind === 'start' ? RISE_SPEED : s.kind === 'end' ? END_SPEED : MOVE_SPEED;
        play(at, s.key, speed, 'move', i, s.text);
        this.pieces.push({ phase: 'hold', seconds: holdFor(s.text), f0: s.key, f1: s.key, step: i, text: s.text });
        at = s.key;
      });
      play(at, last, 1, 'rest', steps.length - 1, steps[steps.length - 1].text); // back down to rest, normal speed
      this.pieces.push({ phase: 'rest', seconds: REST_S, f0: last, f1: last, step: -1, text: '' });
    }
    this.duration = this.pieces.reduce((a, p) => a + p.seconds, 0);
  }

  /** Where the playback is `seconds` into the loop. */
  at(seconds: number): LearnState {
    let t = ((seconds % this.duration) + this.duration) % this.duration;
    for (const p of this.pieces) {
      if (t <= p.seconds || p === this.pieces[this.pieces.length - 1]) {
        const u = p.seconds > 0 ? Math.min(1, t / p.seconds) : 1;
        const step = this.pose.learn?.steps[p.step];
        return {
          frame: Math.round(p.f0 + (p.f1 - p.f0) * u), step: p.step, phase: p.phase, text: p.text, hands: this.hands,
          from: step ? (p.step > 0 ? this.pose.learn!.steps[p.step - 1].key : step.from) : 0, to: step ? step.key : 0,
        };
      }
      t -= p.seconds;
    }
    return { frame: 0, step: -1, phase: 'rest', text: '', hands: this.hands, from: 0, to: 0 };
  }
}

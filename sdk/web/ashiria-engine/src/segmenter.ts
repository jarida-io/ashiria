/** Cuts the live stream of canonical frames into attempts (port of segmenter.py; spec/RUNTIME_SPEC.md §3).
 * Feed exactly one frame per fixed tick (30 Hz). */
import { flexion } from './features.ts';
import { COUNT, DOMINANT } from './joints.ts';
import { median, type V3 } from './math.ts';
import { frames, type SegmenterParams } from './params.ts';
import { emptySegment, type Segment } from './segment.ts';

export interface Attempt { segment: Segment; start: number; end: number; reason: 'hold' | 'drop' | 'timeout' }

/** One canonical frame: handRel [2 × 21 × 3], wristPos [2 × 3], present [2]. */
export interface Frame { handRel: Float64Array; wristPos: Float64Array; present: Float64Array }

interface Buffered extends Frame { tick: number; up: boolean; bends: number[] | null }

export class AttemptSegmenter {
  readonly p: SegmenterParams;
  private readonly startN: number; private readonly preN: number; private readonly dropN: number;
  private readonly holdN: number; private readonly holdAllow: number; private readonly minN: number; private readonly maxN: number;
  private pre: Buffered[] = [];
  private buf: Buffered[] = [];
  private upRun = 0;
  private downRun = 0;
  private heldAt: V3 | null = null;
  tick = -1;
  state: 'idle' | 'active' | 'cooldown' = 'idle';

  constructor(p: SegmenterParams) {
    this.p = p;
    this.startN = frames(p, p.start_s); this.preN = frames(p, p.pre_roll_s); this.dropN = frames(p, p.drop_s);
    this.holdN = frames(p, p.hold_s); this.holdAllow = Math.floor(this.holdN / 10);
    this.minN = frames(p, p.min_attempt_s); this.maxN = frames(p, p.max_attempt_s);
  }

  reset(): void {
    this.tick = -1; this.state = 'idle'; this.pre = []; this.buf = []; this.upRun = 0; this.downRun = 0; this.heldAt = null;
  }

  /** Push one frame (copied). Returns an attempt when one ends. */
  push(frame: Frame): Attempt | null {
    this.tick++;
    const ok = [frame.present[0] > 0.5, frame.present[1] > 0.5];
    const up = [0, 1].some((s) => ok[s] && frame.wristPos[s * 3 + 1] >= this.p.rest_height);
    let bends: number[] | null = null;
    if (ok[DOMINANT]) {
      const h: V3[] = [];
      for (let j = 0; j < COUNT; j++) h.push([frame.handRel[j * 3], frame.handRel[j * 3 + 1], frame.handRel[j * 3 + 2]]);
      bends = flexion(h);
    }
    const f: Buffered = { handRel: frame.handRel.slice(), wristPos: frame.wristPos.slice(), present: frame.present.slice(), tick: this.tick, up, bends };

    if (this.state === 'cooldown') {
      this.downRun = up ? 0 : this.downRun + 1;
      let moved = false;
      if (ok[DOMINANT] && this.heldAt) {
        const d = [f.wristPos[0] - this.heldAt[0], f.wristPos[1] - this.heldAt[1], f.wristPos[2] - this.heldAt[2]];
        // Moving on to a new sign re-arms; lowering the hand (on its way to rest) does not.
        moved = Math.hypot(d[0], d[1], d[2]) > this.p.rearm_move && d[1] > -this.p.rearm_move;
      }
      if (this.downRun < this.dropN && !moved) return null;
      this.toIdle();
    }

    if (this.state === 'idle') {
      this.pre.push(f);
      if (this.pre.length > this.preN + this.startN) this.pre.shift();
      this.upRun = up ? this.upRun + 1 : 0;
      if (this.upRun >= this.startN) { this.state = 'active'; this.buf = [...this.pre]; this.downRun = 0; }
      return null;
    }

    this.buf.push(f);
    this.downRun = up ? 0 : this.downRun + 1;
    if (this.downRun >= this.dropN) return this.finish('drop');
    if (this.held()) return this.finish('hold');
    if (this.buf.length >= this.maxN) return this.finish('timeout');
    return null;
  }

  private toIdle(): void {
    this.state = 'idle'; this.upRun = 0; this.downRun = 0; this.heldAt = null; this.pre = [];
  }

  private held(): boolean {
    if (this.buf.length < this.holdN) return false;
    const window = this.buf.slice(-this.holdN).filter((f) => f.bends !== null);
    if (window.length < this.holdN - this.holdAllow) return false;
    const centre = [0, 1, 2].map((c) => median(window.map((f) => f.wristPos[c])));
    const bendMedian = Array.from({ length: 15 }, (_, b) => median(window.map((f) => f.bends![b])));
    let still = 0;
    for (const f of window) {
      const dist = Math.hypot(f.wristPos[0] - centre[0], f.wristPos[1] - centre[1], f.wristPos[2] - centre[2]);
      let dev = 0;
      for (let b = 0; b < 15; b++) dev += Math.abs(f.bends![b] - bendMedian[b]);
      if (dist < this.p.hold_radius && dev / 15 < this.p.hold_shape) still++;
    }
    return still >= this.holdN - this.holdAllow;
  }

  private finish(reason: Attempt['reason']): Attempt | null {
    const buf = this.buf;
    this.buf = [];
    if (reason === 'drop') this.toIdle();
    else {
      this.state = 'cooldown'; this.downRun = 0;
      const dom = buf.slice(-this.holdN).filter((f) => f.present[DOMINANT] > 0.5);
      this.heldAt = dom.length ? ([0, 1, 2].map((c) => median(dom.map((f) => f.wristPos[c]))) as V3) : null;
    }
    if (buf.filter((f) => f.up).length < this.minN) return null;
    const seg = emptySegment(buf.length, this.p.tick_hz);
    buf.forEach((f, t) => {
      seg.handRel.set(f.handRel, t * 2 * COUNT * 3);
      seg.wristPos.set(f.wristPos, t * 6);
      seg.present.set(f.present, t * 2);
    });
    return { segment: seg, start: buf[0].tick, end: buf[buf.length - 1].tick, reason };
  }
}

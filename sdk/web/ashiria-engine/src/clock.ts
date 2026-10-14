/** Fixed-rate sampling for a variable frame rate (port of TickClock.cs): call `advance` every rendered frame,
 * then push the latest tracking that many times. After a stall, at most `maxCatchUp` ticks and the backlog is dropped. */
export class TickClock {
  readonly hz: number;
  private readonly maxCatchUp: number;
  private acc = 0;

  constructor(hz = 30, maxCatchUp = 3) { this.hz = hz; this.maxCatchUp = maxCatchUp; }

  advance(deltaSeconds: number): number {
    this.acc += Math.max(0, deltaSeconds);
    const period = 1 / this.hz;
    let ticks = Math.floor(this.acc / period + 1e-9);
    this.acc -= ticks * period;
    if (ticks > this.maxCatchUp) { ticks = this.maxCatchUp; this.acc = 0; }
    return ticks;
  }

  reset(): void { this.acc = 0; }
}

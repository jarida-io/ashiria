using System;

namespace Ashiria.Engine
{
    /// <summary>
    /// Fixed-rate sampling for a variable frame rate: call <see cref="Advance"/> every rendered frame with its delta
    /// time, then push the latest tracking that many times. The engine assumes exactly <see cref="Hz"/> frames per
    /// second (segmenter timings, velocities), whatever the display refresh rate.
    /// </summary>
    public sealed class TickClock
    {
        public readonly float Hz;
        readonly int maxCatchUp;
        double accumulator;

        public TickClock(float hz = 30f, int maxCatchUp = 3)
        {
            Hz = hz;
            this.maxCatchUp = maxCatchUp;
        }

        /// <summary>Ticks due after <paramref name="deltaSeconds"/>. After a long stall (app paused, headset off) at
        /// most <c>maxCatchUp</c> ticks are returned and the backlog is dropped, rather than replaying stale poses.</summary>
        public int Advance(double deltaSeconds)
        {
            accumulator += Math.Max(0.0, deltaSeconds);
            double period = 1.0 / Hz;
            int ticks = (int)Math.Floor(accumulator / period + 1e-9);
            accumulator -= ticks * period;
            if (ticks > maxCatchUp)
            {
                ticks = maxCatchUp;
                accumulator = 0;
            }
            return ticks;
        }

        public void Reset() => accumulator = 0;
    }
}

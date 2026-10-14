using System;
using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>Per-user settings measured at onboarding (port of onboarding.py).</summary>
    public static class Onboarding
    {
        public const float RestMargin = 0.25f, RestMin = -1.9f, RestMax = -1.2f;

        /// <summary>The user's rest height (shoulder widths) from wrist heights sampled while their hands rest in their
        /// lap or at their sides: median + 0.25, clamped to [-1.9, -1.2]. Returns <paramref name="fallback"/> with
        /// fewer than 10 samples.</summary>
        public static float RestHeight(IReadOnlyList<float> restingWristY, float fallback = -1.5f)
        {
            if (restingWristY == null || restingWristY.Count < 10) return fallback;
            var values = new double[restingWristY.Count];
            for (int i = 0; i < values.Length; i++) values[i] = restingWristY[i];
            return (float)Math.Min(RestMax, Math.Max(RestMin, Stats.Median(values) + RestMargin));
        }
    }
}

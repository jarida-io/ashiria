using System;

namespace Ashiria.Engine
{
    /// <summary>Shared canonicalisation (spec §5): mirror, gap fill, trim and resample. Port of canonical.py.</summary>
    public static class Canonical
    {
        /// <summary>For every frame, the index of the nearest frame where <paramref name="valid"/> is true
        /// (ties go to the earlier frame). Returns -1 everywhere if no frame is valid.</summary>
        public static int[] NearestValid(bool[] valid)
        {
            int n = valid.Length;
            var prev = new int[n];
            var next = new int[n];
            int last = -1;
            for (int t = 0; t < n; t++)
            {
                if (valid[t]) last = t;
                prev[t] = last;
            }
            last = -1;
            for (int t = n - 1; t >= 0; t--)
            {
                if (valid[t]) last = t;
                next[t] = last;
            }
            var result = new int[n];
            for (int t = 0; t < n; t++)
            {
                int l = prev[t], r = next[t];
                result[t] = l < 0 ? r : r < 0 ? l : (t - l) <= (r - t) ? l : r;
            }
            return result;
        }

        /// <summary>Left-right mirror: negate x and swap the hand slots (makes a left-handed signer right-handed).</summary>
        public static Segment Mirror(Segment seg)
        {
            var s = Segment.Empty(seg.Length, seg.Fps);
            for (int t = 0; t < seg.Length; t++)
            {
                for (int slot = 0; slot < 2; slot++)
                {
                    int from = 1 - slot;
                    for (int j = 0; j < Joints.Count; j++)
                        s.HandRel[t, slot, j] = seg.HandRel[t, from, j].MirrorX();
                    s.WristPos[t, slot] = seg.WristPos[t, from].MirrorX();
                    s.Present[t, slot] = seg.Present[t, from];
                }
            }
            return s;
        }

        /// <summary>Linearly fill dropouts of at most <paramref name="maxGap"/> frames between two tracked frames.</summary>
        public static Segment GapFill(Segment seg, int maxGap)
        {
            var s = seg.WithFps(seg.Fps);
            for (int slot = 0; slot < 2; slot++)
            {
                int a = -1;
                for (int b = 0; b < seg.Length; b++)
                {
                    if (seg.Present[b, slot] <= 0.5f) continue;
                    int gap = b - a - 1;
                    if (a >= 0 && gap > 0 && gap <= maxGap)
                    {
                        for (int k = 1; k <= gap; k++)
                        {
                            float w = k / (float)(gap + 1);
                            int t = a + k;
                            s.WristPos[t, slot] = Vec3.Lerp(seg.WristPos[a, slot], seg.WristPos[b, slot], w);
                            for (int j = 0; j < Joints.Count; j++)
                                s.HandRel[t, slot, j] = Vec3.Lerp(seg.HandRel[a, slot, j], seg.HandRel[b, slot, j], w);
                            s.Present[t, slot] = 1f;
                        }
                    }
                    a = b;
                }
            }
            return s;
        }

        /// <summary>Keep the frames from the first to the last moment a hand is up in signing space, plus a margin.
        /// Returns null if no hand is ever present.</summary>
        public static Segment Trim(Segment seg, float restHeight, int margin)
        {
            int n = seg.Length;
            var active = new bool[n];
            var any = new bool[n];
            bool anyActive = false;
            for (int t = 0; t < n; t++)
            {
                float height = float.NegativeInfinity;
                for (int slot = 0; slot < 2; slot++)
                {
                    if (seg.Present[t, slot] > 0.5f)
                    {
                        any[t] = true;
                        height = MathF.Max(height, seg.WristPos[t, slot].Y);
                    }
                }
                active[t] = any[t] && height >= restHeight;
                anyActive |= active[t];
            }
            var use = anyActive ? active : any;
            int first = Array.IndexOf(use, true), lastIndex = Array.LastIndexOf(use, true);
            if (first < 0) return null;
            return seg.Slice(Math.Max(0, first - margin), Math.Min(n, lastIndex + margin + 1));
        }

        /// <summary>Cut the rise from rest and the drop back: keep frames where the dominant wrist is above
        /// <paramref name="fraction"/> of its own height range. Unchanged if the height range is under 0.3 shoulder widths.</summary>
        public static Segment ActiveWindow(Segment seg, float fraction = 0.3f, int margin = 2, int minFrames = 6)
        {
            int n = seg.Length;
            int count = 0;
            double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
            for (int t = 0; t < n; t++)
            {
                if (seg.Present[t, Joints.Dominant] <= 0.5f) continue;
                count++;
                double y = seg.WristPos[t, Joints.Dominant].Y;
                lo = Math.Min(lo, y);
                hi = Math.Max(hi, y);
            }
            if (count < 3 || hi - lo < 0.3) return seg;
            double threshold = lo + fraction * (hi - lo);
            int first = -1, last = -1;
            for (int t = 0; t < n; t++)
            {
                if (seg.Present[t, Joints.Dominant] > 0.5f && seg.WristPos[t, Joints.Dominant].Y >= threshold)
                {
                    if (first < 0) first = t;
                    last = t;
                }
            }
            int a = Math.Max(0, first - margin), b = Math.Min(n, last + margin + 1);
            return b - a >= minFrames ? seg.Slice(a, b) : seg;
        }

        public const float HoldRadius = 0.12f;

        /// <summary>Frames where the wrist, after a centred 5-frame moving average (zero-padded at the ends),
        /// stays within <see cref="HoldRadius"/> of its median over the present frames.</summary>
        public static bool[] HoldFrames(Segment seg, int slot = Joints.Dominant, int smooth = 5)
        {
            int n = seg.Length;
            var hold = new bool[n];
            int present = 0;
            for (int t = 0; t < n; t++) if (seg.Present[t, slot] > 0.5f) present++;
            if (present < 3) return hold;

            var p = new double[n, 3];
            for (int t = 0; t < n; t++)
            {
                var w = seg.WristPos[t, slot];
                p[t, 0] = w.X; p[t, 1] = w.Y; p[t, 2] = w.Z;
            }
            if (n >= smooth)
            {
                var s = new double[n, 3];
                int half = (smooth - 1) / 2;
                for (int t = 0; t < n; t++)
                    for (int c = 0; c < 3; c++)
                    {
                        double sum = 0;
                        for (int m = t - half; m <= t + half; m++)
                            if (m >= 0 && m < n) sum += p[m, c];
                        s[t, c] = sum / smooth;
                    }
                p = s;
            }
            var centre = new double[3];
            var values = new double[present];
            for (int c = 0; c < 3; c++)
            {
                int k = 0;
                for (int t = 0; t < n; t++) if (seg.Present[t, slot] > 0.5f) values[k++] = p[t, c];
                centre[c] = Median(values);
            }
            for (int t = 0; t < n; t++)
            {
                if (seg.Present[t, slot] <= 0.5f) continue;
                double dx = p[t, 0] - centre[0], dy = p[t, 1] - centre[1], dz = p[t, 2] - centre[2];
                hold[t] = Math.Sqrt(dx * dx + dy * dy + dz * dz) < HoldRadius;
            }
            return hold;
        }

        /// <summary>The frame in the middle 60% where the dominant wrist moves least (first if tied).</summary>
        public static int SteadiestFrame(Segment seg)
        {
            int n = seg.Length;
            int lo = (int)(n * 0.2), hi = Math.Max((int)(n * 0.8), (int)(n * 0.2) + 1);
            int best = -1;
            double bestSpeed = double.PositiveInfinity;
            for (int t = Math.Max(lo, 1); t < Math.Min(hi, n); t++)
            {
                if (seg.Present[t, Joints.Dominant] <= 0.5f) continue;
                double speed = (seg.WristPos[t, Joints.Dominant] - seg.WristPos[t - 1, Joints.Dominant]).Length;
                if (speed < bestSpeed)
                {
                    bestSpeed = speed;
                    best = t;
                }
            }
            return best >= 0 ? best : n / 2;
        }

        static double Median(double[] values)
        {
            var sorted = (double[])values.Clone();
            Array.Sort(sorted);
            int m = sorted.Length / 2;
            return sorted.Length % 2 == 1 ? sorted[m] : 0.5 * (sorted[m - 1] + sorted[m]);
        }

        /// <summary>Resample to exactly <paramref name="n"/> frames: positions interpolate linearly (untracked frames
        /// hold the nearest tracked pose first); presence takes the nearest frame (round half to even).</summary>
        public static Segment Resample(Segment seg, int n)
        {
            int length = seg.Length;
            var held = seg.WithFps(seg.Fps);
            for (int slot = 0; slot < 2; slot++)
            {
                var ok = new bool[length];
                bool anyOk = false;
                for (int t = 0; t < length; t++)
                {
                    ok[t] = seg.Present[t, slot] > 0.5f;
                    anyOk |= ok[t];
                }
                if (!anyOk) continue;
                var near = NearestValid(ok);
                for (int t = 0; t < length; t++)
                {
                    held.WristPos[t, slot] = seg.WristPos[near[t], slot];
                    for (int j = 0; j < Joints.Count; j++)
                        held.HandRel[t, slot, j] = seg.HandRel[near[t], slot, j];
                }
            }

            double duration = Math.Max(length - 1, 1) / (double)seg.Fps;
            var s = Segment.Empty(n, (float)((n - 1) / duration));
            double step = n > 1 ? (length - 1) / (double)(n - 1) : 0.0;
            for (int k = 0; k < n; k++)
            {
                double src = (k == n - 1 && n > 1) ? length - 1 : k * step;
                int i0 = (int)Math.Floor(src);
                int i1 = Math.Min(i0 + 1, length - 1);
                float f = (float)(src - i0);
                int nearest = (int)Math.Round(src, MidpointRounding.ToEven);
                for (int slot = 0; slot < 2; slot++)
                {
                    s.WristPos[k, slot] = Vec3.Lerp(held.WristPos[i0, slot], held.WristPos[i1, slot], f);
                    for (int j = 0; j < Joints.Count; j++)
                        s.HandRel[k, slot, j] = Vec3.Lerp(held.HandRel[i0, slot, j], held.HandRel[i1, slot, j], f);
                    s.Present[k, slot] = seg.Present[nearest, slot];
                }
            }
            return s;
        }
    }
}

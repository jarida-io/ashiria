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

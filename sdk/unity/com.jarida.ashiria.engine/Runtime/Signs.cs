using System;
using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>The dominant hand's shape, orientation and place at a key moment, plus the other hand's place
    /// (port of signs.KeyPose).</summary>
    public sealed class KeyPose
    {
        public double[] Flexion = new double[15];
        public double[] Spread = new double[4];
        public double[] PalmNormal = new double[3];
        public double[] Direction = new double[3];
        public double[] Wrist = new double[3];
        public bool OtherPresent;
        public double[] OtherRel = new double[3];

        const int Other = Features.HandDim;
        const int Inter = 2 * Features.HandDim;

        /// <summary>Per-component median over <paramref name="rows"/> of a [frames, 197] feature array, using only
        /// rows where the dominant hand is present (all rows if none are); directions re-normalised.</summary>
        public static KeyPose From(float[,] f, IReadOnlyList<int> rows)
        {
            var use = new List<int>();
            var both = new List<int>();
            int other = 0;
            foreach (int r in rows)
            {
                bool dom = f[r, Matching.PresentAt] > 0.5f, oth = f[r, Other + Matching.PresentAt] > 0.5f;
                if (dom) use.Add(r);
                if (oth) other++;
                if (dom && oth) both.Add(r);
            }
            if (use.Count == 0) use.AddRange(rows);
            var k = new KeyPose();
            MedianInto(f, use, Matching.FlexionAt, k.Flexion);
            MedianInto(f, use, Matching.SpreadAt, k.Spread);
            MedianInto(f, use, Matching.PalmNormalAt, k.PalmNormal);
            MedianInto(f, use, Matching.DirectionAt, k.Direction);
            MedianInto(f, use, Matching.WristAt, k.Wrist);
            Normalize(k.PalmNormal);
            Normalize(k.Direction);
            k.OtherPresent = other * 2 > rows.Count;
            if (both.Count > 0) MedianInto(f, both, Inter, k.OtherRel);
            return k;
        }

        static void MedianInto(float[,] f, List<int> rows, int start, double[] into)
        {
            var column = new double[rows.Count];
            for (int c = 0; c < into.Length; c++)
            {
                for (int k = 0; k < rows.Count; k++) column[k] = f[rows[k], start + c];
                into[c] = Stats.Median(column);
            }
        }

        static void Normalize(double[] v)
        {
            double n = Math.Max(Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), Features.Eps);
            for (int i = 0; i < 3; i++) v[i] /= n;
        }
    }

    /// <summary>One prepared reference recording with its key frame. Call <see cref="Bind"/> once after loading.</summary>
    public sealed class Take
    {
        public Segment Segment;
        public int KeyFrame;
        public string Source = "studio";
        public string TakeId = "";

        public float[,] Features { get; private set; }
        public float[,] Descriptor { get; private set; }
        public KeyPose Key { get; private set; }
        public double PathLength { get; private set; }

        public Take Bind(EngineParams p)
        {
            Features = Engine.Features.Sequence(Segment);
            Descriptor = Matching.Descriptor(Features, p.Weights);
            Key = KeyPose.From(Features, new[] { KeyFrame });
            PathLength = Signs.PathLength(Segment);
            return this;
        }
    }

    /// <summary>Final per-sign tolerances, computed at pack time (signs.make_tolerances). The headset only
    /// multiplies them by the user's strictness. A tolerance of 0 switches that check off.</summary>
    public sealed class Tolerances
    {
        public float[] Finger = new float[5];
        public bool FingerMean;
        public float Spread, OrientationDeg, Location, Movement, OtherHand, DistanceZ;
        public bool LookalikeGuard;
    }

    public sealed class SignRef
    {
        public string Id;
        public string Tier;  // scored | lenient | watch
        public List<Take> Takes = new List<Take>();
        public Tolerances Tol = new Tolerances();
        public float Tau = 1f;
        public List<string> Lookalikes = new List<string>();
        public int ShadowTake;

        public SignRef Bind(EngineParams p)
        {
            foreach (var t in Takes) t.Bind(p);
            return this;
        }

        /// <summary>Distance to the nearest take and its index (the first take wins ties).</summary>
        public float Closest(float[,] descriptor, int band, out int take)
        {
            float best = float.PositiveInfinity;
            take = -1;
            for (int k = 0; k < Takes.Count; k++)
            {
                float d = Matching.Dtw(descriptor, Takes[k].Descriptor, band);
                if (d < best)
                {
                    best = d;
                    take = k;
                }
            }
            return best;
        }
    }

    public static class Signs
    {
        /// <summary>Distance the dominant wrist travels after a 5-frame moving average ('valid'), so jitter adds
        /// little. Only used to word movement hints (bigger / smaller).</summary>
        public static double PathLength(Segment seg, int smooth = 5)
        {
            int n = seg.Length;
            var pts = new List<double[]>();
            if (n >= smooth)
            {
                for (int t = 0; t + smooth <= n; t++)
                {
                    var s = new double[3];
                    for (int m = t; m < t + smooth; m++)
                    {
                        var w = seg.WristPos[m, Joints.Dominant];
                        s[0] += w.X; s[1] += w.Y; s[2] += w.Z;
                    }
                    pts.Add(new[] { s[0] / smooth, s[1] / smooth, s[2] / smooth });
                }
            }
            else
            {
                for (int t = 0; t < n; t++)
                {
                    var w = seg.WristPos[t, Joints.Dominant];
                    pts.Add(new double[] { w.X, w.Y, w.Z });
                }
            }
            double length = 0;
            for (int k = 1; k < pts.Count; k++)
            {
                double dx = pts[k][0] - pts[k - 1][0], dy = pts[k][1] - pts[k - 1][1], dz = pts[k][2] - pts[k - 1][2];
                length += Math.Sqrt(dx * dx + dy * dy + dz * dz);
            }
            return length;
        }
    }
}

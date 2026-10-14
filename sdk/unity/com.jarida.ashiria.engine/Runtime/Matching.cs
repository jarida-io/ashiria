using System;
using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>Reference matching for Verify and Recognise (v1 engine). Port of matching.py.</summary>
    public static class Matching
    {
        // Feature slices inside one 97-float hand block, in descriptor order (matching.py SLICES).
        public const int FlexionAt = 60, SpreadAt = 75, TipsAt = 79, PalmNormalAt = 84, DirectionAt = 87, WristAt = 90,
                         PresentAt = 96;
        static readonly (string name, int start, int length)[] Parts =
        {
            ("flexion", FlexionAt, 15), ("spread", SpreadAt, 4), ("tips", TipsAt, 5), ("palm_normal", PalmNormalAt, 3),
            ("direction", DirectionAt, 3), ("wrist", WristAt, 3), ("present", PresentAt, 1),
        };
        static readonly DescriptorWeights DefaultWeights = new DescriptorWeights();

        public const int PerHand = 34;
        public const int Dim = 2 * PerHand;

        /// <summary>[frames, 197] features → [frames, 68] weighted matching descriptor for both hands.</summary>
        public static float[,] Descriptor(float[,] features, DescriptorWeights weights = null)
        {
            weights = weights ?? DefaultWeights;
            var w = new float[Parts.Length];
            for (int p = 0; p < Parts.Length; p++) w[p] = weights[Parts[p].name];
            int n = features.GetLength(0);
            var d = new float[n, Dim];
            for (int t = 0; t < n; t++)
            {
                int k = 0;
                for (int slot = 0; slot < 2; slot++)
                {
                    int offset = slot * Features.HandDim;
                    for (int p = 0; p < Parts.Length; p++)
                        for (int i = 0; i < Parts[p].length; i++)
                            d[t, k++] = features[t, offset + Parts[p].start + i] * w[p];
                }
            }
            return d;
        }

        public static float[,] SegmentDescriptor(Segment seg, DescriptorWeights weights = null) =>
            Descriptor(Features.Sequence(seg), weights);

        static double[,] Accumulate(float[,] a, float[,] b, int band)
        {
            int n = a.GetLength(0), m = b.GetLength(0), dim = a.GetLength(1);
            band = Math.Max(band, Math.Abs(n - m));
            var acc = new double[n + 1, m + 1];
            for (int i = 0; i <= n; i++)
                for (int j = 0; j <= m; j++)
                    acc[i, j] = double.PositiveInfinity;
            acc[0, 0] = 0.0;
            for (int i = 1; i <= n; i++)
            {
                for (int j = Math.Max(1, i - band); j <= Math.Min(m, i + band); j++)
                {
                    double sum = 0.0;
                    for (int c = 0; c < dim; c++)
                    {
                        double diff = (double)a[i - 1, c] - b[j - 1, c];
                        sum += diff * diff;
                    }
                    double best = Math.Min(acc[i - 1, j], Math.Min(acc[i, j - 1], acc[i - 1, j - 1]));
                    acc[i, j] = Math.Sqrt(sum) + best;
                }
            }
            return acc;
        }

        /// <summary>Dynamic time warping distance with a Sakoe-Chiba band, normalised by n + m.</summary>
        public static float Dtw(float[,] a, float[,] b, int band = 8)
        {
            int n = a.GetLength(0), m = b.GetLength(0);
            return (float)(Accumulate(a, b, band)[n, m] / (n + m));
        }

        /// <summary>DTW distance and the alignment from (0, 0) to (n-1, m-1). Backtracking prefers the diagonal,
        /// then (i-1, j), then (i, j-1) on ties, exactly as matching.dtw_path.</summary>
        public static float DtwPath(float[,] a, float[,] b, int band, List<(int i, int j)> path)
        {
            int n = a.GetLength(0), m = b.GetLength(0);
            var acc = Accumulate(a, b, band);
            path.Clear();
            int i = n, j = m;
            path.Add((i - 1, j - 1));
            while (i != 1 || j != 1)
            {
                double diag = acc[i - 1, j - 1], up = acc[i - 1, j], left = acc[i, j - 1];
                if (diag <= up && diag <= left) { i--; j--; }
                else if (up <= left) i--;
                else j--;
                path.Add((i - 1, j - 1));
            }
            path.Reverse();
            return (float)(acc[n, m] / (n + m));
        }

        /// <summary>Per finger (thumb..pinky), the largest bend difference in radians, signed:
        /// positive means "bend more". Inputs are 15 bend angles each (Features.Flexion).</summary>
        public static float[] FingerErrors(float[] attempt15, float[] reference15)
        {
            var errors = new float[5];
            for (int finger = 0; finger < 5; finger++)
            {
                float worst = 0f;
                for (int j = 0; j < 3; j++)
                {
                    float diff = reference15[finger * 3 + j] - attempt15[finger * 3 + j];
                    if (MathF.Abs(diff) > MathF.Abs(worst)) worst = diff;
                }
                errors[finger] = worst;
            }
            return errors;
        }
    }
}

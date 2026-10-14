using System;

namespace Ashiria.Engine
{
    /// <summary>Reference matching for Verify and Recognise (v1 engine). Port of matching.py.</summary>
    public static class Matching
    {
        // (start, length, weight) inside one 97-float hand block; order defines the descriptor layout.
        // Must match matching.py BLOCK (provisional weights, tuned on simulation; re-tune on Quest attempts).
        static readonly (int start, int length, float weight)[] Parts =
        {
            (60, 15, 1.0f),  // finger bends
            (75, 4, 1.0f),   // spreads
            (79, 5, 0.5f),   // fingertip distances
            (84, 3, 1.0f),   // palm normal
            (87, 3, 1.0f),   // hand direction
            (90, 3, 3.0f),   // wrist position
            (96, 1, 1.0f),   // present
        };

        public const int PerHand = 34;
        public const int Dim = 2 * PerHand;

        /// <summary>[frames, 197] features → [frames, 68] weighted matching descriptor for both hands.</summary>
        public static float[,] Descriptor(float[,] features)
        {
            int n = features.GetLength(0);
            var d = new float[n, Dim];
            for (int t = 0; t < n; t++)
            {
                int k = 0;
                for (int slot = 0; slot < 2; slot++)
                {
                    int offset = slot * Features.HandDim;
                    foreach (var (start, length, weight) in Parts)
                        for (int i = 0; i < length; i++)
                            d[t, k++] = features[t, offset + start + i] * weight;
                }
            }
            return d;
        }

        /// <summary>Dynamic time warping distance with a Sakoe-Chiba band, normalised by n + m.</summary>
        public static float Dtw(float[,] a, float[,] b, int band = 8)
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
                        double diff = a[i - 1, c] - b[j - 1, c];
                        sum += diff * diff;
                    }
                    double best = Math.Min(acc[i - 1, j], Math.Min(acc[i, j - 1], acc[i - 1, j - 1]));
                    acc[i, j] = Math.Sqrt(sum) + best;
                }
            }
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

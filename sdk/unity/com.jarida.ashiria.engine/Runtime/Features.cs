using System;

namespace Ashiria.Engine
{
    /// <summary>Per-frame features (spec §6, 197 floats per frame). Port of features.py.</summary>
    public static class Features
    {
        public const int SpecVersion = 1;
        public const float Eps = 1e-6f;
        public const int HandDim = 97;
        public const int Dim = 2 * HandDim + 3;

        static readonly int[,] TipPairs = { { 4, 8 }, { 4, 12 }, { 4, 16 }, { 4, 20 }, { 8, 12 } };

        /// <summary>Features for a resampled canonical segment: [frames, 197]. Absent hands are all zeros.</summary>
        public static float[,] Sequence(Segment seg)
        {
            int n = seg.Length;
            float dt = 1f / seg.Fps;
            var f = new float[n, Dim];
            var hand = new Vec3[Joints.Count];
            for (int slot = 0; slot < 2; slot++)
            {
                for (int t = 0; t < n; t++)
                {
                    if (seg.Present[t, slot] <= 0.5f) continue;
                    for (int j = 0; j < Joints.Count; j++)
                        hand[j] = seg.HandRel[t, slot, j];
                    bool steady = t > 0 && seg.Present[t - 1, slot] > 0.5f;
                    var velocity = steady ? (seg.WristPos[t, slot] - seg.WristPos[t - 1, slot]) / dt : Vec3.Zero;
                    HandBlock(hand, seg.WristPos[t, slot], velocity, f, t, slot * HandDim);
                }
            }
            for (int t = 0; t < n; t++)
            {
                if (seg.Present[t, 0] <= 0.5f || seg.Present[t, 1] <= 0.5f) continue;
                var gap = seg.WristPos[t, 1] - seg.WristPos[t, 0];
                f[t, 2 * HandDim] = gap.X;
                f[t, 2 * HandDim + 1] = gap.Y;
                f[t, 2 * HandDim + 2] = gap.Z;
            }
            return f;
        }

        /// <summary>Hand axes: y = wrist to middle knuckle, z = palm normal, x = y × z; plus the palm length.</summary>
        public static void HandAxes(Vec3[] h, out Vec3 x, out Vec3 y, out Vec3 z, out float palm)
        {
            var toMiddle = h[Joints.MiddleMcp] - h[Joints.Wrist];
            palm = toMiddle.Length;
            y = toMiddle.Normalized();
            var normal = Vec3.Cross(h[Joints.IndexMcp] - h[Joints.Wrist], h[Joints.PinkyMcp] - h[Joints.Wrist]);
            x = Vec3.Cross(y, normal).Normalized();
            z = Vec3.Cross(x, y);
        }

        /// <summary>15 bend angles, three per finger, thumb first (0 = straight).</summary>
        public static void Flexion(Vec3[] h, float[] out15)
        {
            int k = 0;
            foreach (var chain in Joints.Fingers)
            {
                for (int m = 1; m <= 3; m++)
                {
                    int prev = m == 1 ? Joints.Wrist : chain[m - 2];
                    int at = chain[m - 1];
                    int next = chain[m];
                    out15[k++] = Vec3.Angle(h[at] - h[prev], h[next] - h[at]);
                }
            }
        }

        static void HandBlock(Vec3[] h, Vec3 wrist, Vec3 velocity, float[,] f, int t, int o)
        {
            HandAxes(h, out var x, out var y, out var z, out var palm);
            float inv = 1f / MathF.Max(palm, Eps);
            int k = o;
            for (int j = 1; j < Joints.Count; j++)
            {
                var p = h[j];
                f[t, k++] = Vec3.Dot(x, p) * inv;
                f[t, k++] = Vec3.Dot(y, p) * inv;
                f[t, k++] = Vec3.Dot(z, p) * inv;
            }

            var bends = new float[15];
            Flexion(h, bends);
            foreach (var b in bends) f[t, k++] = b;

            for (int i = 0; i < 4; i++)
            {
                var c0 = Joints.Fingers[i];
                var c1 = Joints.Fingers[i + 1];
                f[t, k++] = Vec3.Angle(h[c0[1]] - h[c0[0]], h[c1[1]] - h[c1[0]]);
            }

            for (int i = 0; i < TipPairs.GetLength(0); i++)
                f[t, k++] = (h[TipPairs[i, 0]] - h[TipPairs[i, 1]]).Length * inv;

            f[t, k++] = z.X; f[t, k++] = z.Y; f[t, k++] = z.Z;
            f[t, k++] = y.X; f[t, k++] = y.Y; f[t, k++] = y.Z;
            f[t, k++] = wrist.X; f[t, k++] = wrist.Y; f[t, k++] = wrist.Z;
            f[t, k++] = velocity.X; f[t, k++] = velocity.Y; f[t, k++] = velocity.Z;
            f[t, k++] = 1f;
            if (k != o + HandDim) throw new InvalidOperationException("hand block size mismatch");
        }
    }
}

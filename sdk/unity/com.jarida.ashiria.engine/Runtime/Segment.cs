using System;

namespace Ashiria.Engine
{
    /// <summary>
    /// A canonical sign segment (spec §4): for every frame and hand slot, the 21 joints relative to the
    /// wrist in reference-frame axes (metres), the wrist position (shoulder widths from the head) and
    /// whether the hand is present (0 or 1).
    /// </summary>
    public sealed class Segment
    {
        public readonly Vec3[,,] HandRel;  // [frame, slot, joint]
        public readonly Vec3[,] WristPos;  // [frame, slot]
        public readonly float[,] Present;  // [frame, slot]
        public readonly float Fps;

        public int Length => Present.GetLength(0);

        public Segment(Vec3[,,] handRel, Vec3[,] wristPos, float[,] present, float fps)
        {
            int n = present.GetLength(0);
            if (handRel.GetLength(0) != n || wristPos.GetLength(0) != n || present.GetLength(1) != 2 ||
                handRel.GetLength(1) != 2 || handRel.GetLength(2) != Joints.Count || wristPos.GetLength(1) != 2)
                throw new ArgumentException("Segment arrays must be [frames, 2, 21], [frames, 2] and [frames, 2].");
            HandRel = handRel;
            WristPos = wristPos;
            Present = present;
            Fps = fps;
        }

        public static Segment Empty(int frames, float fps) =>
            new Segment(new Vec3[frames, 2, Joints.Count], new Vec3[frames, 2], new float[frames, 2], fps);

        public Segment Slice(int start, int end)
        {
            var s = Empty(end - start, Fps);
            for (int t = start; t < end; t++)
                CopyFrame(this, t, s, t - start);
            return s;
        }

        public Segment WithFps(float fps)
        {
            var s = Empty(Length, fps);
            for (int t = 0; t < Length; t++)
                CopyFrame(this, t, s, t);
            return s;
        }

        internal static void CopyFrame(Segment from, int t, Segment to, int u)
        {
            for (int slot = 0; slot < 2; slot++)
            {
                for (int j = 0; j < Joints.Count; j++)
                    to.HandRel[u, slot, j] = from.HandRel[t, slot, j];
                to.WristPos[u, slot] = from.WristPos[t, slot];
                to.Present[u, slot] = from.Present[t, slot];
            }
        }
    }
}

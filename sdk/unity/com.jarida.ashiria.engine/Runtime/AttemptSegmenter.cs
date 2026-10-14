using System;
using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>One attempt cut from the live stream: raw canonical frames at the tick rate (not yet prepared).</summary>
    public sealed class Attempt
    {
        public Segment Segment;
        public int Start, End;  // ticks of the first and last frame (inclusive)
        public string Reason;   // hold | drop | timeout
    }

    /// <summary>
    /// Cuts the live stream of canonical frames into attempts (port of segmenter.py). Feed exactly one frame per
    /// fixed tick (30 Hz). idle → active after the hand is up (wrist above the user's rest height) for StartS,
    /// keeping PreRollS before; an attempt ends when the hand is down or lost for DropS ("drop"), the dominant
    /// hand is still for HoldS ("hold"), or after MaxAttemptS ("timeout"). After a hold or timeout, a cooldown
    /// waits for the hand to drop below rest or to move RearmMove away other than downwards.
    /// </summary>
    public sealed class AttemptSegmenter
    {
        sealed class Frame
        {
            public int Tick;
            public Vec3[] HandRel = new Vec3[2 * Joints.Count];
            public Vec3[] Wrist = new Vec3[2];
            public float[] Present = new float[2];
            public bool Up;
            public float[] Bends;  // dominant hand, null if absent
        }

        const int Dom = Joints.Dominant;
        readonly SegmenterParams p;
        readonly int startN, preN, dropN, holdN, holdAllow, minN, maxN;
        readonly Queue<Frame> pre = new Queue<Frame>();
        List<Frame> buf = new List<Frame>();
        int upRun, downRun;
        bool hasHeld;
        Vec3 heldAt;

        public int Tick { get; private set; }
        public string State { get; private set; }

        public AttemptSegmenter(SegmenterParams p)
        {
            this.p = p;
            startN = p.Frames(p.StartS);
            preN = p.Frames(p.PreRollS);
            dropN = p.Frames(p.DropS);
            holdN = p.Frames(p.HoldS);
            holdAllow = holdN / 10;
            minN = p.Frames(p.MinAttemptS);
            maxN = p.Frames(p.MaxAttemptS);
            Reset();
        }

        public void Reset()
        {
            Tick = -1;
            State = "idle";
            pre.Clear();
            buf = new List<Frame>();
            upRun = downRun = 0;
            hasHeld = false;
        }

        /// <summary>One canonical frame: handRel[slot, joint] (2×21), wrist[2], present[2]. Returns the attempt
        /// when one ends, else null. The arrays are copied.</summary>
        public Attempt Push(Vec3[,] handRel, Vec3[] wrist, float[] present)
        {
            Tick++;
            var f = new Frame { Tick = Tick };
            for (int s = 0; s < 2; s++)
            {
                for (int j = 0; j < Joints.Count; j++) f.HandRel[s * Joints.Count + j] = handRel[s, j];
                f.Wrist[s] = wrist[s];
                f.Present[s] = present[s];
                if (present[s] > 0.5f && wrist[s].Y >= p.RestHeight) f.Up = true;
            }
            if (present[Dom] > 0.5f)
            {
                var hand = new Vec3[Joints.Count];
                Array.Copy(f.HandRel, 0, hand, 0, Joints.Count);
                f.Bends = new float[15];
                Features.Flexion(hand, f.Bends);
            }

            if (State == "cooldown")
            {
                downRun = f.Up ? 0 : downRun + 1;
                bool moved = false;
                if (present[Dom] > 0.5f && hasHeld)
                {
                    var d = wrist[Dom] - heldAt;
                    // Moving on to a new sign re-arms; lowering the hand (on its way to rest) does not.
                    moved = Norm(d) > p.RearmMove && d.Y > -p.RearmMove;
                }
                if (downRun < dropN && !moved) return null;
                ToIdle();
            }

            if (State == "idle")
            {
                pre.Enqueue(f);
                while (pre.Count > preN + startN) pre.Dequeue();
                upRun = f.Up ? upRun + 1 : 0;
                if (upRun >= startN)
                {
                    State = "active";
                    buf = new List<Frame>(pre);
                    downRun = 0;
                }
                return null;
            }

            buf.Add(f);
            downRun = f.Up ? 0 : downRun + 1;
            if (downRun >= dropN) return Finish("drop");
            if (Held()) return Finish("hold");
            if (buf.Count >= maxN) return Finish("timeout");
            return null;
        }

        void ToIdle()
        {
            State = "idle";
            upRun = downRun = 0;
            hasHeld = false;
            pre.Clear();
        }

        static double Norm(Vec3 v) => Math.Sqrt((double)v.X * v.X + (double)v.Y * v.Y + (double)v.Z * v.Z);

        bool Held()
        {
            if (buf.Count < holdN) return false;
            var window = new List<Frame>(holdN);
            for (int k = buf.Count - holdN; k < buf.Count; k++)
                if (buf[k].Bends != null) window.Add(buf[k]);
            if (window.Count < holdN - holdAllow) return false;
            int n = window.Count;
            var centre = new double[3];
            var column = new double[n];
            for (int c = 0; c < 3; c++)
            {
                for (int k = 0; k < n; k++) column[k] = Component(window[k].Wrist[Dom], c);
                centre[c] = Stats.Median(column);
            }
            var bendMedian = new double[15];
            for (int b = 0; b < 15; b++)
            {
                for (int k = 0; k < n; k++) column[k] = window[k].Bends[b];
                bendMedian[b] = Stats.Median(column);
            }
            int still = 0;
            foreach (var f in window)
            {
                double dx = f.Wrist[Dom].X - centre[0], dy = f.Wrist[Dom].Y - centre[1], dz = f.Wrist[Dom].Z - centre[2];
                double dev = 0;
                for (int b = 0; b < 15; b++) dev += Math.Abs(f.Bends[b] - bendMedian[b]);
                if (Math.Sqrt(dx * dx + dy * dy + dz * dz) < p.HoldRadius && dev / 15 < p.HoldShape) still++;
            }
            return still >= holdN - holdAllow;
        }

        static double Component(Vec3 v, int c) => c == 0 ? v.X : c == 1 ? v.Y : v.Z;

        Attempt Finish(string reason)
        {
            var frames = buf;
            buf = new List<Frame>();
            if (reason == "drop")
            {
                ToIdle();
            }
            else
            {
                State = "cooldown";
                downRun = 0;
                var dom = new List<Vec3>();
                for (int k = Math.Max(0, frames.Count - holdN); k < frames.Count; k++)
                    if (frames[k].Present[Dom] > 0.5f) dom.Add(frames[k].Wrist[Dom]);
                hasHeld = dom.Count > 0;
                if (hasHeld)
                {
                    var column = new double[dom.Count];
                    var c = new float[3];
                    for (int axis = 0; axis < 3; axis++)
                    {
                        for (int k = 0; k < dom.Count; k++) column[k] = Component(dom[k], axis);
                        c[axis] = (float)Stats.Median(column);
                    }
                    heldAt = new Vec3(c[0], c[1], c[2]);
                }
            }
            int up = 0;
            foreach (var f in frames) if (f.Up) up++;
            if (up < minN) return null;

            var seg = Segment.Empty(frames.Count, p.TickHz);
            for (int t = 0; t < frames.Count; t++)
                for (int s = 0; s < 2; s++)
                {
                    for (int j = 0; j < Joints.Count; j++) seg.HandRel[t, s, j] = frames[t].HandRel[s * Joints.Count + j];
                    seg.WristPos[t, s] = frames[t].Wrist[s];
                    seg.Present[t, s] = frames[t].Present[s];
                }
            return new Attempt { Segment = seg, Start = frames[0].Tick, End = frames[frames.Count - 1].Tick, Reason = reason };
        }
    }

    static class Stats
    {
        /// <summary>numpy's median: the middle value, or the mean of the two middle values.</summary>
        public static double Median(double[] values)
        {
            var sorted = (double[])values.Clone();
            Array.Sort(sorted);
            int m = sorted.Length / 2;
            return sorted.Length % 2 == 1 ? sorted[m] : 0.5 * (sorted[m - 1] + sorted[m]);
        }
    }
}

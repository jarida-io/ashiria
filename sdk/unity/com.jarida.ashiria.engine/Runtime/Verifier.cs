using System;
using System.Collections.Generic;
using System.Linq;

namespace Ashiria.Engine
{
    /// <summary>A feedback hint as a code; the app maps codes to words. Directions are canonical
    /// (dominant_side / other_side / up / down / forward / back), so they hold for left-handed users too.</summary>
    public sealed class Hint
    {
        public string Code;
        public double Severity;
        public string Finger, Direction, SignId;

        public Hint(string code, double severity, string finger = null, string direction = null, string signId = null)
        {
            Code = code;
            Severity = severity;
            Finger = finger;
            Direction = direction;
            SignId = signId;
        }
    }

    public sealed class VerifyResult
    {
        public string Status;  // pass | fail | unscored | no_hands
        public double Score;   // 0..1; 0.5 is the pass line
        public double Distance = double.PositiveInfinity;
        public int Take = -1;
        public Dictionary<string, double> Components = new Dictionary<string, double>();  // error / tolerance; <= 1 passes
        public double[] FingerErrors = new double[0];  // signed radians, + means bend more
        public List<Hint> Hints = new List<Hint>();
        public bool Passed => Status == "pass";
    }

    /// <summary>Verify: "was that sign X?" (port of verify.py). The attempt must be prepared (Pipeline.Prepare).</summary>
    public static class Verifier
    {
        public const int KeySpan = 2;
        public static readonly string[] FingerNames = { "thumb", "index", "middle", "ring", "pinky" };
        static readonly string[,] Axes = { { "dominant_side", "other_side" }, { "up", "down" }, { "forward", "back" } };
        static readonly double[] LocationWeights = { 1.0, 1.0, 0.5 };
        const double Eps = 1e-6;

        public static string DirectionName(double[] v, double[] weights = null)
        {
            int axis = 0;
            double best = -1;
            for (int i = 0; i < 3; i++)
            {
                double w = Math.Abs(v[i]) * (weights == null ? 1.0 : weights[i]);
                if (w > best) { best = w; axis = i; }
            }
            return Axes[axis, v[axis] >= 0 ? 0 : 1];
        }

        static double AngleDeg(double[] a, double[] b)
        {
            double cx = a[1] * b[2] - a[2] * b[1], cy = a[2] * b[0] - a[0] * b[2], cz = a[0] * b[1] - a[1] * b[0];
            double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            return Math.Atan2(Math.Sqrt(cx * cx + cy * cy + cz * cz), dot) * 180.0 / Math.PI;
        }

        public static VerifyResult Verify(Segment attempt, SignRef sign, EngineParams p, double toleranceScale = 1.0,
                                          IReadOnlyList<SignRef> lookalikes = null)
        {
            if (sign.Tier == "watch") return new VerifyResult { Status = "unscored" };
            int n = attempt.Length;
            int present = 0;
            for (int t = 0; t < n; t++) if (attempt.Present[t, Joints.Dominant] > 0.5f) present++;
            if (present < p.Verify.MinPresent * n)
                return new VerifyResult { Status = "no_hands", Hints = { new Hint("tracking.lost", 1.0) } };

            var feats = Features.Sequence(attempt);
            var desc = Matching.Descriptor(feats, p.Weights);
            var path = new List<(int i, int j)>();
            var bestPath = new List<(int i, int j)>();
            double d = double.PositiveInfinity;
            int k = -1;
            for (int t = 0; t < sign.Takes.Count; t++)
            {
                double dt = Matching.DtwPath(desc, sign.Takes[t].Descriptor, p.DtwBand, path);
                if (k < 0 || dt < d)
                {
                    d = dt;
                    k = t;
                    bestPath = new List<(int i, int j)>(path);
                }
            }
            var take = sign.Takes[k];
            int first = int.MaxValue, last = -1;
            foreach (var (i, j) in bestPath)
                if (j == take.KeyFrame) { first = Math.Min(first, i); last = Math.Max(last, i); }
            var rows = new List<int>();
            for (int r = Math.Max(0, first - KeySpan); r < Math.Min(n, last + KeySpan + 1); r++) rows.Add(r);
            var pose = KeyPose.From(feats, rows);
            var refPose = take.Key;
            var tol = sign.Tol;
            double s = toleranceScale;
            var comps = new Dictionary<string, double>();
            var hints = new List<Hint>();

            var err = new double[5];
            var ratio = new double[5];
            for (int f = 0; f < 5; f++)
            {
                double worst = 0;
                for (int j = 0; j < 3; j++)
                {
                    double diff = refPose.Flexion[f * 3 + j] - pose.Flexion[f * 3 + j];
                    if (Math.Abs(diff) > Math.Abs(worst)) worst = diff;
                }
                err[f] = worst;
                ratio[f] = Math.Abs(worst) / (tol.Finger[f] * s);
            }
            comps["shape"] = tol.FingerMean ? ratio.Average() : ratio.Max();
            if (comps["shape"] > 1)
            {
                var order = Enumerable.Range(0, 5).OrderByDescending(f => ratio[f]).Where(f => ratio[f] > 1).ToList();
                if (order.Count == 0) order.Add(Array.IndexOf(ratio, ratio.Max()));
                foreach (int f in order)
                    hints.Add(new Hint(err[f] > 0 ? "finger.bend_more" : "finger.straighten", ratio[f], finger: FingerNames[f]));
            }

            if (tol.Spread > 0)
            {
                int worst = 0;
                for (int i = 1; i < 4; i++)
                    if (Math.Abs(refPose.Spread[i] - pose.Spread[i]) > Math.Abs(refPose.Spread[worst] - pose.Spread[worst])) worst = i;
                double ds = refPose.Spread[worst] - pose.Spread[worst];
                comps["spread"] = Math.Abs(ds) / (tol.Spread * s);
                // Bending a finger at the knuckle also changes the spread angles: only hint at spread if the shape is right.
                if (comps["spread"] > 1 && comps["shape"] <= 1)
                    hints.Add(new Hint(ds > 0 ? "spread.more" : "spread.less", comps["spread"]));
            }

            double palm = AngleDeg(pose.PalmNormal, refPose.PalmNormal);
            double point = AngleDeg(pose.Direction, refPose.Direction);
            comps["orientation"] = Math.Max(palm, point) / (tol.OrientationDeg * s);
            if (comps["orientation"] > 1)
                hints.Add(palm >= point
                    ? new Hint("palm.face", comps["orientation"], direction: DirectionName(refPose.PalmNormal))
                    : new Hint("hand.point", comps["orientation"], direction: DirectionName(refPose.Direction)));

            var dw = new double[3];
            double loc = 0;
            for (int i = 0; i < 3; i++)
            {
                dw[i] = refPose.Wrist[i] - pose.Wrist[i];
                loc += Math.Pow(dw[i] * LocationWeights[i], 2);
            }
            comps["location"] = Math.Sqrt(loc) / (tol.Location * s);
            if (comps["location"] > 1)
                hints.Add(new Hint("move", comps["location"], direction: DirectionName(dw, LocationWeights)));

            if (tol.Movement > 0)
            {
                var track = WristTrack(attempt, pose.Wrist);
                var refTrack = WristTrack(take.Segment, refPose.Wrist);
                comps["movement"] = 2 * Matching.Dtw(track, refTrack, p.DtwBand) / (tol.Movement * s);
                if (comps["movement"] > 1)
                {
                    double size = Signs.PathLength(attempt) / Math.Max(take.PathLength, Eps);
                    string code = size < 0.6 ? "movement.bigger" : size > 1.6 ? "movement.smaller" : "movement.path";
                    hints.Add(new Hint(code, comps["movement"]));
                }
            }

            if (tol.OtherHand > 0 && refPose.OtherPresent)
            {
                if (!pose.OtherPresent)
                {
                    comps["other_hand"] = 2.0;
                    hints.Add(new Hint("other_hand.missing", 2.0));
                }
                else
                {
                    var diff = new double[3];
                    double dist = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        diff[i] = refPose.OtherRel[i] - pose.OtherRel[i];
                        dist += diff[i] * diff[i];
                    }
                    comps["other_hand"] = Math.Sqrt(dist) / (tol.OtherHand * s);
                    if (comps["other_hand"] > 1)
                        hints.Add(new Hint("other_hand.place", comps["other_hand"], direction: DirectionName(diff)));
                }
            }

            comps["overall"] = d / Math.Max(sign.Tau * tol.DistanceZ * s, Eps);
            if (comps["overall"] > 1) hints.Add(new Hint("overall.different", comps["overall"]));

            if (tol.LookalikeGuard && lookalikes != null && lookalikes.Count > 0)
            {
                double e = double.NegativeInfinity;
                string other = null;
                foreach (var l in lookalikes)  // the first look-alike with the largest error
                {
                    double el = p.Verify.LookalikeRatio * d / Math.Max(l.Closest(desc, p.DtwBand, out _), Eps) / s;
                    if (el > e) { e = el; other = l.Id; }
                }
                comps["lookalike"] = e;
                if (e > 1) hints.Add(new Hint("lookalike", e, signId: other));
            }

            double worstE = comps.Values.Max();
            return new VerifyResult
            {
                Status = worstE <= 1 ? "pass" : "fail",
                Score = Math.Min(1, Math.Max(0, 1 - worstE / 2)),
                Distance = d,
                Take = k,
                Components = comps,
                FingerErrors = err,
                Hints = hints.OrderByDescending(h => h.Severity).ToList(),  // stable, like Python's sort
            };
        }

        static float[,] WristTrack(Segment seg, double[] centre)
        {
            var track = new float[seg.Length, 3];
            for (int t = 0; t < seg.Length; t++)
            {
                var w = seg.WristPos[t, Joints.Dominant];
                track[t, 0] = (float)(w.X - centre[0]);
                track[t, 1] = (float)(w.Y - centre[1]);
                track[t, 2] = (float)(w.Z - centre[2]);
            }
            return track;
        }
    }
}

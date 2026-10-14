using System;
using System.Collections.Generic;
using System.Linq;

namespace Ashiria.Engine
{
    public sealed class RecogniseResult
    {
        public string Status;  // match | ambiguous | not_sure | no_hands
        public string Best, RunnerUp;
        public double BestZ = double.PositiveInfinity, RunnerUpZ = double.PositiveInfinity, Margin = double.PositiveInfinity;
        public double Confidence;
        public List<(string id, double distance, double z)> Ranking = new List<(string, double, double)>();
    }

    /// <summary>Recognise: "which sign was that?" for Free Sign (port of recognise.py). Each candidate's distance is
    /// divided by its tau, so z ≈ 1 is as close as a normal correct attempt; an absolute gate on z says "not sure"
    /// to anything unlike every candidate. The attempt must be prepared.</summary>
    public static class Recogniser
    {
        public static RecogniseResult Recognise(Segment attempt, IReadOnlyList<SignRef> candidates, EngineParams p)
        {
            var rp = p.Recognise;
            int present = 0;
            for (int t = 0; t < attempt.Length; t++) if (attempt.Present[t, Joints.Dominant] > 0.5f) present++;
            if (candidates.Count == 0 || present < rp.MinPresent * attempt.Length)
                return new RecogniseResult { Status = "no_hands" };

            var desc = Matching.SegmentDescriptor(attempt, p.Weights);
            var dist = new double[candidates.Count];
            var z = new double[candidates.Count];
            for (int c = 0; c < candidates.Count; c++)
            {
                dist[c] = candidates[c].Closest(desc, p.DtwBand, out _);
                z[c] = dist[c] / candidates[c].Tau;
            }
            var order = Enumerable.Range(0, candidates.Count).OrderBy(c => z[c]).ToList();  // stable: ties keep order
            double maxLogit = order.Select(c => -z[c] / rp.Temperature).Max();
            double sum = 0;
            for (int c = 0; c < z.Length; c++) sum += Math.Exp(-z[c] / rp.Temperature - maxLogit);
            int b = order[0];
            var r = new RecogniseResult
            {
                Status = "match",
                Best = candidates[b].Id,
                BestZ = z[b],
                Confidence = Math.Exp(-z[b] / rp.Temperature - maxLogit) / sum,
            };
            foreach (int c in order.Take(rp.Top)) r.Ranking.Add((candidates[c].Id, dist[c], z[c]));
            if (order.Count > 1)
            {
                int ru = order[1];
                r.RunnerUp = candidates[ru].Id;
                r.RunnerUpZ = z[ru];
                r.Margin = r.RunnerUpZ - r.BestZ;
            }
            if (r.BestZ > rp.AcceptZ) r.Status = "not_sure";
            else if (r.Margin < rp.MinMarginZ && r.RunnerUpZ <= rp.AcceptZ) r.Status = "ambiguous";
            return r;
        }
    }
}

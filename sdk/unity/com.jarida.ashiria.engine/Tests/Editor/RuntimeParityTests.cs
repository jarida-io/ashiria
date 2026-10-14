using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ashiria.Engine.Content;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace Ashiria.Engine.Tests
{
    /// <summary>
    /// The C# runtime (prepare, Quest frames, segmenter, Verify, Recognise, pack loading) must reproduce the Python
    /// reference on spec/golden/runtime_v1.json and the synthetic mini pack spec/golden/pack_v1 (`ashiria golden`).
    /// </summary>
    public class RuntimeParityTests
    {
        JObject golden;
        ContentPack pack;

        [OneTimeSetUp]
        public void Load()
        {
            string file = EngineParityTests.GoldenFile("runtime_v1.json");
            Assert.That(File.Exists(file), $"Golden vectors not found at {file}. Run `ashiria golden` in training/.");
            golden = JObject.Parse(File.ReadAllText(file));
            string dir = EngineParityTests.GoldenFile((string)golden["pack"]);
            pack = ContentPack.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")),
                                     File.ReadAllText(Path.Combine(dir, "catalogue.json")),
                                     File.ReadAllText(Path.Combine(dir, "references.json")));
        }

        double Tol(string what) => (double)golden["tolerance"][what];

        [Test]
        public void PackLoadsAndPassesItsSelfCheck()
        {
            Assert.That(pack.Signs.Count, Is.EqualTo(pack.Catalogue.Count));
            Assert.That(pack.SelfCheck(out var expected, out var actual), $"self-check {actual} vs {expected}");
            Assert.That(pack.Catalogue.Select(s => s.Tier).Distinct().OrderBy(t => t), Is.EqualTo(new[] { "lenient", "scored", "watch" }));
            Assert.That(pack.Params.SeqLen, Is.EqualTo(32));
        }

        [Test]
        public void DamagedPackIsRejected()
        {
            string dir = EngineParityTests.GoldenFile((string)golden["pack"]);
            string refs = File.ReadAllText(Path.Combine(dir, "references.json")).Replace("\"tau\":", "\"tau\": ");
            Assert.Throws<System.FormatException>(() => ContentPack.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")),
                File.ReadAllText(Path.Combine(dir, "catalogue.json")), refs));
        }

        [Test]
        public void PrepareMatchesPython()
        {
            foreach (var c in golden["prepare"])
            {
                var seg = Pipeline.Prepare(EngineParityTests.ToSegment(c["segment"]), pack.Params, (float)c["rest_height"]);
                Assert.That(seg.Fps, Is.EqualTo((float)c["fps"]).Within(1e-3f), $"{c["name"]}: fps");
                var present = (JArray)c["present"];
                for (int t = 0; t < present.Count; t++)
                    for (int s = 0; s < 2; s++)
                        Assert.That(seg.Present[t, s], Is.EqualTo((float)present[t][s]), $"{c["name"]}: present[{t},{s}]");
                AssertClose((JArray)c["descriptor"], Matching.SegmentDescriptor(seg, pack.Params.Weights), Tol("descriptor"), (string)c["name"]);
            }
        }

        [Test]
        public void QuestFramesMatchPython()
        {
            foreach (var c in golden["quest"])
            {
                var axes = ((JArray)c["last_axes"]).Select(V).ToArray();
                var hands = new Dictionary<string, HandInput>();
                foreach (var side in new[] { "right", "left" })
                {
                    var h = c["hands"][side];
                    hands[side] = new HandInput
                    {
                        Joints26 = ((JArray)h["joints"]).Select(V).ToArray(),
                        Tracked = (bool)h["tracked"], HighConfidence = true,
                    };
                }
                var handRel = new Vec3[2, Joints.Count];
                var wrist = new Vec3[2];
                var present = new float[2];
                QuestCanonical.Frame(V(c["head_pos"]), V(c["head_forward"]), hands["right"], hands["left"], (bool)c["left_handed"],
                                     (float)c["shoulder_width"], ref axes, handRel, wrist, present);
                double tol = Tol("quest");
                for (int s = 0; s < 2; s++)
                {
                    Assert.That(present[s], Is.EqualTo((float)c["present"][s]));
                    AssertVec(c["wrist_pos"][s], wrist[s], tol);
                    for (int j = 0; j < Joints.Count; j++) AssertVec(c["hand_rel"][s][j], handRel[s, j], tol);
                }
            }
        }

        [Test]
        public void SegmenterMatchesPython()
        {
            foreach (var c in golden["segmenter"])
            {
                var seg = EngineParityTests.ToSegment(c["segment"]);
                var segmenter = new AttemptSegmenter(pack.Params.Segmenter);
                var found = new List<string>();
                var handRel = new Vec3[2, Joints.Count];
                var wrist = new Vec3[2];
                var present = new float[2];
                for (int t = 0; t < seg.Length; t++)
                {
                    for (int s = 0; s < 2; s++)
                    {
                        for (int j = 0; j < Joints.Count; j++) handRel[s, j] = seg.HandRel[t, s, j];
                        wrist[s] = seg.WristPos[t, s];
                        present[s] = seg.Present[t, s];
                    }
                    var a = segmenter.Push(handRel, wrist, present);
                    if (a != null) found.Add($"{a.Start}-{a.End} {a.Reason}");
                }
                var expected = c["attempts"].Select(a => $"{a["start"]}-{a["end"]} {a["reason"]}").ToList();
                Assert.That(found, Is.EqualTo(expected), (string)c["name"]);
            }
        }

        [Test]
        public void VerifyMatchesPython()
        {
            double tol = Tol("verify");
            foreach (var c in golden["verify"])
            {
                string name = (string)c["name"], id = (string)c["sign"];
                var r = Verifier.Verify(EngineParityTests.ToSegment(c["attempt"]), pack.Signs[id], pack.Params,
                                        (double)c["tolerance_scale"], pack.LookalikesOf(id));
                Assert.That(r.Status, Is.EqualTo((string)c["status"]), name);
                Assert.That(r.Take, Is.EqualTo((int)c["take"]), $"{name}: take");
                Assert.That(r.Score, Is.EqualTo((double)c["score"]).Within(tol), $"{name}: score");
                var comps = (JObject)c["components"];
                Assert.That(r.Components.Keys.OrderBy(k => k), Is.EqualTo(comps.Properties().Select(p => p.Name).OrderBy(k => k)), $"{name}: components");
                foreach (var kv in comps)
                    Assert.That(r.Components[kv.Key], Is.EqualTo((double)kv.Value).Within(tol), $"{name}: {kv.Key}");
                var fingers = (JArray)c["finger_errors"];
                for (int f = 0; f < fingers.Count; f++)
                    Assert.That(r.FingerErrors[f], Is.EqualTo((double)fingers[f]).Within(tol), $"{name}: finger {f}");
                var hints = c["hints"].Select(h => $"{h["code"]}|{h["finger"]}|{h["direction"]}|{h["sign_id"]}").ToList();
                Assert.That(r.Hints.Select(h => $"{h.Code}|{h.Finger}|{h.Direction}|{h.SignId}"), Is.EqualTo(hints), $"{name}: hints");
            }
        }

        [Test]
        public void RecogniseMatchesPython()
        {
            double tol = Tol("recognise");
            var candidates = pack.Catalogue.Where(s => s.Recognise).Select(s => pack.Signs[s.Id]).ToList();
            foreach (var c in golden["recognise"])
            {
                string name = (string)c["name"];
                var p = pack.Params;
                var saved = (p.Recognise.AcceptZ, p.Recognise.MinMarginZ);
                p.Recognise.AcceptZ = (float)c["accept_z"];
                p.Recognise.MinMarginZ = (float)c["min_margin_z"];
                var r = Recogniser.Recognise(EngineParityTests.ToSegment(c["attempt"]), candidates, p);
                (p.Recognise.AcceptZ, p.Recognise.MinMarginZ) = saved;
                Assert.That(r.Status, Is.EqualTo((string)c["status"]), name);
                Assert.That(r.Best, Is.EqualTo((string)c["best"]), $"{name}: best");
                Assert.That(r.RunnerUp, Is.EqualTo((string)c["runner_up"]), $"{name}: runner-up");
                Assert.That(r.BestZ, Is.EqualTo((double)c["best_z"]).Within(tol), $"{name}: best z");
                Assert.That(r.RunnerUpZ, Is.EqualTo((double)c["runner_up_z"]).Within(tol), $"{name}: runner-up z");
                Assert.That(r.Confidence, Is.EqualTo((double)c["confidence"]).Within(tol), $"{name}: confidence");
                Assert.That(r.Ranking.Select(x => x.id), Is.EqualTo(c["ranking"].Select(x => (string)x[0])), $"{name}: ranking");
            }
        }

        static Vec3 V(JToken t) => new Vec3((float)t[0], (float)t[1], (float)t[2]);

        static void AssertVec(JToken expected, Vec3 actual, double tol)
        {
            Assert.That(actual.X, Is.EqualTo((float)expected[0]).Within(tol));
            Assert.That(actual.Y, Is.EqualTo((float)expected[1]).Within(tol));
            Assert.That(actual.Z, Is.EqualTo((float)expected[2]).Within(tol));
        }

        static void AssertClose(JArray expected, float[,] actual, double tol, string what)
        {
            Assert.That(actual.GetLength(0), Is.EqualTo(expected.Count), $"{what}: rows");
            double worst = 0;
            for (int t = 0; t < expected.Count; t++)
                for (int k = 0; k < ((JArray)expected[t]).Count; k++)
                    worst = System.Math.Max(worst, System.Math.Abs((double)expected[t][k] - actual[t, k]));
            Assert.That(worst, Is.LessThanOrEqualTo(tol), $"{what}: max difference {worst:E2}");
        }
    }
}

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ashiria.Engine.Content;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace Ashiria.Engine.Tests
{
    /// <summary>Studio takes (read Python's, write our own), the fixed tick, onboarding and hint wording.</summary>
    public class AppSupportTests
    {
        static JObject GoldenTake() => JObject.Parse(File.ReadAllText(EngineParityTests.GoldenFile("studio_take_v1.json")));

        static Vec3 At(JArray a, int i) => new Vec3((float)a[i], (float)a[i + 1], (float)a[i + 2]);

        static HandSample Sample(JToken side, int frame)
        {
            var pos = (JArray)side["pos"];
            var rot = (JArray)side["rot"];
            var s = new HandSample
            {
                Tracked = (int)side["tracked"][frame] == 1, HighConfidence = (int)side["high_conf"][frame] == 1,
                Inferred = (int)side["inferred"][frame] == 1, SystemGesture = (int)side["system_gesture"][frame] == 1,
            };
            for (int j = 0; j < 26; j++) s.Pos[j] = At(pos, (frame * 26 + j) * 3);
            for (int k = 0; k < 26 * 4; k++) s.Rot[k] = (float)rot[frame * 26 * 4 + k];
            return s;
        }

        static HandInput Input(HandSample s) => new HandInput
        {
            Joints26 = s.Pos, Tracked = s.Tracked, HighConfidence = s.HighConfidence, Inferred = s.Inferred,
            SystemGesture = s.SystemGesture,
        };

        [Test]
        public void StudioTakeCanonicalisesLikePythonAndWritesTheSameFormat()
        {
            var take = GoldenTake();
            var f = take["frames"];
            int n = (int)f["n"];
            bool leftHanded = (string)take["handedness"] == "left";
            float sw = (float)take["shoulder_width_m"];
            var headPos = (JArray)f["head_pos"];
            var headRot = (JArray)f["head_rot"];
            var canon = f["canonical"];
            var writer = new StudioTakeWriter(new StudioHeader
            {
                TakeId = (string)take["take_id"], SessionId = (string)take["session_id"], SignId = (string)take["sign_id"],
                PackId = (string)take["pack_id"], PackVersion = (string)take["pack_version"], Signer = (string)take["signer"],
                Role = (string)take["role"], Handedness = (string)take["handedness"], ConsentRef = (string)take["consent_ref"],
                ShoulderWidthM = sw, RestHeight = (float)take["rest_height"], TickHz = (float)take["tick_hz"],
                Device = (string)take["device"], AppVersion = (string)take["app_version"],
                RecordedAt = System.DateTime.Parse("2026-01-01T00:00:00Z").ToUniversalTime(),
            });

            Vec3[] axes = null;
            var handRel = new Vec3[2, Joints.Count];
            var wrist = new Vec3[2];
            var present = new float[2];
            for (int t = 0; t < n; t++)
            {
                var q = Enumerable.Range(0, 4).Select(i => (float)headRot[t * 4 + i]).ToArray();
                var right = Sample(f["right"], t);
                var left = Sample(f["left"], t);
                QuestCanonical.Frame(At(headPos, t * 3), QuestCanonical.Forward(q[0], q[1], q[2], q[3]), Input(right), Input(left),
                                     leftHanded, sw, ref axes, handRel, wrist, present);
                for (int s = 0; s < 2; s++)
                {
                    Assert.That(present[s], Is.EqualTo((float)canon["present"][t * 2 + s]), $"present[{t},{s}]");
                    if (present[s] < 0.5f) continue;
                    AssertNear(At((JArray)canon["wrist_pos"], (t * 2 + s) * 3), wrist[s], 1e-4, $"wrist[{t},{s}]");
                    for (int j = 0; j < Joints.Count; j++)
                        AssertNear(At((JArray)canon["hand_rel"], ((t * 2 + s) * Joints.Count + j) * 3), handRel[s, j], 1e-4, $"joint[{t},{s},{j}]");
                }
                writer.AddFrame((double)f["t"][t], At(headPos, t * 3), q, right, left, handRel, wrist, present);
            }

            var written = JObject.Parse(writer.ToJson());
            Assert.That(Keys(written), Is.EqualTo(Keys(take)));
            foreach (var key in new[] { "format", "format_version", "take_id", "sign_id", "signer", "handedness", "joint_layout", "status" })
                Assert.That(written[key].ToString(), Is.EqualTo(take[key].ToString()), key);
            Assert.That(Keys((JObject)written["frames"]), Is.EqualTo(Keys((JObject)f)));
            foreach (var path in new[] { "t", "head_pos", "head_rot", "right.pos", "right.tracked", "left.rot", "canonical.hand_rel", "canonical.present" })
            {
                var a = (JArray)written["frames"].SelectToken(path);
                var b = (JArray)f.SelectToken(path);
                Assert.That(a.Count, Is.EqualTo(b.Count), path);
                for (int i = 0; i < a.Count; i++)
                    Assert.That((double)a[i], Is.EqualTo((double)b[i]).Within(2e-4), $"{path}[{i}]");
            }
        }

        static IEnumerable<string> Keys(JObject o) => o.Properties().Select(p => p.Name).OrderBy(k => k);

        static void AssertNear(Vec3 expected, Vec3 actual, double tol, string what)
        {
            Assert.That(actual.X, Is.EqualTo(expected.X).Within(tol), what);
            Assert.That(actual.Y, Is.EqualTo(expected.Y).Within(tol), what);
            Assert.That(actual.Z, Is.EqualTo(expected.Z).Within(tol), what);
        }

        [Test]
        public void TickClockSamplesAtAFixedRate()
        {
            var clock = new TickClock(30f);
            int ticks = 0;
            for (int frame = 0; frame < 90; frame++) ticks += clock.Advance(1.0 / 90);  // one second at 90 Hz
            Assert.That(ticks, Is.EqualTo(30));
            ticks = 0;
            for (int frame = 0; frame < 72; frame++) ticks += clock.Advance(1.0 / 72);
            Assert.That(ticks, Is.InRange(29, 31));
            Assert.That(clock.Advance(5.0), Is.EqualTo(3));  // a stall: no replay of stale poses
            Assert.That(clock.Advance(0.0), Is.EqualTo(0));
        }

        [Test]
        public void RestHeightFromRestingWrists()
        {
            Assert.That(Onboarding.RestHeight(Enumerable.Repeat(-2.0f, 20).ToList()), Is.EqualTo(-1.75f).Within(1e-6));
            Assert.That(Onboarding.RestHeight(Enumerable.Repeat(-1.0f, 20).ToList()), Is.EqualTo(-1.2f).Within(1e-6));
            Assert.That(Onboarding.RestHeight(Enumerable.Repeat(-3.0f, 20).ToList()), Is.EqualTo(-1.9f).Within(1e-6));
            Assert.That(Onboarding.RestHeight(new List<float> { -2f }, -1.5f), Is.EqualTo(-1.5f));
        }

        [Test]
        public void EveryHintCodeHasWords()
        {
            var codes = new[]
            {
                "tracking.lost", "finger.bend_more", "finger.straighten", "spread.more", "spread.less", "palm.face", "hand.point",
                "move", "movement.bigger", "movement.smaller", "movement.path", "other_hand.missing", "other_hand.place",
                "overall.different", "lookalike",
            };
            Assert.That(HintText.Codes.Keys.OrderBy(k => k), Is.EqualTo(codes.OrderBy(k => k)));
            var gloss = new Dictionary<string, string> { ["letter-s"] = "S" };
            foreach (var code in codes)
            {
                var text = HintText.For(new Hint(code, 1.5, finger: "ring", direction: "up", signId: "letter-s"), gloss);
                Assert.That(text, Does.Not.Contain("{"), code);
            }
            Assert.That(HintText.For(new Hint("finger.straighten", 2, finger: "ring")), Is.EqualTo("Straighten your ring finger."));
            Assert.That(HintText.For(new Hint("move", 2, direction: "down")), Is.EqualTo("Move your hand lower."));
            Assert.That(HintText.For(new Hint("lookalike", 2, signId: "letter-s"), gloss), Does.Contain("S"));
        }

        [Test]
        public void VerifyHintsFromThePackHaveWords()
        {
            var golden = JObject.Parse(File.ReadAllText(EngineParityTests.GoldenFile("runtime_v1.json")));
            foreach (var c in golden["verify"])
                foreach (var h in c["hints"])
                    Assert.That(HintText.Codes.ContainsKey((string)h["code"]), (string)h["code"]);
        }
    }
}

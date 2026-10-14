using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json.Linq;

namespace Ashiria.Engine.Content
{
    /// <summary>One catalogue entry: what to teach and how (spec/CONTENT_PACK.md).</summary>
    public sealed class SignInfo
    {
        public string Id, Gloss, Unit, Tier, TierSource, Hint, SourceKind;
        public int Order;
        public bool Recognise, FacialGrammar;
        public List<string> Aliases = new List<string>(), TierReasons = new List<string>(), Lookalikes = new List<string>();
        public string Hands, Movement;
        public bool NearFace, HandsClose;
        public string ReviewStatus;
    }

    public sealed class UnitInfo
    {
        public string Id, Title;
        public int Order;
    }

    /// <summary>
    /// A content pack: manifest.json, catalogue.json and references.json (written by `ashiria pack`). Parsing is by
    /// hand over JToken (no reflection, so IL2CPP stripping cannot break it). The app reads the three files itself
    /// (StreamingAssets, or a sideloaded override in development builds) and passes their text here, ideally off the
    /// main thread.
    /// </summary>
    public sealed class ContentPack
    {
        public const string Format = "ashiria.content-pack";
        public const int FormatVersion = 1;

        public string PackId, Version, Distribution, ContentHash;
        public EngineParams Params;
        public List<UnitInfo> Units = new List<UnitInfo>();
        public List<SignInfo> Catalogue = new List<SignInfo>();
        public Dictionary<string, SignRef> Signs = new Dictionary<string, SignRef>();
        JObject selfcheck;

        public static ContentPack Parse(string manifestJson, string catalogueJson, string referencesJson, bool checkHashes = true)
        {
            var manifest = JObject.Parse(manifestJson);
            if ((string)manifest["format"] != Format)
                throw new FormatException("not an Ashiria content pack");
            if ((int)manifest["format_version"] != FormatVersion)
                throw new FormatException($"content pack format {manifest["format_version"]}; this app reads {FormatVersion}");
            if ((int)manifest["spec_version"] != Features.SpecVersion)
                throw new FormatException($"pack spec_version {manifest["spec_version"]}; this engine is {Features.SpecVersion}");
            if ((int)manifest["runtime_version"] != EngineParams.RuntimeVersion)
                throw new FormatException($"pack runtime_version {manifest["runtime_version"]}; this engine is {EngineParams.RuntimeVersion}");
            if (checkHashes)
            {
                Check(manifest["files"]["catalogue"], catalogueJson);
                Check(manifest["files"]["references"], referencesJson);
            }

            var pack = new ContentPack
            {
                PackId = (string)manifest["pack_id"],
                Version = (string)manifest["version"],
                Distribution = (string)manifest["distribution"],
                ContentHash = (string)manifest["content_hash"],
                Params = ParseParams((JObject)manifest["engine"]),
                selfcheck = (JObject)manifest["selfcheck"],
            };

            var catalogue = JObject.Parse(catalogueJson);
            foreach (var u in catalogue["units"])
                pack.Units.Add(new UnitInfo { Id = (string)u["id"], Title = (string)u["title"], Order = (int)u["order"] });
            foreach (var s in catalogue["signs"])
            {
                var tags = s["tags"];
                pack.Catalogue.Add(new SignInfo
                {
                    Id = (string)s["id"], Gloss = (string)s["gloss"], Unit = (string)s["unit"], Order = (int)s["order"],
                    Tier = (string)s["tier"], TierSource = (string)s["tier_source"], Hint = (string)s["hint"] ?? "",
                    Recognise = (bool)s["recognise"], FacialGrammar = (bool?)s["facial_grammar"] ?? false,
                    Aliases = Strings(s["aliases"]), TierReasons = Strings(s["tier_reasons"]), Lookalikes = Strings(s["lookalikes"]),
                    SourceKind = (string)s["source"]?["kind"],
                    Hands = (string)tags?["hands"], Movement = (string)tags?["movement"],
                    NearFace = (bool?)tags?["near_face"] ?? false, HandsClose = (bool?)tags?["hands_close"] ?? false,
                    ReviewStatus = (string)s["review"]?["status"],
                });
            }

            var refs = JObject.Parse(referencesJson);
            var tiers = new Dictionary<string, SignInfo>();
            foreach (var info in pack.Catalogue) tiers[info.Id] = info;
            foreach (var kv in (JObject)refs["signs"])
            {
                var r = kv.Value;
                var sign = new SignRef
                {
                    Id = kv.Key,
                    Tier = tiers.TryGetValue(kv.Key, out var info) ? info.Tier : "watch",
                    Tau = (float)r["tau"],
                    ShadowTake = (int)r["shadow_take"],
                    Tol = ParseTolerances(r["tol"]),
                    Lookalikes = info != null ? info.Lookalikes : new List<string>(),
                };
                foreach (var t in r["takes"])
                    sign.Takes.Add(new Take
                    {
                        Segment = ParseSegment(t),
                        KeyFrame = (int)t["key_frame"],
                        Source = (string)t["source"],
                        TakeId = (string)t["take_id"],
                    });
                pack.Signs[kv.Key] = sign.Bind(pack.Params);
            }
            return pack;
        }

        /// <summary>Recompute the manifest's self-check from the loaded data. False means this engine and the pack
        /// builder disagree (or the files are damaged): do not use the pack.</summary>
        public bool SelfCheck(out double expected, out double actual)
        {
            expected = (double)selfcheck["descriptor_sum"];
            var take = Signs[(string)selfcheck["sign_id"]].Takes[(int)selfcheck["take"]];
            actual = 0;
            foreach (var v in take.Descriptor) actual += v;
            return Math.Abs(actual - expected) <= (double)selfcheck["tolerance"];
        }

        /// <summary>The look-alike SignRefs of a sign (for Verify's look-alike guard).</summary>
        public List<SignRef> LookalikesOf(string id)
        {
            var list = new List<SignRef>();
            foreach (var other in Signs[id].Lookalikes)
                if (Signs.TryGetValue(other, out var r)) list.Add(r);
            return list;
        }

        static void Check(JToken file, string text)
        {
            using (var sha = SHA256.Create())
            {
                var hash = BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(text))).Replace("-", "").ToLowerInvariant();
                if (hash != (string)file["sha256"])
                    throw new FormatException($"{file["path"]} does not match the manifest (damaged or edited)");
            }
        }

        static List<string> Strings(JToken t)
        {
            var list = new List<string>();
            if (t != null) foreach (var s in t) list.Add((string)s);
            return list;
        }

        static Segment ParseSegment(JToken t)
        {
            int n = (int)t["frames"];
            var hand = (JArray)t["hand_rel"];
            var wrist = (JArray)t["wrist_pos"];
            var present = (JArray)t["present"];
            if (hand.Count != n * 2 * Joints.Count * 3 || wrist.Count != n * 6 || present.Count != n * 2)
                throw new FormatException("reference take arrays have the wrong size");
            var seg = Segment.Empty(n, (float)t["fps"]);
            for (int f = 0; f < n; f++)
                for (int s = 0; s < 2; s++)
                {
                    for (int j = 0; j < Joints.Count; j++)
                    {
                        int k = ((f * 2 + s) * Joints.Count + j) * 3;
                        seg.HandRel[f, s, j] = new Vec3((float)hand[k], (float)hand[k + 1], (float)hand[k + 2]);
                    }
                    int w = (f * 2 + s) * 3;
                    seg.WristPos[f, s] = new Vec3((float)wrist[w], (float)wrist[w + 1], (float)wrist[w + 2]);
                    seg.Present[f, s] = (float)present[f * 2 + s];
                }
            return seg;
        }

        static Tolerances ParseTolerances(JToken t)
        {
            var tol = new Tolerances
            {
                FingerMean = (bool)t["finger_mean"], Spread = (float)t["spread"], OrientationDeg = (float)t["orientation_deg"],
                Location = (float)t["location"], Movement = (float)t["movement"], OtherHand = (float)t["other_hand"],
                DistanceZ = (float)t["distance_z"], LookalikeGuard = (bool)t["lookalike_guard"],
            };
            var finger = (JArray)t["finger"];
            for (int f = 0; f < 5; f++) tol.Finger[f] = (float)finger[f];
            return tol;
        }

        /// <summary>EngineParams from the manifest's "engine" object (params.EngineParams.to_json).</summary>
        public static EngineParams ParseParams(JObject e)
        {
            var p = new EngineParams
            {
                SeqLen = (int)e["seq_len"], GapFillFrames = (int)e["gap_fill_frames"], TrimMarginFrames = (int)e["trim_margin_frames"],
                ActiveFraction = (float)e["active_fraction"], ActiveMinRange = (float)e["active_min_range"],
                ActiveMargin = (int)e["active_margin"], ActiveMinFrames = (int)e["active_min_frames"], DtwBand = (int)e["dtw_band"],
            };
            foreach (var kv in (JObject)e["weights"]) p.Weights[kv.Key] = (float)kv.Value;
            var s = e["segmenter"];
            p.Segmenter = new SegmenterParams
            {
                TickHz = (float)s["tick_hz"], RestHeight = (float)s["rest_height"], StartS = (float)s["start_s"],
                PreRollS = (float)s["pre_roll_s"], DropS = (float)s["drop_s"], HoldS = (float)s["hold_s"],
                HoldRadius = (float)s["hold_radius"], HoldShape = (float)s["hold_shape"], MinAttemptS = (float)s["min_attempt_s"],
                MaxAttemptS = (float)s["max_attempt_s"], RearmMove = (float)s["rearm_move"],
            };
            var v = e["verify"];
            p.Verify = new VerifyParams
            {
                MinPresent = (float)v["min_present"], ThumbScale = (float)v["thumb_scale"], DictionaryScale = (float)v["dictionary_scale"],
                TakeSpreadK = (float)v["take_spread_k"], TakeSpreadCap = (float)v["take_spread_cap"],
                LookalikeRatio = (float)v["lookalike_ratio"], Scored = Tier(v["scored"]), Lenient = Tier(v["lenient"]),
            };
            var r = e["recognise"];
            p.Recognise = new RecogniseParams
            {
                MinPresent = (float)r["min_present"], AcceptZ = (float)r["accept_z"], MinMarginZ = (float)r["min_margin_z"],
                Temperature = (float)r["temperature"], Top = (int)r["top"],
            };
            return p;
        }

        static TierTolerances Tier(JToken t) => new TierTolerances
        {
            Finger = (float)t["finger"], FingerMean = (bool)t["finger_mean"], Spread = (float)t["spread"],
            OrientationDeg = (float)t["orientation_deg"], Location = (float)t["location"], Movement = (float)t["movement"],
            OtherHand = (float)t["other_hand"], DistanceZ = (float)t["distance_z"], LookalikeGuard = (bool)t["lookalike_guard"],
        };
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Ashiria.Engine.Content
{
    /// <summary>One hand's raw tracking for one tick, as Studio records it.</summary>
    public sealed class HandSample
    {
        public Vec3[] Pos = new Vec3[26];       // Unity world space, OpenXR joint order
        public float[] Rot = new float[26 * 4];  // (x, y, z, w) per joint
        public bool Tracked, HighConfidence, Inferred, SystemGesture;
    }

    /// <summary>Who and what a take is (spec/STUDIO_FORMAT.md header). Signer ids are pseudonymous, never names.</summary>
    public sealed class StudioHeader
    {
        public string TakeId = Guid.NewGuid().ToString(), SessionId = "", SignId = "", PackId = "", PackVersion = "";
        public string Signer = "", Role = "reference", Handedness = "right", ConsentRef = "";
        public float ShoulderWidthM = QuestCanonical.DefaultShoulderWidth, RestHeight = -1.5f, TickHz = 30f;
        public bool ShoulderCalibrated;
        public string Device = "", AppVersion = "";
        public DateTime RecordedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Builds a Studio take file (spec/STUDIO_FORMAT.md): raw world poses plus the canonical frames the app computed,
    /// one JSON file per take. Add one frame per fixed tick, then write <see cref="ToJson"/> to
    /// persistentDataPath/studio/&lt;session&gt;/&lt;sign&gt;__&lt;n&gt;.json.
    /// </summary>
    public sealed class StudioTakeWriter
    {
        public const string Format = "ashiria.studio-take";
        public const int FormatVersion = 1;

        public readonly StudioHeader Header;
        readonly List<double> t = new List<double>();
        readonly List<double> headPos = new List<double>(), headRot = new List<double>();
        readonly Dictionary<string, List<double>> pos = new Dictionary<string, List<double>>(), rot = new Dictionary<string, List<double>>();
        readonly Dictionary<string, List<int>[]> flags = new Dictionary<string, List<int>[]>();
        readonly List<double> canonHand = new List<double>(), canonWrist = new List<double>(), canonPresent = new List<double>();

        public int Frames => t.Count;

        public StudioTakeWriter(StudioHeader header)
        {
            Header = header;
            foreach (var side in new[] { "right", "left" })
            {
                pos[side] = new List<double>();
                rot[side] = new List<double>();
                flags[side] = new[] { new List<int>(), new List<int>(), new List<int>(), new List<int>() };
            }
        }

        public void AddFrame(double time, Vec3 head, float[] headRotation, HandSample right, HandSample left,
                             Vec3[,] handRel, Vec3[] wristPos, float[] present)
        {
            t.Add(time);
            Add(headPos, head);
            for (int i = 0; i < 4; i++) headRot.Add(headRotation[i]);
            foreach (var (side, h) in new[] { ("right", right), ("left", left) })
            {
                var f = flags[side];
                f[0].Add(h.Tracked ? 1 : 0);
                f[1].Add(h.HighConfidence ? 1 : 0);
                f[2].Add(h.Inferred ? 1 : 0);
                f[3].Add(h.SystemGesture ? 1 : 0);
                foreach (var p in h.Pos) Add(pos[side], p);
                foreach (var r in h.Rot) rot[side].Add(r);
            }
            for (int s = 0; s < 2; s++)
            {
                for (int j = 0; j < Joints.Count; j++) Add(canonHand, handRel[s, j]);
                Add(canonWrist, wristPos[s]);
                canonPresent.Add(present[s]);
            }
        }

        static void Add(List<double> list, Vec3 v)
        {
            list.Add(v.X);
            list.Add(v.Y);
            list.Add(v.Z);
        }

        public string ToJson(int trimStart = 0, int trimEnd = -1, string status = "keep")
        {
            var h = Header;
            var frames = new JObject
            {
                ["n"] = Frames,
                ["t"] = Numbers(t, 4),
                ["head_pos"] = Numbers(headPos, 6),
                ["head_rot"] = Numbers(headRot, 7),
            };
            foreach (var side in new[] { "right", "left" })
            {
                var f = flags[side];
                frames[side] = new JObject
                {
                    ["tracked"] = new JArray(f[0]), ["high_conf"] = new JArray(f[1]),
                    ["inferred"] = new JArray(f[2]), ["system_gesture"] = new JArray(f[3]),
                    ["pos"] = Numbers(pos[side], 6), ["rot"] = Numbers(rot[side], 6),
                };
            }
            frames["canonical"] = new JObject
            {
                ["hand_rel"] = Numbers(canonHand, 6), ["wrist_pos"] = Numbers(canonWrist, 6), ["present"] = Numbers(canonPresent, 0),
            };
            var take = new JObject
            {
                ["format"] = Format, ["format_version"] = FormatVersion,
                ["take_id"] = h.TakeId, ["session_id"] = h.SessionId, ["sign_id"] = h.SignId,
                ["pack_id"] = h.PackId, ["pack_version"] = h.PackVersion, ["signer"] = h.Signer, ["role"] = h.Role,
                ["handedness"] = h.Handedness, ["consent_ref"] = h.ConsentRef,
                ["shoulder_width_m"] = h.ShoulderWidthM, ["shoulder_calibrated"] = h.ShoulderCalibrated,
                ["rest_height"] = h.RestHeight, ["device"] = h.Device, ["app_version"] = h.AppVersion,
                ["spec_version"] = Features.SpecVersion, ["runtime_version"] = EngineParams.RuntimeVersion,
                ["joint_layout"] = "openxr26", ["tick_hz"] = h.TickHz,
                ["recorded_at"] = h.RecordedAt.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture),
                ["trim"] = new JObject { ["start"] = trimStart, ["end"] = trimEnd < 0 ? Frames : trimEnd },
                ["status"] = status,
                ["frames"] = frames,
            };
            return take.ToString(Formatting.None);
        }

        static JArray Numbers(List<double> values, int digits)
        {
            var a = new JArray();
            foreach (var v in values) a.Add(Math.Round(v, digits));
            return a;
        }
    }
}

using System;
using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>Every engine parameter (port of params.py). Content packs carry them (manifest "engine"), so a
    /// sideloaded pack can retune the headset without a rebuild. Defaults must equal the Python ones.</summary>
    public sealed class EngineParams
    {
        public const int RuntimeVersion = 1;

        public int SeqLen = 32;
        public int GapFillFrames = 5;
        public int TrimMarginFrames = 2;
        public float ActiveFraction = 0.3f;
        public float ActiveMinRange = 0.3f;
        public int ActiveMargin = 2;
        public int ActiveMinFrames = 6;
        public int DtwBand = 8;
        public DescriptorWeights Weights = new DescriptorWeights();
        public SegmenterParams Segmenter = new SegmenterParams();
        public VerifyParams Verify = new VerifyParams();
        public RecogniseParams Recognise = new RecogniseParams();
    }

    /// <summary>Weights of the descriptor parts (matching.py SLICES). Provisional: tuned on simulation.</summary>
    public sealed class DescriptorWeights
    {
        public float Flexion = 1f, Spread = 1f, Tips = 0.5f, PalmNormal = 1f, Direction = 1f, Wrist = 3f, Present = 1f;

        public static readonly string[] Names = { "flexion", "spread", "tips", "palm_normal", "direction", "wrist", "present" };

        public float this[string name]
        {
            get
            {
                switch (name)
                {
                    case "flexion": return Flexion;
                    case "spread": return Spread;
                    case "tips": return Tips;
                    case "palm_normal": return PalmNormal;
                    case "direction": return Direction;
                    case "wrist": return Wrist;
                    case "present": return Present;
                    default: throw new KeyNotFoundException(name);
                }
            }
            set
            {
                switch (name)
                {
                    case "flexion": Flexion = value; break;
                    case "spread": Spread = value; break;
                    case "tips": Tips = value; break;
                    case "palm_normal": PalmNormal = value; break;
                    case "direction": Direction = value; break;
                    case "wrist": Wrist = value; break;
                    case "present": Present = value; break;
                    default: throw new KeyNotFoundException(name);
                }
            }
        }
    }

    /// <summary>Attempt segmentation (segmenter.py). Times in seconds at <see cref="TickHz"/>.</summary>
    public sealed class SegmenterParams
    {
        public float TickHz = 30f;
        public float RestHeight = -1.5f;  // shoulder widths below the nose; set per user at onboarding
        public float StartS = 0.10f;
        public float PreRollS = 0.20f;
        public float DropS = 0.13f;
        public float HoldS = 0.6f;
        public float HoldRadius = 0.12f;
        public float HoldShape = 0.35f;
        public float MinAttemptS = 0.35f;
        public float MaxAttemptS = 4.0f;
        public float RearmMove = 0.3f;

        /// <summary>Seconds to whole ticks, at least one (Python's round-half-to-even).</summary>
        public int Frames(float seconds) => Math.Max(1, (int)Math.Round((double)seconds * TickHz, MidpointRounding.ToEven));

        public SegmenterParams Clone() => (SegmenterParams)MemberwiseClone();
    }

    /// <summary>Tier tolerances (params.py TierTolerances). A tolerance of 0 switches that check off.</summary>
    public sealed class TierTolerances
    {
        public float Finger, Spread, OrientationDeg, Location, Movement, OtherHand, DistanceZ;
        public bool FingerMean, LookalikeGuard;
    }

    public sealed class VerifyParams
    {
        public float MinPresent = 0.6f;
        public float ThumbScale = 1.4f;
        public float DictionaryScale = 1.25f;
        public float TakeSpreadK = 2.5f;
        public float TakeSpreadCap = 2.0f;
        public float LookalikeRatio = 0.9f;
        public TierTolerances Scored = new TierTolerances
        {
            Finger = 0.35f, FingerMean = false, Spread = 0.30f, OrientationDeg = 35f, Location = 0.35f,
            Movement = 0.15f, OtherHand = 0.5f, DistanceZ = 1.3f, LookalikeGuard = true,
        };
        public TierTolerances Lenient = new TierTolerances
        {
            Finger = 0.5f, FingerMean = false, Spread = 0.45f, OrientationDeg = 50f, Location = 0.5f,
            Movement = 0.22f, OtherHand = 0.75f, DistanceZ = 1.6f, LookalikeGuard = true,
        };
    }

    public sealed class RecogniseParams
    {
        public float MinPresent = 0.5f;
        public float AcceptZ = 1.25f;
        public float MinMarginZ = 0.15f;
        public float Temperature = 0.5f;
        public int Top = 5;
    }
}

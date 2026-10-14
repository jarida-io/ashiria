namespace Ashiria.Engine
{
    /// <summary>One hand's tracking for one tick: 26 OpenXR joint positions (Unity world space) and its state.</summary>
    public sealed class HandInput
    {
        public Vec3[] Joints26 = new Vec3[26];
        public bool Tracked, HighConfidence, Inferred, SystemGesture;
    }

    /// <summary>
    /// Quest hand tracking → one canonical frame (spec §4.2). Positions are Unity world space in metres
    /// (left-handed: x right, y up, z forward), e.g. from OVRSkeleton bones with the OpenXR hand skeleton.
    /// The Unity layer reads the SDK and calls this; nothing here depends on UnityEngine.
    /// </summary>
    public static class QuestCanonical
    {
        public const float DefaultShoulderWidth = 0.38f;
        static readonly Vec3 Up = new Vec3(0f, 1f, 0f);

        /// <summary>Nose tip relative to the centre eye in head-yaw axes (x right, y up, z forward), metres: an
        /// average-adult approximation so headset wrist positions line up with the video references' nose origin.</summary>
        public static readonly Vec3 NoseOffset = new Vec3(0f, -0.045f, 0.025f);

        /// <summary>Head-yaw reference axes: y = world up, z = head forward with its vertical part removed,
        /// x = y × z (right). Returns false when looking almost straight up or down; keep the previous axes then.</summary>
        public static bool HeadAxes(Vec3 headForward, out Vec3 x, out Vec3 y, out Vec3 z)
        {
            y = Up;
            var flat = new Vec3(headForward.X, 0f, headForward.Z);
            z = flat.Normalized();
            x = Vec3.Cross(y, z);
            return flat.Length > 1e-3f;
        }

        /// <summary>Fill one hand slot of a frame from the 26 OpenXR joint positions; <paramref name="head"/> is the
        /// centre-eye position. Set <paramref name="mirror"/> for left-handed users (slot 0 must be the dominant hand).</summary>
        public static void Hand(Vec3[] joints26, Vec3 head, Vec3 x, Vec3 y, Vec3 z, float shoulderWidth, bool mirror,
                                Vec3[] handRel21, out Vec3 wristPos)
        {
            var wrist = joints26[Joints.QuestIndex[Joints.Wrist]];
            for (int j = 0; j < Joints.Count; j++)
            {
                var p = joints26[Joints.QuestIndex[j]] - wrist;
                var v = new Vec3(Vec3.Dot(x, p), Vec3.Dot(y, p), Vec3.Dot(z, p));
                handRel21[j] = mirror ? v.MirrorX() : v;
            }
            var w = wrist - (head + x * NoseOffset.X + y * NoseOffset.Y + z * NoseOffset.Z);
            var local = new Vec3(Vec3.Dot(x, w), Vec3.Dot(y, w), Vec3.Dot(z, w)) / shoulderWidth;
            wristPos = mirror ? local.MirrorX() : local;
        }

        /// <summary>Which tracked hand goes in which slot: right-handed users put the right hand in slot 0;
        /// left-handed users put the (mirrored) left hand there.</summary>
        public static int SlotFor(bool isLeftHand, bool leftHanded) => (isLeftHand == leftHanded) ? Joints.Dominant : Joints.Other;

        /// <summary>A unit quaternion (x, y, z, w) applied to (0, 0, 1): the head's forward vector, as Unity's
        /// <c>rotation * Vector3.forward</c>.</summary>
        public static Vec3 Forward(float qx, float qy, float qz, float qw) =>
            new Vec3(2f * (qx * qz + qw * qy), 2f * (qy * qz - qw * qx), 1f - 2f * (qx * qx + qy * qy));

        /// <summary>A hand counts as present only when tracked with high confidence, not pose-inferred, and not
        /// performing the system gesture.</summary>
        public static bool Usable(HandInput h) => h != null && h.Tracked && h.HighConfidence && !h.Inferred && !h.SystemGesture;

        /// <summary>One canonical frame from one tick of tracking (port of quest.frame). <paramref name="headForward"/>
        /// is the head rotation applied to (0, 0, 1). <paramref name="axes"/> carries the reference axes between ticks:
        /// pass the previous tick's (null at first); when looking straight up or down they are kept.
        /// Fills handRel[slot, joint], wristPos[slot] and present[slot].</summary>
        public static void Frame(Vec3 headPos, Vec3 headForward, HandInput right, HandInput left, bool leftHanded,
                                 float shoulderWidth, ref Vec3[] axes, Vec3[,] handRel, Vec3[] wristPos, float[] present)
        {
            if (HeadAxes(headForward, out var x, out var y, out var z))
                axes = new[] { x, y, z };
            else if (axes == null)
                axes = new[] { new Vec3(1f, 0f, 0f), Up, new Vec3(0f, 0f, 1f) };
            var hand = new Vec3[Joints.Count];
            for (int side = 0; side < 2; side++)
            {
                bool isLeft = side == 1;
                var input = isLeft ? left : right;
                int slot = SlotFor(isLeft, leftHanded);
                present[slot] = 0f;
                wristPos[slot] = Vec3.Zero;
                for (int j = 0; j < Joints.Count; j++) handRel[slot, j] = Vec3.Zero;
                if (!Usable(input)) continue;
                Hand(input.Joints26, headPos, axes[0], axes[1], axes[2], shoulderWidth, leftHanded, hand, out var wrist);
                for (int j = 0; j < Joints.Count; j++) handRel[slot, j] = hand[j];
                wristPos[slot] = wrist;
                present[slot] = 1f;
            }
        }
    }
}

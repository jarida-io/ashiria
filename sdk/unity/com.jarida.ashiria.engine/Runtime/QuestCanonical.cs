namespace Ashiria.Engine
{
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
    }
}

namespace Ashiria.Engine
{
    /// <summary>Canonical 21-joint hand layout (MediaPipe order) and the Quest/OpenXR mapping. Spec §3.</summary>
    public static class Joints
    {
        public const int Count = 21;
        public const int Wrist = 0;
        public const int IndexMcp = 5;
        public const int MiddleMcp = 9;
        public const int PinkyMcp = 17;

        /// <summary>Thumb, index, middle, ring, pinky: four joints each, base to tip.</summary>
        public static readonly int[][] Fingers =
        {
            new[] { 1, 2, 3, 4 },
            new[] { 5, 6, 7, 8 },
            new[] { 9, 10, 11, 12 },
            new[] { 13, 14, 15, 16 },
            new[] { 17, 18, 19, 20 },
        };

        /// <summary>OpenXR XR_EXT_hand_tracking index for each canonical joint (palm and finger metacarpals unused).</summary>
        public static readonly int[] QuestIndex =
            { 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25 };

        /// <summary>Hand slots after canonicalisation: 0 = dominant, 1 = the other hand.</summary>
        public const int Dominant = 0;
        public const int Other = 1;
    }
}

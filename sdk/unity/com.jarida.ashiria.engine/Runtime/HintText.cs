using System.Collections.Generic;

namespace Ashiria.Engine
{
    /// <summary>
    /// Default English wording for Verify hints (RUNTIME_SPEC §4). The Deaf co-creators review this wording; a sign's
    /// own catalogue hint is shown after two misses instead. Directions are canonical, so "your signing hand side"
    /// is right for right-handed users and left for left-handed ones.
    /// </summary>
    public static class HintText
    {
        public static readonly Dictionary<string, string> Codes = new Dictionary<string, string>
        {
            ["tracking.lost"] = "I lost sight of your hand. Keep it in front of you.",
            ["finger.bend_more"] = "Bend your {finger} more.",
            ["finger.straighten"] = "Straighten your {finger}.",
            ["spread.more"] = "Spread your fingers apart.",
            ["spread.less"] = "Bring your fingers closer together.",
            ["palm.face"] = "Turn your palm to face {direction}.",
            ["hand.point"] = "Point your fingers {direction}.",
            ["move"] = "Move your hand {direction}.",
            ["movement.bigger"] = "Make the movement bigger.",
            ["movement.smaller"] = "Make the movement smaller.",
            ["movement.path"] = "Follow the shadow hand's movement.",
            ["other_hand.missing"] = "This sign uses both hands.",
            ["other_hand.place"] = "Move your other hand {direction}.",
            ["overall.different"] = "Not quite. Watch the shadow hand and try again.",
            ["lookalike"] = "That looks like {sign}. Check the handshape.",
        };

        static readonly Dictionary<string, string> Fingers = new Dictionary<string, string>
        {
            ["thumb"] = "thumb", ["index"] = "index finger", ["middle"] = "middle finger", ["ring"] = "ring finger",
            ["pinky"] = "little finger",
        };

        static readonly Dictionary<string, string> PalmDirections = new Dictionary<string, string>
        {
            ["up"] = "up", ["down"] = "down", ["forward"] = "away from you", ["back"] = "towards you",
            ["dominant_side"] = "out to the side", ["other_side"] = "across your body",
        };

        static readonly Dictionary<string, string> MoveDirections = new Dictionary<string, string>
        {
            ["up"] = "higher", ["down"] = "lower", ["forward"] = "further from you", ["back"] = "closer to you",
            ["dominant_side"] = "out to the side", ["other_side"] = "across your body",
        };

        /// <summary>Words for a hint. <paramref name="gloss"/> maps sign ids to their glosses (for look-alikes).</summary>
        public static string For(Hint h, IReadOnlyDictionary<string, string> gloss = null)
        {
            if (!Codes.TryGetValue(h.Code, out var text)) return Codes["overall.different"];
            bool place = h.Code == "move" || h.Code == "other_hand.place";
            var directions = place ? MoveDirections : PalmDirections;
            string sign = h.SignId;
            if (sign != null && gloss != null && gloss.TryGetValue(sign, out var g)) sign = g;
            return text
                .Replace("{finger}", h.Finger != null && Fingers.TryGetValue(h.Finger, out var f) ? f : "fingers")
                .Replace("{direction}", h.Direction != null && directions.TryGetValue(h.Direction, out var d) ? d : "")
                .Replace("{sign}", sign ?? "another sign");
        }
    }
}

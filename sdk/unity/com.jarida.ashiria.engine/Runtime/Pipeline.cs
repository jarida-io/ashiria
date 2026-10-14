namespace Ashiria.Engine
{
    /// <summary>The one path from raw canonical frames to a matchable sequence (port of pipeline.py).
    /// Live attempts go through exactly the steps the references went through when the pack was built.</summary>
    public static class Pipeline
    {
        /// <summary>Gap fill → trim to signing space → active window → resample to SeqLen.
        /// Returns null if no hand is ever present.</summary>
        public static Segment Prepare(Segment seg, EngineParams p, float restHeight)
        {
            var s = Canonical.Trim(Canonical.GapFill(seg, p.GapFillFrames), restHeight, p.TrimMarginFrames);
            if (s == null) return null;
            s = Canonical.ActiveWindow(s, p.ActiveFraction, p.ActiveMargin, p.ActiveMinFrames, p.ActiveMinRange);
            return Canonical.Resample(s, p.SeqLen);
        }
    }
}

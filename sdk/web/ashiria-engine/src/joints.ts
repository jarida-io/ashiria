/** Joint layouts (spec/FEATURE_SPEC.md §3). Canonical = MediaPipe's 21 hand landmarks. */

export const COUNT = 21;
export const WRIST = 0;
export const FINGERS: readonly (readonly [number, number, number, number])[] = [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20],
];
export const INDEX_MCP = 5;
export const MIDDLE_MCP = 9;
export const PINKY_MCP = 17;

/** Hand slots: 0 = dominant (the signer's right after mirroring), 1 = the other hand. */
export const DOMINANT = 0;
export const OTHER = 1;

/** OpenXR XR_EXT_hand_tracking (26 joints, palm first) index for each canonical joint. */
export const OPENXR26 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25] as const;
/** WebXR Hand Input (25 joints: OpenXR without the palm) index for each canonical joint. */
export const WEBXR25 = OPENXR26.map((i) => i - 1);

export type JointLayout = 'openxr26' | 'webxr25';
export const layoutIndex = (layout: JointLayout): readonly number[] => (layout === 'openxr26' ? OPENXR26 : WEBXR25);
export const layoutCount = (layout: JointLayout): number => (layout === 'openxr26' ? 26 : 25);

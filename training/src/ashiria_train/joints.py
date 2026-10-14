"""Joint layouts shared by the training pipeline and the headset (spec/FEATURE_SPEC.md §3)."""

# Canonical hand layout: MediaPipe's 21 hand landmarks.
WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, tip
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
NUM_JOINTS = 21

NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Quest (OpenXR XR_EXT_hand_tracking) joint index for each canonical joint, in canonical order.
# OpenXR's palm (0) and the four finger metacarpals (6, 11, 16, 21) are not used.
QUEST_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# Hand slots in every array: 0 = signer's right hand, 1 = signer's left hand.
# After dominant-hand canonicalisation: 0 = dominant hand, 1 = the other hand.
RIGHT, LEFT = 0, 1

# MediaPipe pose landmarks (anatomical left/right) used for the reference frame and hand locations.
POSE_NOSE = 0
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
NUM_POSE = 33

import numpy as np

from ashiria_train import joints as J
from ashiria_train.extract import assign_slots


def det(wrist_x: float, label: str):
    image = np.zeros((21, 3), np.float32)
    image[J.WRIST, :2] = [wrist_x, 0.6]
    return (image, image, label, 0.9)


def pose_with_wrists(right_x: float, left_x: float) -> np.ndarray:
    pose = np.zeros((J.NUM_POSE, 4), np.float32)
    pose[J.POSE_RIGHT_WRIST, :2] = [right_x, 0.6]
    pose[J.POSE_LEFT_WRIST, :2] = [left_x, 0.6]
    return pose


def test_pose_wrists_decide_slots():
    # Facing the camera, the signer's right wrist is on the image's left.
    pose = pose_with_wrists(right_x=0.3, left_x=0.7)
    assert assign_slots([det(0.72, "Right"), det(0.31, "Left")], pose) == {J.RIGHT: 1, J.LEFT: 0}
    assert assign_slots([det(0.29, "Left")], pose) == {J.RIGHT: 0}


def test_handedness_label_without_pose():
    # MediaPipe assumes mirrored input, so "Left" on normal footage is the signer's right hand.
    assert assign_slots([det(0.3, "Left")], None) == {J.RIGHT: 0}
    assert assign_slots([det(0.3, "Right"), det(0.7, "Right")], None) == {J.RIGHT: 0, J.LEFT: 1}

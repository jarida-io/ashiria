import numpy as np
import pytest

from ashiria_train import joints as J
from ashiria_train.augment import rotation
from ashiria_train.canonical import Segment, gap_fill, mirror, resample, trim
from ashiria_train.synthetic import open_hand
from ashiria_train.features import FEATURE_DIM, HAND_DIM, flexion, hand_axes, sequence_features, spread, tip_distances


def segment(frames: int = 10, both: bool = True) -> Segment:
    hand = np.stack([open_hand(), open_hand() * [-1, 1, 1]])
    hand_rel = np.repeat(hand[None], frames, axis=0)
    wrist = np.stack([np.linspace([0.3, -1.0, 0.5], [0.4, -0.5, 0.5], frames),
                      np.tile([-0.5, -1.2, 0.4], (frames, 1))], axis=1)
    present = np.ones((frames, 2))
    if not both:
        present[:, J.LEFT] = 0
    return Segment(hand_rel, wrist, present, fps=30.0)


def test_shape_features_ignore_rotation():
    hand = open_hand()
    R = rotation(40, -25, 70)
    turned = hand @ R.T
    for fn in (flexion, spread):
        np.testing.assert_allclose(fn(turned), fn(hand), atol=1e-9)
    axes, palm = hand_axes(hand)
    axes_t, palm_t = hand_axes(turned)
    np.testing.assert_allclose(tip_distances(turned, palm_t), tip_distances(hand, palm), atol=1e-9)
    np.testing.assert_allclose(axes_t, axes @ R.T, atol=1e-9)  # orientation turns with the hand


def test_open_hand_is_straight():
    middle_and_tip_joints = [4, 5, 7, 8, 10, 11, 13, 14]  # PIP and DIP of index..pinky
    assert flexion(open_hand())[middle_and_tip_joints].max() < 1e-6


def test_feature_layout_and_missing_hand():
    seg = resample(segment(both=False), 32)
    f = sequence_features(seg.hand_rel, seg.wrist_pos, seg.present, 1 / seg.fps)
    assert f.shape == (32, FEATURE_DIM) and f.dtype == np.float32
    assert np.all(f[:, HAND_DIM - 1] == 1)                 # dominant hand present flag
    assert np.all(f[:, HAND_DIM : 2 * HAND_DIM] == 0)      # missing hand is all zeros
    assert np.all(f[:, 2 * HAND_DIM :] == 0)               # no inter-hand vector without both hands


def test_mirror_twice_is_identity():
    seg = segment()
    back = mirror(mirror(seg))
    np.testing.assert_allclose(back.hand_rel, seg.hand_rel)
    np.testing.assert_allclose(back.wrist_pos, seg.wrist_pos)


def test_mirror_swaps_hands():
    seg = segment()
    m = mirror(seg)
    np.testing.assert_allclose(m.wrist_pos[:, J.RIGHT, 0], -seg.wrist_pos[:, J.LEFT, 0])


def test_resample_keeps_endpoints_and_timing():
    seg = segment(frames=61)
    r = resample(seg, 32)
    assert len(r) == 32
    np.testing.assert_allclose(r.wrist_pos[0], seg.wrist_pos[0])
    np.testing.assert_allclose(r.wrist_pos[-1], seg.wrist_pos[-1])
    assert r.fps == pytest.approx(31 / 2.0)  # 61 frames at 30 fps last 2 s


def test_gap_fill_short_but_not_long():
    seg = segment(frames=20)
    seg.present[3:5, 0] = 0
    seg.present[8:16, 0] = 0
    filled = gap_fill(seg, max_gap=5)
    assert filled.present[3:5, 0].all()
    assert not filled.present[8:16, 0].any()


def test_trim_drops_resting_frames():
    seg = segment(frames=20)
    seg.wrist_pos[:5, :, 1] = -3.0
    seg.wrist_pos[15:, :, 1] = -3.0
    out = trim(seg, rest_height=-1.8, margin=0)
    assert len(out) == 10


def test_quest_mapping_covers_canonical_joints():
    assert len(J.QUEST_INDEX) == J.NUM_JOINTS == len(J.NAMES)
    assert len(set(J.QUEST_INDEX)) == J.NUM_JOINTS
    assert {0, 6, 11, 16, 21}.isdisjoint(J.QUEST_INDEX)

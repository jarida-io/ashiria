import numpy as np

from ashiria_train.steps import SHOULDER_M, _detrend, _reversals, _rhythm, _winding, fit_pieces
from ashiria_train.steps_check import read_description

FPS = 24.0


def _circle(cm: float, turns: float, seconds: float, drift_cm: float = 0.0, noise_cm: float = 0.0) -> np.ndarray:
    """A hand path in shoulder widths: circles in front of the body, with forward depth drift and jitter."""
    t = np.arange(int(seconds * FPS)) / FPS
    a = 2 * np.pi * turns * t / seconds
    r = cm / 2 / 100 / SHOULDER_M
    p = np.stack([r * np.cos(a), r * np.sin(a), drift_cm / 100 / SHOULDER_M * t / seconds], 1)
    return p + np.random.default_rng(0).normal(0, noise_cm / 100 / SHOULDER_M, p.shape)


def test_small_circle_with_depth_drift_still_winds():
    # Sorry: a 3 cm circle twice, while video depth creeps 4 cm forward.
    q = _detrend(_circle(3, 2, 1.2, drift_cm=4, noise_cm=0.15))
    front, _ = _winding(q[:, :2])
    assert 1.6 < front < 2.4


def test_rhythm_finds_the_repeat_not_the_next_frame():
    t = np.arange(48) / FPS
    strength, period = _rhythm(np.sin(2 * np.pi * t / 0.4), FPS)
    assert strength > 0.5 and abs(period - 0.4) < 0.05


def test_reversals_ignore_jitter_and_count_taps():
    rng = np.random.default_rng(1)
    still = rng.normal(0, 0.003, 40)
    assert _reversals(still, 0.02) == 0
    taps = np.concatenate([np.linspace(0, -0.1, 6), np.linspace(-0.1, 0, 6)] * 3)  # down, up, three times
    assert _reversals(taps, 0.02) == 5


def test_description_reading():
    d = read_description("Right S-Handshape with palm facing signer moves in small circles on the chest. Repeat.")
    assert d == {"two": False, "shape": "fist", "palm": "towards", "loc": "chest", "moves": ["circle", "repeat"],
                 "other_shape": None, "becomes": None, "still": False, "along": None}
    # The base hand is not the signing hand, and a left base means two hands.
    d = read_description("Left Full Closed Handshape is held. Right X-Handshape with palm facing down taps the left palm.")
    assert d["two"] and d["shape"] == "x" and d["palm"] == "down" and d["moves"] == ["tap"]
    d = read_description("Right V-Handshape with palm facing down taps the left palm twice.")
    assert d["shape"] == "v" and d["two"]
    d = read_description("Both Thumbs-Up Handshapes with palms facing one another are shaken slightly from side to side.")
    assert d["two"] and d["shape"] == d["other_shape"] == "thumb_up" and d["palm"] == "each_other" and d["moves"] == ["repeat"]
    d = read_description("Right Full Closed Handshape with palm facing signer is placed on the chin. Fingers close into Flat-O Handshape.")
    assert d["becomes"] == "flat_o" and d["loc"] == "mouth_chin"
    # The base hand named earlier in its clause, and its palm, are not the signing hand's.
    d = read_description("The left hand is in an S-Handshape with palm facing right and the right Extended Index Finger "
                         "Handshape touches upon the knuckles of the left hand.")
    assert d["shape"] == "index" and d["other_shape"] == "fist" and d["palm"] is None and d["two"]
    d = read_description("Hand is in a closed fist. Palm is facing out.")
    assert d["palm"] == "away" and d["still"]
    assert read_description("Both Thumbs-Up Handshapes are shaken from side to side.")["along"] == "side to side"


def test_written_steps_fit_the_movement_pieces():
    pieces = [(10, 14), (14, 30), (30, 33), (33, 50)]
    for n in range(1, 7):
        fit = fit_pieces(pieces, n)
        assert len(fit) == n and fit[0][0] == 10 and fit[-1][1] == 50
        assert all(a < b for a, b in fit) and all(fit[i][1] == fit[i + 1][0] for i in range(n - 1))
    assert fit_pieces(pieces, 2) == [(10, 33), (33, 50)]  # the short pieces join their neighbours
    assert fit_pieces(pieces, 0) == []

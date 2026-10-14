"""Runtime engine: parameters, DTW paths, prepare, the attempt segmenter, Verify and Recognise."""

import dataclasses

import numpy as np
import pytest

from ashiria_train.augment import rotation
from ashiria_train.canonical import steadiest_frame
from ashiria_train.matching import dtw, dtw_path
from ashiria_train.params import EngineParams
from ashiria_train.pipeline import prepare
from ashiria_train.recognise import recognise
from ashiria_train.segmenter import segment_stream
from ashiria_train.signs import SignRef, Take, make_tolerances
from ashiria_train.synthetic import sign_stream
from ashiria_train.verify import verify

P = EngineParams()
REST = P.segmenter.rest_height
FLAT = np.zeros((5, 3))
FIST = np.full((5, 3), 1.2)
FIST[0] = 0.3
V = FIST.copy()
V[1:3] = 0.0
V[0] = 0.5


def attempt(bends, **kw):
    return prepare(sign_stream(bends, **kw), P, REST)


def sign(id, bends, tier="scored", tau=0.5, **kw) -> SignRef:
    seg = attempt(bends, **kw)
    take = Take(seg, steadiest_frame(seg), "synthetic").bind(P)
    tol = make_tolerances(tier, P.verify, [take], moving=kw.get("path") is not None, two_hands=kw.get("other_at") is not None)
    return SignRef(id, tier, [take], tol, tau=tau).bind(P)


def codes(result):
    return [h.code for h in result.hints]


# --- parameters and matching -------------------------------------------------------------------------

def test_params_round_trip_and_overrides():
    assert EngineParams.from_json(P.to_json()) == P
    cfg = {"features": {"seq_len": 24}, "engine": {"dtw_band": 6, "weights": {"wrist": 2.0},
                                                   "verify": {"scored": {"finger": 0.3}}}}
    p = EngineParams.from_config(cfg)
    assert (p.seq_len, p.dtw_band, p.weights["wrist"], p.weights["flexion"]) == (24, 6, 2.0, 1.0)
    assert p.verify.scored.finger == 0.3 and p.verify.scored.orientation_deg == P.verify.scored.orientation_deg
    with pytest.raises(ValueError):
        EngineParams.from_config({"engine": {"dtw_bnad": 6}})


def test_dtw_path_is_monotone_and_matches_distance():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(32, 5)), rng.normal(size=(28, 5))
    d, path = dtw_path(a, b, 8)
    assert d == pytest.approx(dtw(a, b, 8))
    assert path[0] == (0, 0) and path[-1] == (31, 27)
    steps = np.diff(np.array(path), axis=0)
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(axis=1) >= 1).all()


def test_prepare_cuts_rest_and_resamples():
    seg = prepare(sign_stream(FLAT, rest_frames=20, after=20), P, REST)
    assert len(seg) == P.seq_len
    assert (seg.wrist_pos[:, 0, 1] > REST - 0.5).all()  # the long rest at both ends is gone


# --- segmenter -----------------------------------------------------------------------------------------

def test_raise_and_hold_gives_one_hold_attempt():
    (a,) = segment_stream(P.segmenter, sign_stream(FLAT, hold=40))
    assert a.reason == "hold"
    assert a.end - a.start + 1 == len(a.segment) and len(a.segment) < 40


def test_moving_sign_ends_when_the_hand_drops():
    (a,) = segment_stream(P.segmenter, sign_stream(FLAT, path=(0, 0.6, 0), hold=30))
    assert a.reason == "drop"


def test_fidgeting_below_rest_height_gives_nothing():
    assert segment_stream(P.segmenter, sign_stream(FLAT, at=(0.4, -2.0, 0.6))) == []


def test_lost_tracking_ends_the_attempt():
    attempts = segment_stream(P.segmenter, sign_stream(FLAT, path=(0.9, 0, 0), hold=30, lost=((26, 10),)))
    assert attempts[0].reason == "drop" and attempts[0].end < 36  # tracking returns later: a new attempt


def test_holding_still_does_not_fire_twice_but_moving_on_rearms():
    seg = sign_stream(FLAT, hold=120)
    assert len(segment_stream(P.segmenter, seg)) == 1
    # Hold, then move to a new place and hold again without dropping the hand.
    first = sign_stream(FLAT, hold=40, path=None, drop=1, after=0)
    second = sign_stream(FLAT, at=(0.4, -0.4, 0.6), rest_frames=0, rise=1, hold=40)
    second.wrist_pos[:, 0] = np.concatenate([np.linspace(first.wrist_pos[-2, 0], (0.4, -0.4, 0.6), 10),
                                             second.wrist_pos[10:, 0]])
    both = dataclasses.replace(first, hand_rel=np.concatenate([first.hand_rel[:-1], second.hand_rel]),
                               wrist_pos=np.concatenate([first.wrist_pos[:-1], second.wrist_pos]),
                               present=np.concatenate([first.present[:-1], second.present]))
    assert [a.reason for a in segment_stream(P.segmenter, both)] == ["hold", "hold"]


def test_endless_movement_times_out():
    seg = sign_stream(FLAT, hold=200)
    seg.wrist_pos[:, 0, 0] += 0.4 * np.sin(np.arange(len(seg)) / 5.0)  # never still
    reasons = [a.reason for a in segment_stream(P.segmenter, seg)]
    assert reasons[0] == "timeout"


def test_segmented_attempt_verifies_against_a_full_reference():
    ref = sign("flat", FLAT)
    (a,) = segment_stream(P.segmenter, sign_stream(FLAT, hold=40))
    assert verify(prepare(a.segment, P, REST), ref, P).passed


# --- verify ---------------------------------------------------------------------------------------------

def test_identical_attempt_passes_with_full_score():
    r = verify(attempt(FLAT), sign("flat", FLAT), P)
    assert r.passed and r.score == pytest.approx(1.0) and r.hints == []


def test_bent_ring_finger_is_named():
    bends = FLAT.copy()
    bends[3] += 0.8
    r = verify(attempt(bends), sign("flat", FLAT), P)
    assert not r.passed
    assert (r.hints[0].code, r.hints[0].finger) == ("finger.straighten", "ring")
    assert r.finger_errors[3] < -0.5 and max(abs(e) for i, e in enumerate(r.finger_errors) if i != 3) < 0.1


def test_turned_palm_and_wrong_place_are_named():
    r = verify(attempt(FLAT, orient=rotation(60, 0, 0)), sign("flat", FLAT), P)
    assert r.components["orientation"] > 1 and r.hints[0].code in ("palm.face", "hand.point")
    r = verify(attempt(FLAT, at=(0.4, -0.4, 0.6)), sign("flat", FLAT), P)
    assert (r.hints[0].code, r.hints[0].direction) == ("move", "down")


def test_moving_sign_needs_the_movement():
    ref = sign("wave", FLAT, path=(0.6, 0, 0))
    assert verify(attempt(FLAT, path=(0.6, 0, 0)), ref, P).passed
    r = verify(attempt(FLAT), ref, P)
    assert not r.passed and "movement.bigger" in codes(r)
    r = verify(attempt(FLAT, path=(1.8, 0, 0)), ref, P)
    assert not r.passed and "movement.smaller" in codes(r)


def test_strictness_scale_turns_a_near_miss_into_a_pass():
    bends = FLAT.copy()
    bends[3, 1] += 0.45  # one ring joint: 1.3 × the scored tolerance
    ref = sign("flat", FLAT)
    assert not verify(attempt(bends), ref, P).passed
    assert verify(attempt(bends), ref, P, tolerance_scale=1.5).passed


def test_lookalike_guard_and_tiers():
    a, b = sign("v", V), sign("fist", FIST)
    r = verify(attempt(FIST), a, P, lookalikes=[b])
    assert any(h.code == "lookalike" and h.sign_id == "fist" for h in r.hints)
    assert verify(attempt(FLAT), sign("flat", FLAT, tier="watch"), P).status == "unscored"
    lenient = sign("flat", FLAT, tier="lenient")
    assert lenient.tol.finger[1] > a.tol.finger[1] and lenient.tol.lookalike_guard


def test_no_hands():
    att = attempt(FLAT)
    att.present[:] = 0
    assert verify(att, sign("flat", FLAT), P).status == "no_hands"


def test_dictionary_and_take_spread_widen_finger_tolerances():
    seg = attempt(FLAT)
    takes = [Take(seg, steadiest_frame(seg), "ksl_dictionary").bind(P)]
    tol = make_tolerances("scored", P.verify, takes, False, False)
    assert tol.finger[1] == pytest.approx(0.35 * 1.25) and tol.finger[0] == pytest.approx(0.35 * 1.25 * 1.4)
    bent = FLAT.copy()
    bent[2] += 0.3
    seg2 = attempt(bent)
    takes = [Take(seg, steadiest_frame(seg)).bind(P), Take(seg2, steadiest_frame(seg2)).bind(P)]
    tol = make_tolerances("scored", P.verify, takes, False, False)
    assert tol.finger[2] > 0.35 and tol.finger[2] <= 0.7 + 1e-9 and tol.finger[1] == pytest.approx(0.35)


# --- recognise ------------------------------------------------------------------------------------------

def test_recognise_match_not_sure_and_ambiguous():
    signs = [sign("flat", FLAT), sign("fist", FIST), sign("v", V)]
    r = recognise(attempt(V), signs, P)
    assert (r.status, r.best, r.ranking[0][0]) == ("match", "v", "v") and r.confidence > 0.5
    claw = np.full((5, 3), 0.6)
    r = recognise(attempt(claw, at=(-0.3, -0.2, 0.9), orient=rotation(0, 70, 0)), signs, P)
    assert r.status == "not_sure"
    twins = [sign("flat", FLAT), sign("flat_too", FLAT + 0.02)]
    assert recognise(attempt(FLAT + 0.01), twins, P).status == "ambiguous"
    assert recognise(attempt(FLAT), [], P).status == "no_hands"

"""Verify: "was that sign X?", with feedback on what to fix (spec/RUNTIME_SPEC.md §4).

The attempt is aligned to each of the sign's takes by DTW and the closest take is kept. The attempt
frames aligned to that take's key frame (± KEY_SPAN) give the key pose, which is checked component by component
against the take's own key pose. Each component's error is divided by its tolerance, so 1.0 is the
pass line for every check. Hints are codes; the app turns them into words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonical import Segment
from .features import angle_between
from .matching import FINGERS, descriptor, dtw, dtw_path, finger_errors, segment_features
from .params import EngineParams
from .signs import DOM, SignRef, key_pose, path_length

EPS = 1e-6
KEY_SPAN = 2  # attempt frames either side of those aligned to the key frame that join the median
# Canonical directions in words that hold for left-handed users too (their data is mirrored).
AXES = (("dominant_side", "other_side"), ("up", "down"), ("forward", "back"))
LOCATION_WEIGHTS = np.array([1.0, 1.0, 0.5])  # forward/back is the hardest to judge, so it counts half


def direction_name(v: np.ndarray, weights: np.ndarray | None = None) -> str:
    w = np.abs(v) * (1.0 if weights is None else weights)
    axis = int(np.argmax(w))
    return AXES[axis][0 if v[axis] >= 0 else 1]


@dataclass
class Hint:
    code: str
    severity: float
    finger: str | None = None
    direction: str | None = None
    sign_id: str | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in {"code": self.code, "severity": round(self.severity, 4), "finger": self.finger,
                                  "direction": self.direction, "sign_id": self.sign_id}.items() if v is not None}


@dataclass
class VerifyResult:
    status: str                  # pass | fail | unscored | no_hands
    score: float = 0.0           # 0..1; 0.5 is the pass line
    distance: float = float("inf")
    take: int = -1
    components: dict = field(default_factory=dict)  # name -> error / tolerance; <= 1 passes
    finger_errors: list = field(default_factory=list)  # [5] signed radians, + means bend more
    hints: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def verify(attempt: Segment, sign: SignRef, p: EngineParams, tolerance_scale: float = 1.0,
           lookalikes: list[SignRef] = ()) -> VerifyResult:
    """`attempt` must be prepared (pipeline.prepare). `tolerance_scale` > 1 is more forgiving."""
    if sign.tier == "watch":
        return VerifyResult("unscored")
    present = attempt.present[:, DOM] > 0.5
    if present.mean() < p.verify.min_present:
        return VerifyResult("no_hands", hints=[Hint("tracking.lost", 1.0)])

    feats = segment_features(attempt)
    desc = descriptor(feats, p.weights)
    best = None
    for k, take in enumerate(sign.takes):
        d, path = dtw_path(desc, take.descriptor, p.dtw_band)
        if best is None or d < best[0]:
            best = (d, path, k)
    d, path, k = best
    take = sign.takes[k]
    aligned = [i for i, j in path if j == take.key_frame]
    rows = range(max(0, aligned[0] - KEY_SPAN), min(len(attempt), aligned[-1] + KEY_SPAN + 1))
    pose, ref, tol, s = key_pose(feats, rows), take.key, sign.tol, tolerance_scale

    comps: dict[str, float] = {}
    hints: list[Hint] = []

    err = finger_errors(pose.flexion, ref.flexion)
    ratio = np.abs(err) / (np.asarray(tol.finger) * s)
    comps["shape"] = float(ratio.mean() if tol.finger_mean else ratio.max())
    if comps["shape"] > 1:
        worst = [f for f in np.argsort(-ratio, kind="stable") if ratio[f] > 1] or [int(np.argmax(ratio))]
        hints += [Hint("finger.bend_more" if err[f] > 0 else "finger.straighten", float(ratio[f]), finger=FINGERS[f])
                  for f in worst]

    if tol.spread > 0:
        ds = ref.spread - pose.spread
        comps["spread"] = float(np.abs(ds).max() / (tol.spread * s))
        # Bending a finger at the knuckle also changes the spread angles: only hint at spread if the shape is right.
        if comps["spread"] > 1 and comps["shape"] <= 1:
            hints.append(Hint("spread.more" if ds[np.argmax(np.abs(ds))] > 0 else "spread.less", comps["spread"]))

    palm = np.degrees(angle_between(pose.palm_normal, ref.palm_normal))
    point = np.degrees(angle_between(pose.direction, ref.direction))
    comps["orientation"] = float(max(palm, point) / (tol.orientation_deg * s))
    if comps["orientation"] > 1:
        hints.append(Hint("palm.face", comps["orientation"], direction=direction_name(ref.palm_normal)) if palm >= point
                     else Hint("hand.point", comps["orientation"], direction=direction_name(ref.direction)))

    dw = ref.wrist - pose.wrist
    comps["location"] = float(np.linalg.norm(dw * LOCATION_WEIGHTS) / (tol.location * s))
    if comps["location"] > 1:
        hints.append(Hint("move", comps["location"], direction=direction_name(dw, LOCATION_WEIGHTS)))

    if tol.movement > 0:
        track = attempt.wrist_pos[:, DOM] - pose.wrist
        ref_track = take.segment.wrist_pos[:, DOM] - ref.wrist
        # 2 × DTW (normalised by n + m) ≈ mean distance between aligned frames.
        comps["movement"] = float(2 * dtw(track, ref_track, p.dtw_band) / (tol.movement * s))
        if comps["movement"] > 1:
            size = path_length(attempt.wrist_pos[:, DOM]) / max(take.path_length, EPS)
            code = "movement.bigger" if size < 0.6 else "movement.smaller" if size > 1.6 else "movement.path"
            hints.append(Hint(code, comps["movement"]))

    if tol.other_hand > 0 and ref.other_present:
        if not pose.other_present:
            comps["other_hand"] = 2.0
            hints.append(Hint("other_hand.missing", 2.0))
        else:
            comps["other_hand"] = float(np.linalg.norm(pose.other_rel - ref.other_rel) / (tol.other_hand * s))
            if comps["other_hand"] > 1:
                hints.append(Hint("other_hand.place", comps["other_hand"],
                                  direction=direction_name(ref.other_rel - pose.other_rel)))

    comps["overall"] = float(d / max(sign.tau * tol.distance_z * s, EPS))
    if comps["overall"] > 1:
        hints.append(Hint("overall.different", comps["overall"]))

    if tol.lookalike_guard and lookalikes:
        e, other = -np.inf, None
        for L in lookalikes:  # the first look-alike with the largest error
            e_l = p.verify.lookalike_ratio * d / max(L.closest(desc, p.dtw_band)[0], EPS) / s
            if e_l > e:
                e, other = e_l, L.id
        comps["lookalike"] = float(e)
        if e > 1:
            hints.append(Hint("lookalike", float(e), sign_id=other))

    worst = max(comps.values())
    hints.sort(key=lambda h: -h.severity)
    return VerifyResult("pass" if worst <= 1 else "fail", score=float(np.clip(1 - worst / 2, 0, 1)), distance=d,
                        take=k, components=comps, finger_errors=err.tolist(), hints=hints)

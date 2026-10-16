"""A sign broken into steps for learners: where the hands start, each movement, and how it ends, in plain words.

Steps come from the motion itself (our own wording, not the dictionary's text, so they can ship): the sign is cut at
its pauses, and each key moment is described by handshape, location, palm direction and movement. Those calls come
from small models fitted to the KSL Dictionary's descriptions (steps_fit.py, exports/steps/model.json) when this
computer has them, otherwise from the hand-written rules below. Works on a clip's body-frame pose
(`study.clip_pose`): wrist_pos in shoulder widths from the nose (x the signer's right, y up, z forward), hand_rel in
metres from the wrist, slot 0 the signer's right hand.
"""

from __future__ import annotations

import json
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from . import joints as J
from .features import flexion, spread

PAUSE = 0.45         # a pause: motion below this share of the sign's median motion (MediaPipe's jitter never stops)
MIN_MOVE_S = 0.15    # shorter movements merge into their neighbours
MAX_MOVES = 3
STRAIGHT, BENT = 1.45, 1.7  # a finger's summed bend (radians): straight below, curled above (video landmarks)
SHOULDER_M = 0.34    # metres per shoulder width (the dictionary signer's median), to place hand points
MODEL = "exports/steps/model.json"
MOVE_KINDS = ("tap", "circle", "repeat", "twist", "arc")
RHYTHMIC = ("circle", "tap", "repeat", "rhythm")  # the pause finder cuts these at their own turning points

SHAPE_WORDS = {
    "flat": "a flat hand, fingers together (B)",
    "open": "an open hand, fingers spread (5)",
    "index": "your index finger pointing (1)",
    "fist": "a fist (S)",
    "thumb_up": "a thumbs-up",
    "bent_b": "a flat hand bent at the knuckles",
    "claw": "a claw, fingers spread and bent",
    "o": "an O, fingertips touching your thumb",
    "flat_o": "a flat O, straight fingers touching your thumb",
    "c": "a C, your hand curved",
    "x": "a hooked index finger (X)",
    "v": "a V, index and middle fingers apart",
    "u": "index and middle fingers up, together (U)",
    "l": "an L, thumb and index finger out",
    "f": "an F, index finger touching your thumb, other fingers up",
    "i": "your little finger up (I)",
    "y": "a Y, thumb and little finger out",
    "three": "three fingers up",
    "g": "index finger and thumb pointing (G)",
}
LOCATION_KEYS = ["above_head", "forehead", "face", "mouth_chin", "neck", "chest", "stomach", "waist"]
LOCATION_WORDS = {"above_head": "above your head", "forehead": "at your forehead", "mouth_chin": "at your mouth and chin",
                  "neck": "at your neck", "chest": "in front of your chest", "stomach": "at your stomach",
                  "waist": "at your waist"}
PALM_WORDS = {"towards": "towards you", "away": "away from you", "up": "up", "down": "down"}
# Body points a hand is placed against, per frame (see _targets).
TARGETS = ("forehead", "eyes", "nose", "ear", "far ear", "mouth", "chin", "shoulders", "chest", "stomach", "hips")
TARGET_WORDS = {"far ear": "other ear", "shoulders": "shoulder", "hips": "hip"}
FACE_TARGETS = 7  # forehead to chin
HAND_POINTS = (J.MIDDLE[0], J.INDEX[3], J.THUMB[3], J.MIDDLE[3])
OWN_SIDE = {J.RIGHT: np.array([1.0, 1, 1]), J.LEFT: np.array([-1.0, 1, 1])}  # x towards the hand's own side
MOTION_NAMES = ["seconds", "log_size", "log_travelled", "log_chord", "straightness", "flat2", "flat3", "winding",
                "turning", "front_winding", "front_turning", "reversals", "reversal_rate", "rhythm", "period", "cycles",
                "axis_x", "axis_y", "axis_z", "gap_range", "gap_min", "gap_reversal_rate", "body_min", "body_range",
                "body_reversal_rate", "palm_turn", "palm_net", "curl_range", "curl_net", "speed", "top_speed",
                "small_rhythm", "small_winding", "front_round", "round_winding", "round_front_winding"]


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = np.concatenate([x[:1].repeat(k // 2, 0), x, x[-1:].repeat(k // 2, 0)])
    return np.mean([pad[i:i + len(x)] for i in range(k)], axis=0)


def _motion(wrist: np.ndarray, hand: np.ndarray, seen: np.ndarray, fps: float) -> np.ndarray:
    """How much a hand is moving per frame: wrist speed (per 0.8 shoulder widths a second) plus handshape change
    (per 3 radians a second of finger bending), so a pure handshape change also counts as movement."""
    v = np.linalg.norm(np.gradient(_smooth(wrist, 7), axis=0), axis=-1) * fps
    curl = flexion(hand).sum(-1)
    c = np.abs(np.gradient(_smooth(curl, 9), axis=0)) * fps
    m = v / 0.8 + c / 6.0
    m[~seen] = 0
    return _smooth(m, 5)


def _palm(h: np.ndarray, slot: int) -> np.ndarray:
    n = np.cross(h[J.INDEX[0]] - h[J.WRIST], h[J.PINKY[0]] - h[J.WRIST])
    return n / max(np.linalg.norm(n), 1e-9) * (-1 if slot == J.RIGHT else 1)


def _reversals(x: np.ndarray, h: float) -> int:
    """Turning points of x that it moves at least h away from, so jitter doesn't count."""
    count, hi, lo, going = 0, x[0], x[0], 0
    for v in x[1:]:
        hi, lo = max(hi, v), min(lo, v)
        if going >= 0 and v < hi - h:
            count += going == 1
            going, lo = -1, v
        elif going <= 0 and v > lo + h:
            count += going == -1
            going, hi = 1, v
    return count


def _rhythm(x: np.ndarray, fps: float) -> tuple[float, float]:
    """How strongly x repeats itself and the period in seconds: the highest autocorrelation peak within a second,
    after its first dip (a smoothed path always resembles itself a frame or two later)."""
    x = x - x.mean()
    n, hi = len(x), min(len(x) // 2, int(fps))
    if hi < 3 or x.std() < 1e-6:
        return 0.0, 0.0
    ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    dip = next((k for k in range(1, hi) if ac[k] <= ac[k - 1] and ac[k] <= ac[k + 1]), None)
    if dip is None:
        return 0.0, 0.0
    k = dip + int(np.argmax(ac[dip:hi + 1]))
    return float(ac[k]), k / fps


def _detrend(p: np.ndarray) -> np.ndarray:
    """A path without its steady drift (and the depth creep of video tracking), so loops and back-and-forth stand out."""
    t = np.arange(len(p)) - (len(p) - 1) / 2
    q = p - p.mean(0)
    return q - np.outer(t, (t[:, None] * q).sum(0) / max(float((t ** 2).sum()), 1e-9))


def _winding(u: np.ndarray) -> tuple[float, float]:
    """Turns a 2-D path makes around its centre, and turns of its heading (both near 1 per circle)."""
    r = np.linalg.norm(u, axis=1)
    dth = np.angle(np.exp(1j * np.diff(np.arctan2(u[:, 1], u[:, 0]))))
    ok = (r[1:] > 0.3 * r.max()) & (r[:-1] > 0.3 * r.max())
    v = np.diff(u, axis=0)
    sp = np.linalg.norm(v, axis=1)
    dh = np.angle(np.exp(1j * np.diff(np.arctan2(v[:, 1], v[:, 0]))))
    okv = (sp[1:] > 0.2 * sp.max()) & (sp[:-1] > 0.2 * sp.max())
    return abs(dth[ok].sum()) / (2 * np.pi), abs(dh[okv].sum()) / (2 * np.pi)


# Rules (used when there is no fitted model, and as the baseline steps_fit compares against).

def handshape(h: np.ndarray) -> str:
    """A rough handshape (a SHAPE_WORDS key) for joints [21, 3] (metres from the wrist)."""
    f = flexion(h).reshape(5, 3).sum(-1)               # curl per finger, thumb first
    straight = f[1:] < STRAIGHT
    bent = f[1:] > BENT
    palm = np.linalg.norm(h[J.MIDDLE[0]] - h[J.WRIST])
    thumb_out = np.linalg.norm(h[J.THUMB[3]] - h[J.INDEX[0]]) > 0.75 * palm
    ring = np.linalg.norm(h[J.THUMB[3]] - h[J.INDEX[3]]) < 0.3 * palm
    if ring:
        return "f" if straight[1:].sum() >= 2 else "o"
    if all(straight):
        return "open" if spread(h)[1:].mean() > 0.22 else "flat"
    if all(bent):
        return "thumb_up" if thumb_out else "fist"
    if straight[0] and bent[1:].all():
        return "l" if thumb_out else "index"
    if straight[0] and straight[1] and bent[2:].all():
        return "v"
    if straight[:3].all() and bent[3]:
        return "three"
    if straight[3] and bent[:3].all():
        return "y" if thumb_out else "i"
    if not straight.any() and not bent.any():
        return "c"
    return "bent_b"


def location_key(w: np.ndarray) -> str:
    y = w[1]
    for limit, key in [(0.35, "above_head"), (0.12, "forehead"), (-0.12, "face"), (-0.35, "mouth_chin"), (-0.6, "neck"),
                       (-1.15, "chest"), (-1.6, "stomach")]:
        if y > limit:
            return key
    return "waist"


def palm_key(h: np.ndarray, slot: int) -> str:
    """"towards" (facing the signer), "away", "up", "down" or "side"."""
    n = _palm(h, slot)
    return max([(n[2], "away"), (-n[2], "towards"), (n[1], "up"), (-n[1], "down"), (abs(n[0]), "side")])[1]


def palm_words(key: str, h: np.ndarray, slot: int) -> str:
    if key == "side":
        return "to your right" if _palm(h, slot)[0] > 0 else "to your left"
    return PALM_WORDS[key]


def location_words(key: str, w: np.ndarray, slot: int) -> str:
    """Where a hand is (its centre, shoulder widths from the nose), in words."""
    x = w[0]
    words = ("beside your face" if abs(x) > 0.35 else "in front of your face") if key == "face" else LOCATION_WORDS[key]
    side = x if slot == J.RIGHT else -x  # towards the hand's own side
    return words + (" across your body" if side < -0.3 else " out to the side" if side > 0.7 else "")


def direction(d: np.ndarray) -> str:
    """The main directions of a wrist movement (shoulder widths), largest first: those that make up a good share
    of it."""
    parts = [(d[1], "up"), (-d[1], "down"), (d[0], "to your right"), (-d[0], "to your left"),
             (d[2], "forward"), (-d[2], "towards you")]
    floor = max(0.06, 0.4 * float(np.abs(d).max()))
    big = sorted([p for p in parts if p[0] >= floor], reverse=True)[:2]
    return " and ".join(w for _, w in big)


# Measurements (the fitted models' inputs).

def shape_features(h: np.ndarray) -> np.ndarray:
    """Handshape of joints [21, 3]: bends, spreads, and thumb and fingertip distances in palm lengths."""
    palm = max(float(np.linalg.norm(h[J.MIDDLE[0]] - h[J.WRIST])), 1e-6)
    d = lambda a, b: float(np.linalg.norm(h[a] - h[b])) / palm
    thumb = [d(J.THUMB[3], k) for k in (J.INDEX[0], J.INDEX[3], J.MIDDLE[3], J.RING[3], J.PINKY[3], J.MIDDLE[0], J.PINKY[0])]
    reach = [d(c[3], J.WRIST) for c in J.FINGERS]
    pairs = [d(J.INDEX[3], J.MIDDLE[3]), d(J.MIDDLE[3], J.RING[3]), d(J.RING[3], J.PINKY[3])]
    return np.concatenate([flexion(h), spread(h), thumb, reach, pairs])


def palm_features(h: np.ndarray, slot: int) -> np.ndarray:
    """Palm normal, finger and thumb directions, with x towards the hand's own side so both hands share one model."""
    unit = lambda v: v / max(np.linalg.norm(v), 1e-9)
    f, t = unit(h[J.MIDDLE[0]] - h[J.WRIST]), unit(h[J.THUMB[3]] - h[J.THUMB[0]])
    return np.concatenate([_palm(h, slot), f, t]) * np.tile(OWN_SIDE[slot], 3)


def _targets(pose: dict) -> np.ndarray | None:
    """TARGETS per frame and hand [frames, 2, 11, 3] ("ear" is the hand's own side), from the face and body."""
    if "face" not in pose or "hips" not in pose:
        return None
    face = np.asarray(pose["face"], float)  # nose, left eye, right eye, left ear, right ear, mouth left, mouth right
    top, low = np.asarray(pose["shoulders"], float).mean(1), np.asarray(pose["hips"], float).mean(1)
    nose, eyes, mouth = face[:, 0], face[:, 1:3].mean(1), face[:, 5:7].mean(1)
    body = lambda ear, far: np.stack([eyes + 0.8 * (eyes - mouth), eyes, nose, ear, far, mouth, mouth + 0.9 * (mouth - nose),
                                      top, top + 0.35 * (low - top), top + 0.7 * (low - top), low], 1)
    return np.stack([body(face[:, 4], face[:, 3]), body(face[:, 3], face[:, 4])], 1)


def two_features(W: SimpleNamespace) -> np.ndarray:
    """Does the second hand sign: how far it lifts, how much it moves and is seen, how close it comes."""
    d, o = W.dom, W.other
    span = slice(W.a, W.b + 1)
    both = W.seen[span, d] & W.seen[span, o]
    gap = np.linalg.norm(W.centre[span, d] - W.centre[span, o], axis=-1)[both]
    raised = (W.wrist[span, o, 1] - W.rest[o])[W.seen[span, o]]
    nan = float("nan")
    return np.array([W.lift[d], W.lift[o], W.lift[o] / max(W.lift[d], 1e-6), W.seen[span, o].mean(),
                     W.motion[span, o].mean() / max(W.motion[span, d].mean(), 1e-6),
                     np.median(gap) if gap.size else nan, gap.min() if gap.size else nan,
                     np.median(raised) if raised.size else nan, raised.max() if raised.size else nan])


def place_features(W: SimpleNamespace, t: int, s: int) -> np.ndarray:
    """Where hand s is at frame t: its centre (x towards its own side), and how close its nearest point comes to each
    body point, then and at any time during the sign (shoulder widths)."""
    c = W.centre[t, s] * OWN_SIDE[s]
    if W.targets is None:
        return np.concatenate([c, np.full(2 * len(TARGETS) + 2, np.nan)])
    near = W.near[:, s]
    span = np.arange(W.a, W.b + 1)
    span = span[W.seen[span, s]]
    closest = near[span].min(0) if span.size else np.full(len(TARGETS), np.nan)
    rel = [c[1] - W.targets[t, s, 2, 1], c[1] - W.targets[t, s, 7, 1]]  # height from the nose and the shoulders
    return np.concatenate([c, near[t], closest, rel])


def motion_features(W: SimpleNamespace, s0: int, s1: int) -> np.ndarray:
    """How the signing hand moves over frames s0..s1 (MOTION_NAMES). Most measures don't depend on the movement's
    size (winding, reversals, rhythm, straightness), so a 3 cm circle reads like a large one."""
    d, o = W.dom, W.other
    p = _smooth(W.centre[s0:s1 + 1, d], 3)
    k = len(p)
    if k < 6:
        return np.full(len(MOTION_NAMES), np.nan)
    dur = k / W.fps
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    travelled, chord = step.sum(), np.linalg.norm(p[-1] - p[0])
    q = _detrend(p)
    size = float(np.ptp(q, axis=0).max())
    _, sv, vt = np.linalg.svd(q, full_matrices=False)
    sd = sv / np.sqrt(k) + 1e-9
    winding, turning = _winding(q @ vt[:2].T)
    front_winding, front_turning = _winding(q[:, :2])  # video depth is the least reliable axis
    fs = np.linalg.svd(q[:, :2], compute_uv=False) + 1e-9
    # A circle is round (two similar widths); shaking that winds only through jitter is long and thin.
    round_best, round_front = sd[1] / sd[0], fs[1] / fs[0]
    h = lambda x: max(0.02, 0.25 * float(np.ptp(x)))
    axes = [q @ vt[0], q[:, 0], q[:, 1]]
    rev = max(_reversals(x, h(x)) for x in axes)
    rhythm, period = max(_rhythm(x, W.fps) for x in axes)

    nan = float("nan")
    span = slice(s0, s1 + 1)
    both = W.seen[span, d] & W.seen[span, o]
    gap_f = [nan, nan, nan]
    if both.sum() >= 6:
        g = np.linalg.norm(_smooth(W.centre[span, d], 3) - _smooth(W.centre[span, o], 3), axis=1)[both]
        gap_f = [float(np.ptp(g)), float(g.min()), _reversals(g, h(g)) / dur]
    body_f = [nan, nan, nan]
    if W.targets is not None:
        nb = _smooth(W.near[span, d].min(1), 3)
        body_f = [float(nb.min()), float(np.ptp(nb)), _reversals(nb, h(nb)) / dur]
    normals = _smooth(np.array([_palm(hh, d) for hh in W.hand[span, d]]), 5)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    cosines = np.clip((normals[1:] * normals[:-1]).sum(1), -1, 1)
    curl = _smooth(flexion(W.hand[span, d]).sum(-1), 5)
    return np.array([dur, np.log(size + 0.01), np.log(travelled + 0.01), np.log(chord + 0.01), chord / (travelled + 1e-3),
                     sd[1] / sd[0], sd[2] / sd[0], winding, turning, front_winding, front_turning, rev, rev / dur,
                     rhythm, period, dur / period if rhythm > 0.2 and period else 0.0, *np.abs(vt[0]), *gap_f, *body_f,
                     np.arccos(cosines).sum(), np.arccos(np.clip(normals[0] @ normals[-1], -1, 1)), float(np.ptp(curl)),
                     abs(curl[-1] - curl[0]), travelled / dur, step.max() * W.fps, rhythm * (size < 0.15),
                     front_winding * (size < 0.2), round_front, winding * round_best, front_winding * round_front])


def clip_motion(W: SimpleNamespace) -> np.ndarray:
    """The whole sign's movement, between its key moments and over all of its raised part (a small repeated movement
    never pauses, so its key moments can cut off a loop)."""
    return np.concatenate([motion_features(W, W.rise, W.drop), motion_features(W, W.a, W.b)])


# Fitted models (steps_fit.py), applied without scikit-learn.

@lru_cache(maxsize=1)
def _model() -> dict | None:
    from .config import training_relative
    f = training_relative(MODEL)
    return json.loads(f.read_text()) if f.exists() else None


def _probs(m: dict, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x - np.asarray(m["mean"])) / np.asarray(m["scale"]))  # a missing measurement counts as average
    s = np.asarray(m["coef"]) @ z + np.asarray(m["intercept"])
    if len(s) == 1:
        p = 1 / (1 + np.exp(-s[0]))
        return np.array([1 - p, p])
    e = np.exp(s - s.max())
    return e / e.sum()


def predict(name: str, x: np.ndarray, sure: bool = False):
    """The model's call, or None without a model; with `sure`, also None when it isn't confident enough to say."""
    model = _model()
    m = model["models"].get(name) if model else None
    if m is None or len(m["mean"]) != len(x):  # no model, or one fitted to older measurements
        return None
    p = _probs(m, x)
    k = int(np.argmax(p))
    return None if sure and p[k] < m.get("say_above", 0) else m["classes"][k]


def move_scores(x: np.ndarray) -> dict[str, float]:
    """Each movement kind's probability over its threshold: 1 or more means it is there."""
    model = _model()
    return {k: float(_probs(m, x)[1] / m["threshold"]) for k, m in (model or {}).get("moves", {}).items()
            if len(m["mean"]) == len(x)}


def _window(pose: dict, use_model: bool, known: dict | None = None) -> SimpleNamespace | None:
    """The sign proper and its key moments, and which hands sign."""
    fps = pose["fps"]
    wrist = np.asarray(pose["wrist_pos"], float)
    hand = np.asarray(pose["hand_rel"], float)
    seen = np.asarray(pose["present"], float) > 0.5
    if not seen.any():
        return None
    # Each hand's resting height is where it starts; a hand signs when it rises well above that. The dictionary
    # signer rests the spare hand at the stomach, so "raised" is relative, not a fixed height.
    y = np.where(seen, wrist[:, :, 1], np.nan)
    rest = np.array([np.nanpercentile(y[:, s], 10) if seen[:, s].any() else 0.0 for s in (0, 1)])
    peak = np.array([np.nanpercentile(y[:, s], 95) if seen[:, s].any() else 0.0 for s in (0, 1)])
    lift = peak - rest
    # The references are right-handed: the right hand signs unless the left clearly does more.
    dom = J.LEFT if lift[J.LEFT] > 1.3 * lift[J.RIGHT] else J.RIGHT
    other = 1 - dom
    # The sign proper: while the signing hand is above halfway between its rest and its highest point.
    up = np.flatnonzero(seen[:, dom] & (wrist[:, dom, 1] > rest[dom] + 0.5 * lift[dom]))
    if up.size < 3:
        return None
    W = SimpleNamespace(fps=fps, wrist=wrist, hand=hand, seen=seen, rest=rest, lift=lift, dom=dom, other=other,
                        a=int(up[0]), b=int(up[-1]))
    # Where a hand is: its middle knuckle (the centre of the hand), not the wrist, which sits below the fingers.
    W.centre = wrist + hand[:, :, J.MIDDLE[0]] / SHOULDER_M
    W.targets = _targets(pose)
    if W.targets is not None:
        points = wrist[:, :, None] + hand[:, :, HAND_POINTS] / SHOULDER_M
        # Distances in the picture plane: video depth puts a hand on the chest 20 cm in front of it.
        W.near = np.linalg.norm(points[:, :, :, None, :2] - W.targets[:, :, None, :, :2], axis=-1).min(2)  # [frames, 2, 11]
    W.motion = np.stack([_motion(wrist[:, s], hand[:, s], seen[:, s], fps) for s in (0, 1)], 1)
    W.two_x = two_features(W)
    learned = None
    m2 = (_model() or {}).get("models", {}).get("two") if use_model else None
    if m2 and len(m2["mean"]) == len(W.two_x):
        p2 = _probs(m2, W.two_x)
        learned = m2["classes"][int(np.argmax(p2))] if p2.max() >= 0.8 else None  # 74% overall: only when very sure
    if known and known.get("two") is not None:
        learned = known["two"]
    W.two = bool(learned) if learned is not None else bool(lift[other] > 0.4 and lift[other] > 0.45 * lift[dom])
    W.hands = [dom, other] if W.two else [dom]
    m = W.motion[:, W.hands].max(1)
    # The moment the signing hand settles into its starting position, and the moment before it starts back down.
    a, b = W.a, W.b
    quiet = PAUSE * np.median(m[a:b + 1])
    n = b - a
    low = lambda t: m[t] <= quiet * 1.5 and m[t] <= m[t - 1] and m[t] <= m[t + 1]
    rise = next((t for t in range(a + 1, a + int(0.4 * n)) if low(t)), a + n // 8)
    drop = next((t for t in range(b - 1, b - int(0.4 * n), -1) if low(t)), b - n // 8)
    if drop - rise < MIN_MOVE_S * fps:
        rise, drop = a, b
    W.m, W.quiet, W.rise, W.drop = m, quiet, rise, drop
    return W


def _around(W: SimpleNamespace, t: int, s: int, fn) -> np.ndarray:
    """fn of hand s averaged over the frames around t where it is seen (a single frame's fingers are noisy)."""
    ts = [u for u in range(t - 2, t + 3) if 0 <= u < len(W.seen) and W.seen[u, s]] or [t]
    return np.mean([fn(W.hand[u, s]) for u in ts], axis=0)


def measure(pose: dict) -> dict | None:
    """The fitted models' inputs for one clip, at its starting key moment (steps_fit.py)."""
    W = _window(pose, use_model=False)
    if W is None:
        return None
    t, s = W.rise, W.dom
    return {"two": W.two_x, "shape": _around(W, t, s, shape_features),
            "palm": _around(W, t, s, lambda h: palm_features(h, s)), "loc": place_features(W, t, s),
            "move": clip_motion(W)}


def _segments(m: np.ndarray, start: int, end: int, fps: float, quiet: float) -> list[tuple[int, int]]:
    """Cut [start, end] at pauses (local minima of motion below `quiet`) into movements."""
    cuts = [start]
    for t in range(start + 1, end):
        if m[t] < quiet and m[t] <= m[t - 1] and m[t] <= m[t + 1] and t - cuts[-1] >= MIN_MOVE_S * fps:
            cuts.append(t)
    if end - cuts[-1] < MIN_MOVE_S * fps and len(cuts) > 1:
        cuts.pop()
    cuts.append(end)
    segs = list(zip(cuts[:-1], cuts[1:]))
    while len(segs) > MAX_MOVES:  # merge the shortest movement into its neighbour
        k = int(np.argmin([b - a for a, b in segs]))
        j = k - 1 if k > 0 else k + 1
        a, b = min(segs[k][0], segs[j][0]), max(segs[k][1], segs[j][1])
        segs[min(k, j)] = (a, b)
        del segs[max(k, j)]
    return segs


def _rule_kind(W: SimpleNamespace, s0: int, s1: int) -> str | None:
    """circle, repeat or twist by the hand-written rules (None: decide from distance and handshape)."""
    dom = W.dom
    path = W.wrist[s0:s1 + 1, dom]
    chord = np.linalg.norm(W.wrist[s1, dom] - W.wrist[s0, dom])
    axis = np.argmax(np.ptp(path, axis=0)) if len(path) > 2 else 0
    steps_v = np.sign(np.diff(_smooth(path[:, axis], 3))) if len(path) > 3 else np.array([])
    reversals = int((np.diff(steps_v[steps_v != 0]) != 0).sum()) if steps_v.size else 0
    # Distance travelled along a well-smoothed path (raw jitter would add up even for a still hand).
    travelled = float(np.linalg.norm(np.diff(_smooth(path, 7), axis=0), axis=1).sum()) if len(path) > 7 else 0.0
    turn = np.degrees(np.arccos(np.clip(np.dot(*[_palm(W.hand[t, dom], dom) for t in (s0, s1)]), -1, 1)))
    if chord < 0.15 and travelled > 0.3 and reversals < 2:
        return "circle"
    if reversals >= 2 and np.ptp(path[:, axis]) > 0.12:
        return "repeat"
    if chord < 0.15 and turn > 60:
        return "twist"
    return None


def _is_travel(W: SimpleNamespace, s0: int, s1: int) -> bool:
    """A movement that goes somewhere (a long, fairly straight path), not part of a repeated one."""
    path = W.centre[s0:s1 + 1, W.dom]
    chord = float(np.linalg.norm(path[-1] - path[0]))
    travelled = float(np.linalg.norm(np.diff(_smooth(path, 3), axis=0), axis=1).sum()) if len(path) > 3 else chord
    return chord >= 0.3 and chord > 0.6 * travelled


def _assign_moves(W: SimpleNamespace, segs: list, found: dict[str, float]) -> tuple[dict, list]:
    """Attach the movement kinds found in the whole sign to its pieces. A repeated movement (taps, circles, shaking)
    pauses at every turn, so the pause finder cuts it up: its pieces become one step, apart from any long travel."""
    kinds: dict[tuple[int, int], str] = {}
    rhythmic = max((k for k in RHYTHMIC if k in found), key=found.get, default=None)
    if rhythmic:
        rest = [i for i, sg in enumerate(segs) if not _is_travel(W, *sg)]
        i, j = (rest[0], rest[-1]) if rest else (max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0]),) * 2
        segs = segs[:i] + [(segs[i][0], segs[j][1])] + segs[j + 1:]
        kinds[segs[i]] = rhythmic
    free = [sg for sg in segs if sg not in kinds]
    if "twist" in found and free:
        turn = lambda sg: float(np.arccos(np.clip(_palm(W.hand[sg[0], W.dom], W.dom) @ _palm(W.hand[sg[1], W.dom], W.dom), -1, 1)))
        best = max(free, key=turn)
        kinds[best] = "twist"
        free.remove(best)
    if "arc" in found:
        moving = [sg for sg in free if np.linalg.norm(W.wrist[sg[1], W.dom] - W.wrist[sg[0], W.dom]) >= 0.15]
        if moving:
            kinds[max(moving, key=lambda sg: sg[1] - sg[0])] = "arc"
    return kinds, segs


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def sign_steps(pose: dict, use_model: bool = True, known: dict | None = None) -> dict:
    """Steps for one clip's pose (study.clip_pose). Frames are indices into the clip.

    `known` is what is already known about the sign: the catalogue's written steps (steps_text: start, each
    movement, end), or facts read from its dictionary description (steps_check.read_description: two, shape,
    other_shape, palm, loc, moves, becomes, still, along). Known facts win over what the video suggests, because
    video from one camera can't show handshapes, palm direction or contact reliably; the motion supplies timing,
    direction, size and rhythm. Without them, a caption states only what the fitted models are sure of.
    `use_model=False` uses only the hand-written rules (steps_fit compares the two)."""
    k = known or {}
    W = _window(pose, use_model, k)
    if W is None:
        return {"hands": [], "steps": []}
    if k.get("steps_text"):
        return _written_steps(W, k)
    if not use_model or _model() is None:
        return _rule_steps(W)
    dom, other, two = W.dom, W.other, W.two
    name = {J.RIGHT: "right hand", J.LEFT: "left hand"}

    def sure(t: int, s: int) -> tuple:
        """Handshape, palm and location of hand s at frame t, where the models are sure enough (else None)."""
        return (predict("shape", _around(W, t, s, shape_features), sure=True),
                predict("palm", _around(W, t, s, lambda h: palm_features(h, s)), sure=True),
                predict("loc", place_features(W, t, s), sure=True))

    shape, palm, loc = sure(W.rise, dom)
    shape, palm, loc = k.get("shape") or shape, k.get("palm") or palm, k.get("loc") or loc
    other_shape = (k.get("other_shape") or sure(W.rise, other)[0]) if two else None

    def shape_words(key: str | None, s: int) -> str:
        return SHAPE_WORDS.get(key, "") if key else ""

    def hand_words(sh: str | None, pa: str | None, lo: str | None, s: int, plural: bool = False) -> str:
        parts = [shape_words(sh, s), location_words(lo, W.centre[W.rise, s], s) if lo else ""]
        if pa == "each_other":
            parts.append("palms facing each other")
        elif pa:
            parts.append(f"palm{'s' if plural else ''} {palm_words(pa, W.hand[W.rise, s], s)}")
        return ", ".join(p for p in parts if p)

    if two and other_shape and other_shape == shape:
        start = f"Start with both hands: {hand_words(shape, palm, loc, dom, plural=True)}"
    elif two:
        own = hand_words(shape, palm, loc, dom)
        start = (f"Start with both hands. Your {name[dom]}: {own}" if own else "Start with both hands")
        if other_shape:
            start += f". Your {name[other]}: {hand_words(other_shape, 'each_other' if palm == 'each_other' else None, None, other)}"
    else:
        own = hand_words(shape, palm, loc, dom)
        start = f"Start with your {name[dom]}: {own}" if own else f"Start with your {name[dom]}, as shown"
    facts = {"two": two, "shape": shape, "palm": "side" if palm == "each_other" else palm, "loc": loc, "moves": []}
    steps = [{"kind": "start", "from": W.a, "to": W.rise, "key": W.rise, "text": start + "."}]

    def contact(t: int) -> str:
        """What the signing hand is at, at frame t: the other hand, or a part of the face. Only these: in the picture
        a hand in front of the chest always overlaps the body, touching it or not."""
        if two and np.linalg.norm(W.centre[t, dom, :2] - W.centre[t, other, :2]) < 0.35:
            return "your other hand"
        if W.targets is None:
            return ""
        j = int(np.argmin(W.near[t, dom, :FACE_TARGETS]))
        return f"your {TARGET_WORDS.get(TARGETS[j], TARGETS[j])}" if W.near[t, dom, j] < 0.2 else ""

    segs = _segments(W.m, W.rise, W.drop, W.fps, W.quiet)
    whole = motion_features(W, W.a, W.b)
    rhythm, cycles = whole[MOTION_NAMES.index("rhythm")], whole[MOTION_NAMES.index("cycles")]
    if k.get("still"):
        segs, kinds = [], {}  # a held sign: what the video shows moving is the hand settling
    else:
        found = {m: 2.0 for m in k.get("moves") or []}
        if not found:
            found = {m: v for m, v in move_scores(clip_motion(W)).items() if v >= 1}
        if not found and rhythm >= 0.4 and cycles >= 2:
            found = {"rhythm": 1.0}  # clearly repeated, though video can't tell a small circle from a bounce
        kinds, segs = _assign_moves(W, segs, found)

    for s0, s1 in segs:
        kind = kinds.get((s0, s1))
        d = W.wrist[s1, dom] - W.wrist[s0, dom]
        if kind is None:
            if not _is_travel(W, s0, s1) and np.linalg.norm(d) < 0.15:
                continue  # a pause in place, or a movement too small to name
            kind = "straight"
        q = _detrend(_smooth(W.centre[s0:s1 + 1, dom], 3))
        main = np.linalg.svd(q, full_matrices=False)[2][0] if len(q) > 2 else np.array([1.0, 0, 0])
        span = np.ptp(q, axis=0) if len(q) > 1 else np.zeros(3)
        along = k.get("along") or ("back and forth" if span[2] > 2 * span[:2].max() else
                                   ["side to side", "up and down"][int(span[1] > span[0])])
        size_cm = float(np.ptp(q, axis=0).max()) * SHOULDER_M * 100
        turns = max(_reversals(x, max(0.02, 0.25 * float(np.ptp(x)))) for x in (q @ main, q[:, 0], q[:, 1])) if len(q) > 2 else 0
        times = turns // 2 + 1
        if kind in RHYTHMIC and (s0, s1) == (segs[0][0], segs[-1][1]) and rhythm >= 0.3 and cycles >= 1.5:
            times = int(round(cycles))  # the whole sign repeats: count its cycles over all of it
            size_cm = float(np.exp(whole[MOTION_NAMES.index("log_size")]) - 0.01) * SHOULDER_M * 100
        at = contact(s1) if kind != "straight" else ""
        where = (f" on {at}" if "hand" in at else f" at {at}") if at else ""
        if kind == "circle":
            what = f"move in small circles{where}, about {max(2, round(size_cm))} cm across" if size_cm <= 12 else f"move in circles{where}"
        elif kind == "tap":
            what = f"tap {_times(max(2, times))}{where}"
        elif kind == "repeat":
            what = (f"shake slightly {along}" if size_cm < 6 else f"move {along}") + (f", {_times(times)}" if times >= 2 else ", repeating")
        elif kind == "rhythm":
            what = f"make a small movement{where}, about {max(2, round(size_cm))} cm, {_times(max(2, times))}"
        elif kind == "twist":
            what = "turn your wrist" + (f" so your palm faces {palm_words(palm_key(W.hand[s1, dom], dom), W.hand[s1, dom], dom)}"
                                        if palm_key(W.hand[s1, dom], dom) != "side" else "")
        else:
            way = direction(d)
            end = contact(s1)
            what = ("move " + way if way else "move") + (" in an arc" if kind == "arc" else "") + (f" to {end}" if end else "")
        if two and kind != "tap" and np.linalg.norm(W.centre[s1, 0, :2] - W.centre[s1, 1, :2]) < 0.3 \
                and "other hand" not in what:
            what += ", hands meeting"
        facts["moves"].append(kind)
        steps.append({"kind": "move", "from": s0, "to": s1, "key": s1, "text": what[0].upper() + what[1:] + "."})

    becomes = k.get("becomes")
    if becomes and becomes != shape:
        if len(steps) > 1:
            steps[-1]["text"] = steps[-1]["text"][:-1] + f", your hand becoming {SHAPE_WORDS.get(becomes, becomes)}."
        else:
            steps.append({"kind": "move", "from": W.rise, "to": W.drop, "key": W.drop,
                          "text": f"Change your hand to {SHAPE_WORDS.get(becomes, becomes)}."})
    hands = "hands" if two else "hand"
    steps.append({"kind": "end", "from": W.drop, "to": W.b, "key": W.drop, "text": f"Hold it, then lower your {hands}."})
    return {"hands": [name[s] for s in W.hands], "dominant": name[dom], "steps": steps, "facts": facts}


def fit_pieces(segs: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Exactly n movement pieces covering the same frames: the shortest merged into a neighbour, or the longest
    split in two, until the count matches."""
    segs = list(segs) if n else []
    while len(segs) > n:
        j = int(np.argmin([b - a for a, b in segs]))
        i = j - 1 if j > 0 else j + 1
        segs[min(i, j)] = (segs[min(i, j)][0], segs[max(i, j)][1])
        del segs[max(i, j)]
    while segs and len(segs) < n:
        j = int(np.argmax([b - a for a, b in segs]))
        a, b = segs[j]
        segs[j:j + 1] = [(a, (a + b) // 2), ((a + b) // 2, b)]
    return segs


def _written_steps(W: SimpleNamespace, k: dict) -> dict:
    """The catalogue's written steps on the clip's timing: the first is the start, the last the end, and the ones
    between are laid over the sign's movements (pieces merged or split to match their number)."""
    texts = [t.strip() for t in k["steps_text"]]
    moves = texts[1:-1]
    segs = fit_pieces(_segments(W.m, W.rise, W.drop, W.fps, W.quiet), len(moves))
    name = {J.RIGHT: "right hand", J.LEFT: "left hand"}
    steps = [{"kind": "start", "from": W.a, "to": W.rise, "key": W.rise, "text": texts[0]}]
    steps += [{"kind": "move", "from": a, "to": b, "key": b, "text": t} for (a, b), t in zip(segs, moves)]
    end = texts[-1] if len(texts) > 1 else f"Hold it, then lower your {'hands' if W.two else 'hand'}."
    steps.append({"kind": "end", "from": W.drop, "to": W.b, "key": W.drop, "text": end})
    return {"hands": [name[s] for s in W.hands], "dominant": name[W.dom], "steps": steps,
            "facts": {"two": W.two, "shape": k.get("shape"), "palm": k.get("palm"), "loc": k.get("loc"), "moves": []}}


def _rule_steps(W: SimpleNamespace) -> dict:
    """The hand-written rules' steps (no fitted model on this computer, and steps_fit's baseline)."""
    dom, other, two = W.dom, W.other, W.two
    name = {J.RIGHT: "right hand", J.LEFT: "left hand"}
    who = "both hands" if two else f"your {name[dom]}"
    calls = lambda t, s: (handshape(W.hand[t, s]), palm_key(W.hand[t, s], s), location_key(W.centre[t, s]))

    def pose_words(t: int, s: int) -> str:
        shape, palm, loc = calls(t, s)
        return f"{SHAPE_WORDS[shape]}, {location_words(loc, W.centre[t, s], s)}, palm {palm_words(palm, W.hand[t, s], s)}"

    shape0, palm0, loc0 = calls(W.rise, dom)
    start_text = f"Start with {who}: " + pose_words(W.rise, dom)
    if two:
        start_text += f". Other hand: {pose_words(W.rise, other)}"
    facts = {"two": two, "shape": shape0, "palm": palm0, "loc": loc0, "moves": []}
    steps = [{"kind": "start", "from": W.a, "to": W.rise, "key": W.rise, "text": start_text + "."}]
    prev_shape = shape0
    for s0, s1 in _segments(W.m, W.rise, W.drop, W.fps, W.quiet):
        d = W.wrist[s1, dom] - W.wrist[s0, dom]
        chord = float(np.linalg.norm(d))
        shape = calls(s1, dom)[0]
        kind = _rule_kind(W, s0, s1)
        if kind is None:
            if chord >= 0.15:
                path = W.wrist[s0:s1 + 1, dom]
                u = d / chord
                off = (path - W.wrist[s0, dom]) - np.outer((path - W.wrist[s0, dom]) @ u, u)
                kind = "arc" if len(path) > 2 and float(np.linalg.norm(off, axis=1).max()) > 0.25 * chord else "straight"
            elif shape != prev_shape:
                kind = "shape"
            else:
                prev_shape = shape
                continue  # a pause in place: nothing new to learn
        if kind == "circle":
            what = "move in small circles"
        elif kind == "repeat":
            path = W.wrist[s0:s1 + 1, dom]
            what = f"move {['side to side', 'up and down', 'back and forth'][int(np.argmax(np.ptp(path, axis=0)))]}, repeating"
        elif kind == "twist":
            what = f"turn your wrist so your palm faces {palm_words(palm_key(W.hand[s1, dom], dom), W.hand[s1, dom], dom)}"
        elif kind == "shape":
            what = f"change the handshape to {SHAPE_WORDS[shape]}"
        else:
            what = f"move {direction(d)}" + (" in an arc" if kind == "arc" else "") \
                + f", ending {location_words(location_key(W.centre[s1, dom]), W.centre[s1, dom], dom)}"
            if shape != prev_shape:
                what += f", hand becoming {SHAPE_WORDS[shape]}"
        if two and np.linalg.norm(W.wrist[s1, 0] - W.wrist[s1, 1]) < 0.3:
            what += ", hands meeting"
        prev_shape = shape
        facts["moves"].append(kind)
        steps.append({"kind": "move", "from": s0, "to": s1, "key": s1, "text": what[0].upper() + what[1:] + "."})
    steps.append({"kind": "end", "from": W.drop, "to": W.b, "key": W.drop,
                  "text": f"End: {pose_words(W.drop, dom)}, then lower your {'hands' if two else 'hand'}."})
    return {"hands": [name[s] for s in W.hands], "dominant": name[dom], "steps": steps, "facts": facts}

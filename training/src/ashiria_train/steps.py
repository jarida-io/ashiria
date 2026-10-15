"""A sign broken into steps for learners: where the hands start, each movement, and how it ends, in plain words.

Steps come from the motion itself (our own wording, not the dictionary's text, so they can ship): the sign is cut at
its pauses, and each key moment is described by handshape, location, palm direction and movement. Works on a
clip's body-frame pose (`study.clip_pose`): wrist_pos in shoulder widths from the nose (x the signer's right, y up,
z forward), hand_rel in metres from the wrist, slot 0 the signer's right hand.
"""

from __future__ import annotations

import numpy as np

from . import joints as J
from .features import flexion, spread

PAUSE = 0.45         # a pause: motion below this share of the sign's median motion (MediaPipe's jitter never stops)
MIN_MOVE_S = 0.15    # shorter movements merge into their neighbours
MAX_MOVES = 3
STRAIGHT, BENT = 1.45, 1.7  # a finger's summed bend (radians): straight below, curled above (video landmarks)


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


def handshape(h: np.ndarray) -> str:
    """A rough name for the handshape of joints [21, 3] (metres from the wrist)."""
    f = flexion(h).reshape(5, 3).sum(-1)               # curl per finger, thumb first
    straight = f[1:] < STRAIGHT
    bent = f[1:] > BENT
    palm = np.linalg.norm(h[J.MIDDLE[0]] - h[J.WRIST])
    thumb_out = np.linalg.norm(h[J.THUMB[3]] - h[J.INDEX[0]]) > 0.75 * palm
    ring = np.linalg.norm(h[J.THUMB[3]] - h[J.INDEX[3]]) < 0.3 * palm
    spread_out = spread(h)[1:].mean() > 0.22
    if ring and straight[1:].sum() >= 2:
        return "an O (thumb touching the index finger), other fingers up"
    if ring:
        return "an O shape (fingertips touching the thumb)"
    if all(straight):
        return "an open hand, fingers spread" if spread_out else "a flat hand, fingers together" + (", thumb out" if thumb_out else "")
    if all(bent):
        return "a fist, thumb out" if thumb_out else "a fist"
    if straight[0] and bent[1:].all():
        return "an L (thumb and index out)" if thumb_out else "pointing with the index finger"
    if straight[0] and straight[1] and bent[2:].all():
        return "a V (index and middle up, apart)" if spread(h)[1] > 0.25 else "two fingers up, together"
    if straight[:3].all() and bent[3]:
        return "three fingers up"
    if straight[3] and bent[:3].all():
        return "a Y (thumb and little finger out)" if thumb_out else "the little finger up"
    if not straight.any() and not bent.any():
        return "a curved hand, like holding a cup (C)"
    return "bent fingers"


def palm_facing(h: np.ndarray, slot: int) -> str:
    """Which way the palm faces, from the signer's own point of view ("up", "towards you", ...)."""
    n = _palm(h, slot)
    words = [(n[2], "away from you"), (-n[2], "towards you"), (n[1], "up"), (-n[1], "down"),
             (n[0], "to your right"), (-n[0], "to your left")]
    return max(words)[1]


def location(w: np.ndarray, slot: int) -> str:
    """Where a wrist is, from shoulder widths relative to the nose."""
    x, y, z = w
    if y > 0.35:
        height = "above your head"
    elif y > 0.12:
        height = "at your forehead"
    elif y > -0.12:
        height = "beside your face" if abs(x) > 0.35 else "in front of your face"
    elif y > -0.35:
        height = "at your mouth and chin"
    elif y > -0.6:
        height = "at your neck"
    elif y > -1.15:
        height = "in front of your chest"
    elif y > -1.6:
        height = "at your stomach"
    else:
        height = "at your waist"
    side = x if slot == J.RIGHT else -x  # towards the hand's own side
    across = " across your body" if side < -0.3 else " out to the side" if side > 0.7 else ""
    return height + across


def direction(d: np.ndarray) -> str:
    """The main directions of a wrist movement (shoulder widths), largest first: those that make up a good share
    of it."""
    parts = [(d[1], "up"), (-d[1], "down"), (d[0], "to your right"), (-d[0], "to your left"),
             (d[2], "forward"), (-d[2], "towards you")]
    floor = max(0.06, 0.4 * float(np.abs(d).max()))
    big = sorted([p for p in parts if p[0] >= floor], reverse=True)[:2]
    return " and ".join(w for _, w in big)


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


def sign_steps(pose: dict) -> dict:
    """Steps for one clip's pose (study.clip_pose). Frames are indices into the clip."""
    fps = pose["fps"]
    wrist = np.asarray(pose["wrist_pos"], float)
    hand = np.asarray(pose["hand_rel"], float)
    seen = np.asarray(pose["present"], float) > 0.5
    if not seen.any():
        return {"hands": [], "steps": []}
    # Each hand's resting height is where it starts; a hand signs when it rises well above that. The dictionary
    # signer rests the spare hand at the stomach, so "raised" is relative, not a fixed height.
    y = np.where(seen, wrist[:, :, 1], np.nan)
    rest = np.array([np.nanpercentile(y[:, s], 10) if seen[:, s].any() else 0.0 for s in (0, 1)])
    peak = np.array([np.nanpercentile(y[:, s], 95) if seen[:, s].any() else 0.0 for s in (0, 1)])
    lift = peak - rest
    # The references are right-handed: the right hand signs unless the left clearly does more.
    dom = J.LEFT if lift[J.LEFT] > 1.3 * lift[J.RIGHT] else J.RIGHT
    other = 1 - dom
    two = lift[other] > 0.4 and lift[other] > 0.45 * lift[dom]
    hands = [dom, other] if two else [dom]
    m = np.max([_motion(wrist[:, s], hand[:, s], seen[:, s], fps) for s in hands], axis=0)

    # The sign proper: while the signing hand is above halfway between its rest and its highest point; then the
    # moment it settles into its starting position, and the moment before it starts back down.
    up = np.flatnonzero(seen[:, dom] & (wrist[:, dom, 1] > rest[dom] + 0.5 * lift[dom]))
    if up.size < 3:
        return {"hands": [], "steps": []}
    a, b = int(up[0]), int(up[-1])
    quiet = PAUSE * np.median(m[a:b + 1])
    n = b - a
    low = lambda t: m[t] <= quiet * 1.5 and m[t] <= m[t - 1] and m[t] <= m[t + 1]
    rise = next((t for t in range(a + 1, a + int(0.4 * n)) if low(t)), a + n // 8)
    drop = next((t for t in range(b - 1, b - int(0.4 * n), -1) if low(t)), b - n // 8)
    if drop - rise < MIN_MOVE_S * fps:
        rise, drop = a, b

    name = {J.RIGHT: "right hand", J.LEFT: "left hand"}
    who = "both hands" if two else f"your {name[dom]}"

    def pose_words(t: int, s: int) -> str:
        return f"{handshape(hand[t, s])}, {location(wrist[t, s], s)}, palm {palm_facing(hand[t, s], s)}"

    steps = []
    start_text = f"Start with {who}: " + pose_words(rise, dom)
    if two:
        start_text += f". Other hand: {pose_words(rise, other)}"
    steps.append({"kind": "start", "from": a, "to": rise, "key": rise, "text": start_text + "."})

    prev_shape = handshape(hand[rise, dom])
    for k, (s0, s1) in enumerate(_segments(m, rise, drop, fps, quiet)):
        d = wrist[s1, dom] - wrist[s0, dom]
        path = wrist[s0:s1 + 1, dom]
        chord = np.linalg.norm(d)
        bend = 0.0
        if chord > 1e-6 and len(path) > 2:
            u = d / chord
            off = (path - wrist[s0, dom]) - np.outer((path - wrist[s0, dom]) @ u, u)
            bend = float(np.linalg.norm(off, axis=1).max())
        # Back and forth: the movement reverses along its main axis at least twice.
        axis = np.argmax(np.ptp(path, axis=0)) if len(path) > 2 else 0
        steps_v = np.sign(np.diff(_smooth(path[:, axis], 3))) if len(path) > 3 else np.array([])
        reversals = int((np.diff(steps_v[steps_v != 0]) != 0).sum()) if steps_v.size else 0
        shape = handshape(hand[s1, dom])
        # Distance travelled along a well-smoothed path (raw jitter would add up even for a still hand).
        travelled = float(np.linalg.norm(np.diff(_smooth(path, 7), axis=0), axis=1).sum()) if len(path) > 7 else 0.0
        turn = np.degrees(np.arccos(np.clip(np.dot(*[_palm(hand[t, dom], dom) for t in (s0, s1)]), -1, 1)))
        if chord < 0.15 and travelled > 0.3 and reversals < 2:
            what = "move in small circles"
        elif reversals >= 2 and np.ptp(path[:, axis]) > 0.12:
            what = f"move {['side to side', 'up and down', 'back and forth'][axis]}, repeating"
        elif chord < 0.15 and turn > 60:
            what = f"turn your wrist so your palm faces {palm_facing(hand[s1, dom], dom)}"
        elif chord < 0.15 and shape != prev_shape:
            what = f"change the handshape to {shape}"
        elif chord < 0.15:
            prev_shape = shape
            continue  # a pause in place: nothing new to learn
        else:
            what = "move " + direction(d) + (" in an arc" if bend > 0.25 * chord and chord > 0.15 else "")
            what += f", ending {location(wrist[s1, dom], dom)}"
            if shape != prev_shape:
                what += f", hand becoming {shape}"
        if two and np.linalg.norm(wrist[s1, 0] - wrist[s1, 1]) < 0.3:
            what += ", hands meeting"
        prev_shape = shape
        steps.append({"kind": "move", "from": s0, "to": s1, "key": s1, "text": what[0].upper() + what[1:] + "."})

    steps.append({"kind": "end", "from": drop, "to": b, "key": drop,
                  "text": f"End: {pose_words(drop, dom)}, then lower your {'hands' if two else 'hand'}."})
    return {"hands": [name[s] for s in hands], "dominant": name[dom], "steps": steps}


def _palm(h: np.ndarray, slot: int) -> np.ndarray:
    n = np.cross(h[J.INDEX[0]] - h[J.WRIST], h[J.PINKY[0]] - h[J.WRIST])
    return n / max(np.linalg.norm(n), 1e-9) * (-1 if slot == J.RIGHT else 1)

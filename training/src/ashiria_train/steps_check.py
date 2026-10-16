"""`ashiria steps-check`: how often the generated steps agree with the KSL Dictionary's own descriptions.

The descriptions are used only here, to measure and tune the step generator (steps.py); nothing from them is shown
in the app. Each description is read for: whether both hands sign, the starting handshape, which way the palm faces,
where the sign starts, and the kinds of movement (taps, circles, arcs, twists, repeats).
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .config import load_config, path

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"

# The dictionary's handshape names -> steps.SHAPE_WORDS keys.
HANDSHAPES = {"full closed": "flat", "b": "flat", "full open": "open", "5": "open", "bent full open": "claw",
              "curved full open": "claw", "claw": "claw", "extended index finger": "index", "index finger": "index",
              "s": "fist", "a": "fist", "fist": "fist", "closed fist": "fist", "bent-b": "bent_b", "o": "o",
              "single-o": "o", "flat-o": "flat_o", "c": "c", "modified-c": "c", "v": "v", "l": "l", "x": "x", "y": "y",
              "i": "i", "f": "f", "g": "g", "u": "u", "thumbs-up": "thumb_up", "w": "three", "3": "three", "curved-three": "three"}
PALMS = {"signer": "towards", "out": "away", "outward": "away", "forward": "away", "down": "down", "downward": "down",
         "up": "up", "upward": "up", "left": "side", "right": "side", "one another": "each_other", "each other": "each_other",
         "sides": "side", "side": "side", "inward": "side"}
LOCATIONS = {"forehead": "forehead", "temple": "forehead", "head": "forehead", "eye": "face", "eyes": "face",
             "nose": "face", "cheek": "face", "cheeks": "face", "ear": "face", "ears": "face", "face": "face",
             "mouth": "mouth_chin", "lips": "mouth_chin", "lip": "mouth_chin", "teeth": "mouth_chin", "chin": "mouth_chin",
             "jaw": "mouth_chin", "neck": "neck", "throat": "neck", "shoulder": "chest", "shoulders": "chest",
             "chest": "chest", "heart": "chest", "stomach": "stomach", "belly": "stomach", "tummy": "stomach",
             "waist": "waist", "hip": "waist", "hips": "waist"}
MOVES = {"tap": "tap", "taps": "tap", "tapping": "tap", "tapped": "tap", "knock": "tap", "knocks": "tap", "pat": "tap",
         "pats": "tap", "circle": "circle", "circles": "circle", "circled": "circle", "circling": "circle", "circular": "circle", "rub": "circle", "rubs": "circle",
         "rubbing": "circle", "arc": "arc", "arcs": "arc", "twist": "twist", "twists": "twist", "twisting": "twist",
         "turns over": "twist", "flips": "twist", "twice": "repeat", "repeat": "repeat", "repeats": "repeat",
         "repeated": "repeat", "repetitively": "repeat", "repeatedly": "repeat", "back and forth": "repeat",
         "up and down": "repeat", "side to side": "repeat", "to and fro": "repeat", "shake": "repeat",
         "shakes": "repeat", "shaken": "repeat", "shaking": "repeat", "wiggle": "repeat", "wiggles": "repeat",
         "flap": "repeat", "flaps": "repeat", "flapping": "repeat", "wave": "repeat", "waves": "repeat"}

_alt = lambda words: "|".join(sorted(map(re.escape, words), key=len, reverse=True))
_shape_re = re.compile(r"\b(left |right |both )?(?:an? )?(" + _alt(HANDSHAPES) + r")[ -]handshapes?\b", re.I)
_palm_re = re.compile(r"palms? (?:is |are )?facing (?:the )?(" + _alt(PALMS) + r")\b", re.I)
_loc_re = re.compile(r"\b(" + _alt(LOCATIONS) + r")\b", re.I)
# Words that say the hand moves (a description without any describes a held sign, like most letters).
_motion_re = re.compile(r"\b(mov(?:e|es|ing|ed)|brought|brings?|draws?|swings?|slides?|push(?:es)?|pulls?|turns|twists?|"
                        r"tap\w*|shak\w*|circl\w*|rub\w*|flap\w*|wav(?:e|es|ing)|wiggl\w*|repeat\w*|closes?|opens?|"
                        r"flicks?|lifts?|raises?|lowers?|touch(?:es)?|lands?|hits?|strikes?|bounces?|separates?|twice|"
                        r"alternately)\b", re.I)
_along_re = re.compile(r"\b(side to side|up and down|back and forth|to and fro)\b", re.I)
_becomes_re = re.compile(r"\binto (?:an? )?(" + _alt(HANDSHAPES) + r")[ -]handshapes?\b", re.I)
_move_re = re.compile(r"\b(" + _alt(MOVES) + r")\b", re.I)
_two_re = re.compile(r"^\s*both\b|\bboth hands\b|\bhands\b|\bhandshapes\b|\bpalms\b|\barms\b|\bleft hand\b"
                     r"|\bleft [a-z\- ]*handshape|\bleft (?:palm|arm|forearm|wrist|elbow|fingers?|thumb)\b|\bone another\b"
                     r"|\beach other\b", re.I)


def _hand_of(text: str, at: int) -> str:
    """Which hand the phrase at `at` is about, from its clause ("The left hand is in an S-Handshape ...")."""
    clause = re.split(r"[.;,(]|\band\b", text[:at])[-1].lower()
    return next((h for h in ("left", "right", "both") if re.search(rf"\b{h}\b", clause)), "")


def read_description(text: str) -> dict:
    """What a description says about the start (the signing hand's shape, palm and place, from the first sentence
    that describes it rather than a left base hand; the other hand's shape) and the whole sign (two hands, kinds of
    movement, the handshape it changes into)."""
    t = " ".join(text.split())
    sentences = re.split(r"(?<=\.)\s", t)
    hand = lambda m: (m.group(1) or "").lower().strip() or _hand_of(t, m.start())
    shapes = [m for m in _shape_re.finditer(t) if t.index(sentences[0]) <= m.start()]
    first2 = len(" ".join(sentences[:2]))
    shape = next((m for m in shapes if m.start() < first2 and hand(m) != "left"), None)
    palms = [m for m in _palm_re.finditer(t) if _hand_of(t, m.start()) != "left"]
    palm = next((m for m in palms if shape and m.start() > shape.end()), None) or (palms[0] if palms else None)
    first = next((x for x in sentences if shape and shape.group(0) in x), sentences[0])
    loc = _loc_re.search(first)
    two = bool(_two_re.search(t))
    left = next((m for m in shapes if hand(m) == "left"), None)
    both = shape is not None and (hand(shape) == "both" or bool(re.match(r"\s*both\b", first, re.I)))
    becomes = _becomes_re.search(t)
    key = lambda m: HANDSHAPES[m.group(2).lower()] if m else None
    return {"two": two, "shape": key(shape), "palm": PALMS[palm.group(1).lower()] if palm else None,
            "loc": LOCATIONS[loc.group(1).lower()] if loc else None,
            "moves": sorted({MOVES[m.lower()] for m in _move_re.findall(t)}),
            "other_shape": (key(shape) if both else key(left)) if two else None,
            "becomes": HANDSHAPES[becomes.group(1).lower()] if becomes else None,
            "still": not _motion_re.search(t),
            "along": (lambda m: m and ("back and forth" if m.group(1).lower() == "to and fro" else m.group(1).lower()))(_along_re.search(t))}


def described_clips() -> pd.DataFrame:
    dcfg = load_config(DICTIONARY_CONFIG)
    labels = pd.read_csv(path(dcfg, "manifests") / "dictionary_labels.csv", dtype=str).fillna("")
    videos = pd.read_csv(path(dcfg, "manifests") / "videos.csv", dtype=str).set_index("path")
    labels["clip"] = labels["path"].map(lambda p: videos.loc[p, "clip_id"] if p in videos.index else None)
    return labels[(labels["description"].str.strip() != "") & labels["clip"].notna()]


def check(limit: int | None = None, quiet: bool = False, use_model: bool = True) -> dict:
    from .steps import sign_steps
    from .study import clip_pose

    rows = []
    for _, r in described_clips().head(limit).iterrows():
        pose = clip_pose(r["clip"])
        learn = sign_steps(pose, use_model) if pose is not None else {}
        if not learn.get("facts"):
            continue
        said = read_description(r["description"])
        said["palm"] = "side" if said["palm"] == "each_other" else said["palm"]
        rows.append((said, learn["facts"], r["label"]))
    out = {"clips": len(rows)}
    for key in ("two", "shape", "palm", "loc"):
        pairs = [(d[key], f[key]) for d, f, _ in rows if d[key] is not None]
        stated = [(a, b) for a, b in pairs if b is not None]
        agree = sum(a == b for a, b in stated)
        out[key] = {"described": len(pairs), "stated": round(len(stated) / max(len(pairs), 1), 3),
                    "agree": round(agree / max(len(stated), 1), 3)}
        if not quiet:
            conf = Counter(f"{a} -> {b}" for a, b in stated if a != b).most_common(6)
            print(f"{key:6s} stated for {out[key]['stated']:.0%} of {len(pairs)}, agree {out[key]['agree']:.0%}  "
                  f"most common mix-ups: {conf}")
    for move in ("tap", "circle", "arc", "twist", "repeat"):
        said = [move in d["moves"] for d, _, _ in rows]
        ours = [move in f["moves"] for _, f, _ in rows]
        tp = sum(a and b for a, b in zip(said, ours))
        out[move] = {"described": sum(said), "found": round(tp / max(sum(said), 1), 3),
                     "false": round(sum(b and not a for a, b in zip(said, ours)) / max(len(said) - sum(said), 1), 3)}
        if not quiet:
            print(f"{move:6s} described in {sum(said)}: found {out[move]['found']:.0%}; claimed where not described {out[move]['false']:.0%}")
    return out

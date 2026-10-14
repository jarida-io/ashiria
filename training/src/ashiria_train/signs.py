"""What the engine knows about a sign: its reference takes and how strictly to check it.

A Take is one prepared reference recording (a Studio take on Quest, or a dictionary draft) with its
key frame. Everything derived from it (features, descriptor, key pose) is computed on load, in
Python and C# alike, so packs carry only the recording itself. Tolerances are final per-sign values
computed here at pack time; the headset only multiplies them by the user's strictness setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import joints as J
from .canonical import Segment
from .features import HAND_DIM, normalize
from .matching import SLICES, descriptor, dtw, segment_features
from .params import EngineParams, VerifyParams

DOM, OTHER = J.RIGHT, J.LEFT
OTHER_OFFSET = HAND_DIM
INTER = slice(2 * HAND_DIM, 2 * HAND_DIM + 3)  # other wrist minus dominant wrist (both present)
TIERS = ("scored", "lenient", "watch")


@dataclass
class KeyPose:
    """The dominant hand's shape, orientation and place at a key moment, plus the other hand's place."""
    flexion: np.ndarray      # [15] radians
    spread: np.ndarray       # [4] radians
    palm_normal: np.ndarray  # [3] unit
    direction: np.ndarray    # [3] unit
    wrist: np.ndarray        # [3] shoulder widths from the nose
    other_present: bool
    other_rel: np.ndarray    # [3] other wrist minus dominant wrist; zeros if the other hand is absent


def key_pose(features: np.ndarray, rows) -> KeyPose:
    """Per-component median over `rows` of a [T, 197] feature array, using only rows where the dominant
    hand is present (all rows if none are). Directions are re-normalised after the median."""
    rows = np.asarray(list(rows), int)
    f = features[rows].astype(np.float64)
    dom = f[:, SLICES["present"].start] > 0.5
    use = f[dom] if dom.any() else f
    other = f[:, OTHER_OFFSET + SLICES["present"].start] > 0.5
    both = dom & other

    def med(sl: slice) -> np.ndarray:
        return np.median(use[:, sl], axis=0)

    return KeyPose(
        flexion=med(SLICES["flexion"]),
        spread=med(SLICES["spread"]),
        palm_normal=normalize(med(SLICES["palm_normal"])),
        direction=normalize(med(SLICES["direction"])),
        wrist=med(SLICES["wrist"]),
        other_present=bool(other.sum() * 2 > len(rows)),
        other_rel=np.median(f[both][:, INTER], axis=0) if both.any() else np.zeros(3),
    )


@dataclass
class Take:
    segment: Segment         # prepared: p.seq_len frames
    key_frame: int
    source: str = "studio"   # studio | ksl_dictionary | synthetic
    take_id: str = ""
    features: np.ndarray = field(init=False, repr=False)
    descriptor: np.ndarray = field(init=False, repr=False)
    key: KeyPose = field(init=False, repr=False)
    path_length: float = field(init=False)

    def bind(self, p: EngineParams) -> Take:
        """Compute what matching needs. Call once after loading, with the pack's parameters."""
        self.features = segment_features(self.segment)
        self.descriptor = descriptor(self.features, p.weights)
        self.key = key_pose(self.features, [self.key_frame])
        self.path_length = path_length(self.segment.wrist_pos[:, DOM])
        return self


def path_length(track: np.ndarray, smooth: int = 5) -> float:
    """Distance travelled along a [T, 3] wrist track after a moving average ('valid'), so jitter adds little.
    Used only to word movement hints (bigger / smaller)."""
    if len(track) >= smooth:
        kernel = np.ones(smooth) / smooth
        track = np.stack([np.convolve(track[:, i], kernel, mode="valid") for i in range(3)], axis=1)
    return float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())


@dataclass(frozen=True)
class Tolerances:
    finger: tuple            # [5] radians, thumb first
    finger_mean: bool
    spread: float            # 0 = not checked
    orientation_deg: float
    location: float
    movement: float          # 0 = not checked (static signs)
    other_hand: float        # 0 = not checked (one-handed signs)
    distance_z: float
    lookalike_guard: bool

    def to_json(self) -> dict:
        d = asdict(self)
        d["finger"] = [round(float(x), 4) for x in self.finger]
        return d

    @classmethod
    def from_json(cls, d: dict) -> Tolerances:
        return cls(**{**d, "finger": tuple(float(x) for x in d["finger"])})


def make_tolerances(tier: str, vp: VerifyParams, takes: list[Take], moving: bool, two_hands: bool) -> Tolerances:
    """Tier defaults, widened for the thumb, for dictionary-sourced references (fingers and spread), and for fingers that vary
    between Studio takes (k × their spread, capped). Watch-only signs get lenient values for display."""
    t = vp.tier("scored" if tier == "scored" else "lenient")
    base = np.full(5, t.finger)
    base[0] *= vp.thumb_scale
    source_scale = vp.dictionary_scale if any(k.source == "ksl_dictionary" for k in takes) else 1.0
    base *= source_scale
    finger = base
    if len(takes) >= 2:
        flex = np.stack([k.key.flexion for k in takes]).reshape(len(takes), 5, 3)
        spread = flex.std(axis=0).max(axis=1)
        finger = np.clip(vp.take_spread_k * spread, base, vp.take_spread_cap * base)
    return Tolerances(
        finger=tuple(float(x) for x in finger), finger_mean=t.finger_mean, spread=t.spread * source_scale,
        orientation_deg=t.orientation_deg, location=t.location,
        movement=t.movement if moving else 0.0, other_hand=t.other_hand if two_hands else 0.0,
        distance_z=t.distance_z, lookalike_guard=t.lookalike_guard)


@dataclass
class SignRef:
    id: str
    tier: str                # scored | lenient | watch
    takes: list[Take]
    tol: Tolerances
    tau: float = 1.0         # typical DTW distance of a correct attempt (from calibration)
    lookalikes: list[str] = field(default_factory=list)
    shadow_take: int = 0     # the take the app animates (the medoid)

    def bind(self, p: EngineParams) -> SignRef:
        for t in self.takes:
            t.bind(p)
        return self

    def closest(self, desc: np.ndarray, band: int) -> tuple[float, int]:
        """(distance, take index) of the take nearest to an attempt descriptor; the first take wins ties."""
        best = (np.inf, -1)
        for k, take in enumerate(self.takes):
            d = dtw(desc, take.descriptor, band)
            if d < best[0]:
                best = (d, k)
        return best

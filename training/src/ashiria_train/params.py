"""Engine parameters in one place (spec/RUNTIME_SPEC.md).

The defaults here are the reference values. `ashiria pack` writes them, with any calibrated values,
into every content pack, and the headset reads them from the pack. So a sideloaded pack can retune
the engine without rebuilding the app.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass

RUNTIME_VERSION = 1


@dataclass(frozen=True)
class SegmenterParams:
    """Cuts a live stream of frames into attempts (segmenter.py). Times in seconds at `tick_hz`."""
    tick_hz: float = 30.0
    rest_height: float = -1.5   # shoulder widths below the nose; set per user at onboarding
    start_s: float = 0.10       # hand up this long starts an attempt
    pre_roll_s: float = 0.20    # frames kept from before the start
    drop_s: float = 0.13        # hand down (or lost) this long ends an attempt
    hold_s: float = 0.6         # hand held still this long ends an attempt
    hold_radius: float = 0.12   # shoulder widths: the wrist counts as still within this of its median
    hold_shape: float = 0.35    # radians: mean bend deviation from the median that still counts as still
    min_attempt_s: float = 0.35  # shorter attempts (hand up) are dropped
    max_attempt_s: float = 4.0
    rearm_move: float = 0.3     # after a hold, moving the wrist this far (shoulder widths) re-arms

    def frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.tick_hz)))


@dataclass(frozen=True)
class TierTolerances:
    """How close an attempt must be to pass. A tolerance of 0 switches that check off."""
    finger: float           # radians, largest bend difference per finger
    finger_mean: bool       # true: average over fingers instead of the worst finger
    spread: float           # radians, largest difference between neighbouring fingers
    orientation_deg: float  # palm normal and hand direction
    location: float         # shoulder widths, wrist at the key frame (forward counts half)
    movement: float         # shoulder widths, mean distance between DTW-aligned wrist paths
    other_hand: float       # shoulder widths, the other wrist relative to the dominant one
    distance_z: float       # overall: DTW distance may be this many times the sign's tau
    lookalike_guard: bool   # fail when a look-alike sign matches clearly better


@dataclass(frozen=True)
class VerifyParams:
    min_present: float = 0.6     # dominant hand tracked in at least this share of frames
    thumb_scale: float = 1.4     # the thumb's base joint is placed differently by MediaPipe and OpenXR
    dictionary_scale: float = 1.25  # finger tolerances for references from 320×240 video
    take_spread_k: float = 2.5   # with several takes, a finger's tolerance grows to k × their spread...
    take_spread_cap: float = 2.0  # ...capped at this multiple of the tier tolerance
    lookalike_ratio: float = 0.9  # guard fires when a look-alike's distance < ratio × the target's
    scored: TierTolerances = field(default_factory=lambda: TierTolerances(
        finger=0.35, finger_mean=False, spread=0.30, orientation_deg=35.0, location=0.35,
        movement=0.15, other_hand=0.5, distance_z=1.3, lookalike_guard=True))
    # Lenient: about 1.4× wider, but still guarded against look-alikes. Without the guard, letters pass for
    # any hand in the right place (simulation, 14 Oct 2026: look-alikes passed 44% of the time).
    lenient: TierTolerances = field(default_factory=lambda: TierTolerances(
        finger=0.5, finger_mean=False, spread=0.45, orientation_deg=50.0, location=0.5,
        movement=0.22, other_hand=0.75, distance_z=1.6, lookalike_guard=True))

    def tier(self, name: str) -> TierTolerances:
        return self.scored if name == "scored" else self.lenient


@dataclass(frozen=True)
class RecogniseParams:
    min_present: float = 0.5
    accept_z: float = 1.25     # best sign's distance / its tau must be at most this
    min_margin_z: float = 0.15  # below this gap to the runner-up, show both ("ambiguous")
    temperature: float = 0.5   # softmax(-z / T) for the confidence shown to the user
    top: int = 5


DEFAULT_WEIGHTS = {"flexion": 1.0, "spread": 1.0, "tips": 0.5, "palm_normal": 1.0, "direction": 1.0,
                   "wrist": 3.0, "present": 1.0}


@dataclass(frozen=True)
class EngineParams:
    seq_len: int = 32
    gap_fill_frames: int = 5
    trim_margin_frames: int = 2
    active_fraction: float = 0.3
    active_min_range: float = 0.3
    active_margin: int = 2
    active_min_frames: int = 6
    dtw_band: int = 8
    # Provisional (14 Oct 2026): wrist 3.0 and orientation 1.0 raised simulated top-1 on the 72-sign
    # vocabulary from 82% to 90%. Re-tune on real Quest attempts.
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    segmenter: SegmenterParams = field(default_factory=SegmenterParams)
    verify: VerifyParams = field(default_factory=VerifyParams)
    recognise: RecogniseParams = field(default_factory=RecogniseParams)

    def to_json(self) -> dict:
        return {"runtime_version": RUNTIME_VERSION, **asdict(self)}

    @classmethod
    def from_json(cls, data: dict) -> EngineParams:
        return _build(cls, {k: v for k, v in data.items() if k != "runtime_version"})

    @classmethod
    def from_config(cls, cfg) -> EngineParams:
        """Defaults, then features.* (shared with training), then an optional `engine:` section."""
        f = cfg.get("features", {})
        base = {"seq_len": f.get("seq_len", 32), "gap_fill_frames": f.get("gap_fill_frames", 5),
                "trim_margin_frames": f.get("trim_margin_frames", 2)}
        return _build(cls, _merge(base, dict(cfg.get("engine") or {})))


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, data: dict):
    """Nested dataclass from a dict; missing keys keep their defaults, unknown keys are an error."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} parameters: {sorted(unknown)}")
    if any(f.default is MISSING and f.default_factory is MISSING for f in known.values()):
        return cls(**data)  # no defaults (TierTolerances): every value is given
    defaults = cls()
    kwargs = {}
    for name, value in data.items():
        current = getattr(defaults, name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[name] = _build(type(current), {**asdict(current), **value})
        elif name == "weights":
            kwargs[name] = {**current, **value}
        else:
            kwargs[name] = value
    return cls(**kwargs)

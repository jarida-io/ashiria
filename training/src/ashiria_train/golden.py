"""`ashiria golden`: engine test vectors from synthetic hands (no real people, so they can be committed).

Each case runs the shared runtime path (mirror → gap fill → trim → resample → features → descriptor)
and records the expected outputs, plus DTW distances and per-finger errors between cases. It also
records how attempts and references are cut (active window, held frames, steadiest frame). The C#
(and later Kotlin) ports must reproduce them within the stated tolerances.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from . import joints as J
from .canonical import active_window, gap_fill, hold_frames, mirror, resample, steadiest_frame, trim
from .config import TRAINING_DIR, Config
from .features import SPEC_VERSION, flexion, sequence_features
from .matching import descriptor, dtw, finger_errors
from .synthetic import random_segment

CASES = [
    {"name": "one_hand_static", "frames": 40, "two_hands": False, "moving": False, "fps": 30.0},
    {"name": "one_hand_moving_gaps", "frames": 55, "two_hands": False, "moving": True, "fps": 25.0,
     "gaps": [(0, 10, 3), (0, 30, 8)]},
    {"name": "two_hands_moving", "frames": 60, "two_hands": True, "moving": True, "fps": 30.0,
     "gaps": [(0, 5, 2), (1, 20, 4), (1, 40, 9)]},
    {"name": "two_hands_left_handed", "frames": 60, "two_hands": True, "moving": True, "fps": 30.0,
     "gaps": [(1, 12, 5)], "mirror": True},
    {"name": "rest_at_both_ends", "frames": 70, "two_hands": True, "moving": True, "fps": 24.0, "rest_frames": 12},
    {"name": "short_odd_length", "frames": 23, "two_hands": False, "moving": True, "fps": 30.0},
]


def _arr(a: np.ndarray, digits: int = 7) -> list:
    return np.round(np.asarray(a, np.float64), digits).tolist()


def build_golden(cfg: Config) -> str:
    f = cfg.features
    seq_len, max_gap, rest, margin = int(f.seq_len), int(f.gap_fill_frames), float(f.rest_height), int(f.trim_margin_frames)
    rng = np.random.default_rng(2026)
    cases, descriptors, key_flexion = [], [], []
    for spec in CASES:
        seg = random_segment(rng, spec["frames"], spec["two_hands"], spec["moving"], spec["fps"],
                             tuple(spec.get("gaps", ())), spec.get("rest_frames", 0))
        work = mirror(seg) if spec.get("mirror") else seg
        work = trim(gap_fill(work, max_gap), rest, margin)
        res = resample(work, seq_len)
        feats = sequence_features(res.hand_rel, res.wrist_pos, res.present, 1.0 / res.fps)
        desc = descriptor(feats)
        descriptors.append(desc)
        key = seq_len // 2
        key_flexion.append(flexion(res.hand_rel[key, J.RIGHT]))
        win = active_window(work)
        win_res = resample(win, seq_len)
        cases.append({
            "name": spec["name"],
            "mirror": bool(spec.get("mirror", False)),
            "segment": {"fps": seg.fps, "hand_rel": _arr(seg.hand_rel), "wrist_pos": _arr(seg.wrist_pos),
                        "present": seg.present.tolist()},
            "trimmed_length": len(work),
            "resampled_fps": res.fps,
            "resampled_present": res.present.tolist(),
            "features": _arr(feats),
            "descriptor": _arr(desc),
            "key_frame": key,
            "key_flexion_dominant": _arr(key_flexion[-1]),
            "window_length": len(win),
            "window_hold": hold_frames(win).tolist(),
            "window_steadiest": steadiest_frame(win_res),
        })

    n = len(cases)
    pairs = [{"a": i, "b": j, "distance": dtw(descriptors[i], descriptors[j])} for i in range(n) for j in range(i + 1, n)]
    errors = [{"attempt": i, "reference": j, "errors": finger_errors(key_flexion[i], key_flexion[j]).tolist()}
              for i in range(n) for j in range(n) if i != j][:12]
    golden = {
        "spec_version": SPEC_VERSION,
        "seq_len": seq_len,
        "gap_fill_frames": max_gap,
        "rest_height": rest,
        "trim_margin_frames": margin,
        "tolerance": {"features": 1e-3, "descriptor": 1e-3, "dtw": 1e-3, "finger_errors": 2e-3},
        "cases": cases,
        "dtw": pairs,
        "finger_errors": errors,
    }
    out = TRAINING_DIR.parent / "spec" / "golden" / f"engine_v{SPEC_VERSION}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden))
    cut = sum(c["window_length"] < c["trimmed_length"] for c in cases)
    print(f"active window shortened {cut}/{len(cases)} cases")
    print(f"{len(cases)} cases, {len(pairs)} DTW pairs -> {out} ({out.stat().st_size / 1e3:.0f} KB)")
    return str(out)


# --- runtime golden vectors: prepare, Quest frames, segmenter, Verify, Recognise, a mini content pack ----------

def _rounded(seg, digits: int = 6):
    """Inputs are written rounded; expectations are computed from exactly what the ports will read."""
    from .canonical import Segment
    return Segment(np.round(seg.hand_rel, digits), np.round(seg.wrist_pos, digits), seg.present.copy(), seg.fps)


def _seg_json(seg) -> dict:
    return {"fps": seg.fps, "hand_rel": seg.hand_rel.tolist(), "wrist_pos": seg.wrist_pos.tolist(),
            "present": seg.present.tolist()}


def _mini_catalogue_and_sources(p):
    from .augment import rotation
    from .canonical import Segment, steadiest_frame
    from .pipeline import cut_reference
    from .references import tag
    from .signs import Take
    from .store import SignSource
    from .synthetic import sign_stream

    fist = np.full((5, 3), 1.2)
    point = fist.copy()
    point[1] = 0.0
    vee = point.copy()
    vee[2] = 0.0
    specs = [("flat", np.zeros((5, 3)), {}, "auto"), ("fist", fist, {}, "lenient"), ("point", point, {}, "auto"),
             ("vee", vee, {"orient": rotation(10, 0, 0)}, "scored"),
             ("wave", np.zeros((5, 3)), {"path": (0.8, 0.0, 0.0)}, "auto"),
             ("both", fist, {"other_at": (-0.3, -1.0, 0.6)}, "watch")]
    catalogue = {"pack_id": "golden", "version": "1.0.0", "units": [{"id": "basics", "title": "Basics", "order": 1}],
                 "signs": []}
    sources, bends = {}, {}
    for k, (sid, b, kw, tier) in enumerate(specs, 1):
        stream = sign_stream(b, **kw)
        takes = []
        for n, shift in enumerate((0.0, 0.03)):  # two takes per sign, the second slightly higher
            moved = Segment(stream.hand_rel, stream.wrist_pos + [0.0, shift, 0.0], stream.present, stream.fps)
            seg = cut_reference(moved, p, p.segmenter.rest_height)
            takes.append(Take(seg, steadiest_frame(seg), "studio", take_id=f"{sid}-{n + 1}"))
        sources[sid] = SignSource("studio", takes, tag(stream), ["G01"])
        catalogue["signs"].append({"id": sid, "gloss": sid, "unit": "basics", "order": k, "tier": tier,
                                   "tier_note": "golden" if tier != "auto" else "", "review": {"status": "approved"}})
        bends[sid] = (b, kw)
    return catalogue, sources, bends


def _far_from_one(values, margin: float = 0.05) -> bool:
    return all(abs(v - 1.0) > margin for v in values)


def build_runtime_golden(cfg: Config) -> str:
    from .augment import rotation
    from .pack import take_segment, write_pack
    from .params import EngineParams
    from .pipeline import prepare
    from .quest import frame, head_axes, to_world
    from .recognise import recognise
    from .segmenter import segment_stream
    from .signs import SignRef, Take, Tolerances
    from .synthetic import sign_stream
    from .verify import verify

    p = EngineParams.from_config(cfg)
    rest = p.segmenter.rest_height
    root = TRAINING_DIR.parent / "spec" / "golden"
    flat, fist = np.zeros((5, 3)), np.full((5, 3), 1.2)

    # The mini pack (synthetic signs only, so it can be committed). Build metadata is pinned for stable diffs.
    catalogue, sources, bends = _mini_catalogue_and_sources(p)
    pack_dir = write_pack(catalogue, sources, [], {}, p, dict(cfg.calibrate.augment), root, attempts=6)
    final = pack_dir.parent / "pack_v1"
    if final.exists():
        for f in final.iterdir():
            f.unlink()
        final.rmdir()
    pack_dir.rename(final)
    manifest = json.loads((final / "manifest.json").read_text())
    manifest["build"] = {"created": "2026-01-01T00:00:00Z", "git": None, "dirty": None}
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    pp = EngineParams.from_json(manifest["engine"])
    refs_json = json.loads((final / "references.json").read_text())["signs"]
    cat_json = {s["id"]: s for s in json.loads((final / "catalogue.json").read_text())["signs"]}
    refs = {sid: SignRef(sid, cat_json[sid]["tier"],
                         [Take(take_segment(t), t["key_frame"], t["source"], t["take_id"]) for t in r["takes"]],
                         Tolerances.from_json(r["tol"]), r["tau"], cat_json[sid]["lookalikes"], r["shadow_take"]).bind(pp)
            for sid, r in refs_json.items()}

    # prepare
    prepare_cases = []
    for name, kw in [("static", {}), ("moving", {"path": (0.6, 0.3, 0.0)}), ("two_hands", {"other_at": (-0.3, -0.9, 0.5)}),
                     ("gaps", {"lost": ((25, 3),)}), ("low_rest", {"rest_y": -1.7, "rest_frames": 4})]:
        raw = _rounded(sign_stream(fist if name == "two_hands" else flat, **kw))
        out = prepare(raw, pp, rest)
        prepare_cases.append({"name": name, "segment": _seg_json(raw), "rest_height": rest,
                              "fps": out.fps, "present": out.present.tolist(),
                              "descriptor": _arr(descriptor(sequence_features(out.hand_rel, out.wrist_pos, out.present,
                                                                              1.0 / out.fps), pp.weights))})

    # Quest frames
    rng = np.random.default_rng(11)
    quest_cases = []
    for k, (left_handed, lost, look_up) in enumerate([(False, None, False), (True, None, False), (False, "left", False),
                                                      (True, "right", True)]):
        head = np.round(rng.uniform([-1, 1.2, -1], [1, 1.7, 1]), 4)
        yaw = rng.uniform(-np.pi, np.pi)
        forward = np.array([0.0, 1.0, 0.0]) if look_up else np.array([np.sin(yaw), -0.3, np.cos(yaw)])
        last_axes, _ = head_axes(np.array([1.0, 0.0, 0.2]))
        R = last_axes if look_up else head_axes(forward)[0]
        hands = {}
        for side in ("right", "left"):
            rel = rng.normal(0, 0.04, (21, 3))
            rel[0] = 0
            joints = np.round(to_world(rel, rng.normal([0, -1, 0.5], 0.3), head, R, 0.4, False), 5)
            hands[side] = {"joints": joints, "tracked": side != lost, "high_confidence": True, "inferred": False,
                           "system_gesture": False}
        hand_rel, wrist, present, _ = frame(head, forward, hands, left_handed, 0.4, last_axes)
        quest_cases.append({"left_handed": left_handed, "shoulder_width": 0.4, "head_pos": head.tolist(),
                            "head_forward": forward.tolist(), "last_axes": last_axes.tolist(),
                            "hands": {s: {"joints": h["joints"].tolist(), "tracked": h["tracked"]} for s, h in hands.items()},
                            "hand_rel": hand_rel.tolist(), "wrist_pos": wrist.tolist(), "present": present.tolist()})

    # Segmenter streams (default segmenter parameters; expectations are exact integers).
    def wiggle(seg):
        seg.wrist_pos[:, 0, 0] += 0.4 * np.sin(np.arange(len(seg)) / 5.0)
        return seg

    streams = [("hold", sign_stream(flat, hold=40)), ("moving_drop", sign_stream(flat, path=(0.6, 0, 0), hold=30)),
               ("lost", sign_stream(flat, path=(0.9, 0, 0), hold=30, lost=((26, 10),))),
               ("below_rest", sign_stream(flat, at=(0.4, -2.0, 0.6))), ("long_hold", sign_stream(fist, hold=100)),
               ("timeout", wiggle(sign_stream(flat, hold=150)))]
    seg_cases = []
    for name, s in streams:
        s = _rounded(s)
        seg_cases.append({"name": name, "segment": _seg_json(s),
                          "attempts": [{"start": a.start, "end": a.end, "reason": a.reason} for a in segment_stream(pp.segmenter, s)]})

    # Verify and Recognise on prepared attempts against the mini pack.
    look = {sid: [refs[x] for x in r.lookalikes] for sid, r in refs.items()}
    bent = flat.copy()
    bent[3] += 0.9
    point = fist.copy()
    point[1] = 0.0
    verify_specs = [("flat_ok", flat, {}, "flat", 1.0), ("ring_bent", bent, {}, "flat", 1.0),
                    ("turned", flat, {"orient": rotation(70, 0, 0)}, "flat", 1.0),
                    ("too_high", flat, {"at": (0.4, -0.3, 0.6)}, "flat", 1.0),
                    ("no_movement", flat, {}, "wave", 1.0), ("wave_ok", flat, {"path": (0.8, 0, 0)}, "wave", 1.0),
                    ("point_for_fist", point, {}, "fist", 1.0), ("watch_only", fist, {"other_at": (-0.3, -1.0, 0.6)}, "both", 1.0),
                    ("lenient_scale", bent * 0.5, {}, "flat", 1.6)]
    verify_cases = []
    for name, b, kw, sid, scale in verify_specs:
        att = _rounded(prepare(sign_stream(b, **kw), pp, rest))
        r = verify(att, refs[sid], pp, scale, look[sid])
        if r.components and not _far_from_one(r.components.values()):
            raise SystemExit(f"golden verify case {name} sits on a threshold: {r.components}")
        verify_cases.append({"name": name, "sign": sid, "tolerance_scale": scale, "attempt": _seg_json(att),
                             "status": r.status, "score": r.score, "distance": r.distance if r.take >= 0 else None,
                             "take": r.take, "components": r.components, "finger_errors": r.finger_errors,
                             "hints": [h.to_json() for h in r.hints]})
    none = _rounded(prepare(sign_stream(flat), pp, rest))
    none.present[:] = 0
    r = verify(none, refs["flat"], pp)
    verify_cases.append({"name": "no_hands", "sign": "flat", "tolerance_scale": 1.0, "attempt": _seg_json(none),
                         "status": r.status, "score": r.score, "distance": None, "take": -1, "components": {},
                         "finger_errors": [], "hints": [h.to_json() for h in r.hints]})

    candidates = [refs[s] for s in refs if cat_json[s]["recognise"]]
    recog_cases = []
    for name, b, kw, override in [("match_vee", None, {}, None), ("match_wave", flat, {"path": (0.8, 0, 0)}, None),
                                  ("not_sure", np.full((5, 3), 0.6), {"at": (-0.3, -0.2, 0.9), "orient": rotation(0, 70, 0)}, None),
                                  ("ambiguous", None, {}, (4.0, 3.0))]:
        if b is None:
            vee = point.copy()
            vee[2] = 0.0
            b, kw = vee, {"orient": rotation(10, 0, 0)}
        att = _rounded(prepare(sign_stream(b, **kw), pp, rest))
        params = pp if override is None else dataclasses.replace(pp, recognise=dataclasses.replace(
            pp.recognise, accept_z=override[0], min_margin_z=override[1]))
        r = recognise(att, candidates, params)
        rp = params.recognise
        if abs(r.best_z - rp.accept_z) < 0.05 or (np.isfinite(r.margin) and abs(r.margin - rp.min_margin_z) < 0.05):
            raise SystemExit(f"golden recognise case {name} sits on a threshold")
        recog_cases.append({"name": name, "attempt": _seg_json(att), "accept_z": rp.accept_z, "min_margin_z": rp.min_margin_z,
                            "status": r.status,
                            "best": r.best, "runner_up": r.runner_up, "best_z": r.best_z, "runner_up_z": r.runner_up_z,
                            "confidence": r.confidence, "ranking": [list(x) for x in r.ranking]})

    golden = {"spec_version": SPEC_VERSION, "pack": "pack_v1",
              "tolerance": {"descriptor": 1e-3, "quest": 1e-4, "verify": 2e-3, "recognise": 2e-3},
              "prepare": prepare_cases, "quest": quest_cases, "segmenter": seg_cases,
              "verify": verify_cases, "recognise": recog_cases}
    out = root / f"runtime_v{SPEC_VERSION}.json"
    out.write_text(json.dumps(golden))
    print(f"runtime golden: {len(prepare_cases)} prepare, {len(quest_cases)} quest, {len(seg_cases)} segmenter, "
          f"{len(verify_cases)} verify, {len(recog_cases)} recognise -> {out} ({out.stat().st_size / 1e3:.0f} KB); "
          f"mini pack -> {final}")
    return str(out)


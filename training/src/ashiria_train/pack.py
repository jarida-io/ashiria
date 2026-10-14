"""`ashiria pack`: the content pack the headset loads (spec/CONTENT_PACK.md).

content/packs/<pack_id>-<version>/ (git-ignored) holds three JSON files:
  manifest.json    format, versions, rights, engine parameters (calibrated), file hashes, a self-check
  catalogue.json   units and signs: what to teach, tiers and their reasons, hints, look-alikes
  references.json  per sign: reference takes (hand poses only), tolerances, tau

Packs never contain video, images or the KSL Dictionary's text, and only pseudonymous signer ids.
`--release` ships only signs with distributable references (Studio takes, or dictionary drafts once the
owners' written permission is recorded in content/rights.yaml), and only signs the team has approved.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .calibrate import build_refs, calibrate
from .catalogue import catalogue_file, load_catalogue
from .config import Config, TRAINING_DIR, training_relative
from .features import SPEC_VERSION
from .matching import segment_descriptor
from .params import RUNTIME_VERSION, EngineParams
from .store import DICTIONARY, STUDIO, assemble, distractor_segments, load_dictionary, load_studio, segment_from_json

FORMAT = "ashiria.content-pack"
FORMAT_VERSION = 1
FORBIDDEN_KEYS = {"video", "videos", "image", "images", "frame_png", "thumbnail", "clip", "clip_id", "url",
                  "description", "note", "notes", "name", "email"}
TAG_KEYS = ("hands", "movement", "near_face", "hands_close")


def content_dir(cfg: Config) -> Path:
    return training_relative(cfg.paths.get("content", "../content"))


def load_rights(cfg: Config) -> dict:
    file = content_dir(cfg) / "rights.yaml"
    return yaml.safe_load(file.read_text()) if file.exists() else {}


def _round(a: np.ndarray, digits: int) -> list:
    return np.round(np.asarray(a, np.float64), digits).reshape(-1).tolist()


def take_json(t, signer: str) -> dict:
    seg = t.segment
    return {"source": t.source, "take_id": t.take_id, "signer": signer, "fps": round(float(seg.fps), 4),
            "frames": len(seg), "key_frame": int(t.key_frame),
            "hand_rel": _round(seg.hand_rel, 5),    # [frames, 2, 21, 3] metres, flattened
            "wrist_pos": _round(seg.wrist_pos, 4),  # [frames, 2, 3] shoulder widths, flattened
            "present": _round(seg.present, 0)}      # [frames, 2]


def take_segment(t: dict):
    n = int(t["frames"])
    return segment_from_json({"hand_rel": np.reshape(t["hand_rel"], (n, 2, 21, 3)),
                              "wrist_pos": np.reshape(t["wrist_pos"], (n, 2, 3)),
                              "present": np.reshape(t["present"], (n, 2)), "fps": t["fps"]})


def forbidden_keys(obj, where: str = "") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                found.append(f"{where}/{k}")
            found += forbidden_keys(v, f"{where}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            found += forbidden_keys(v, f"{where}[{i}]")
    return found


def _dump(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _git() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=TRAINING_DIR, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=TRAINING_DIR, capture_output=True,
                                    text=True).stdout.strip())
        return {"git": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git": None, "dirty": None}


def release_selection(catalogue: dict, sources: dict, missing: list[str], rights: dict
                      ) -> tuple[dict, dict[str, str], list[str]]:
    """What a release pack may contain: (sources to ship, {left-out id: why}, blocking problems).

    Signs with no reference, or whose only reference is a KSL Dictionary draft without written permission in
    content/rights.yaml, are left out automatically, so a release ships whatever the team has recorded in
    Studio. Shipped signs must be approved (review.status) by the team."""
    allowed = (rights.get("ksl_dictionary") or {}).get("permission") == "written"
    left_out = {sid: "no reference yet" for sid in missing}
    ship = {}
    for sid, src in sources.items():
        if src.kind == "ksl_dictionary" and not allowed:
            left_out[sid] = "KSL Dictionary draft only (record it in Studio)"
        else:
            ship[sid] = src
    problems = []
    unapproved = [e["id"] for e in catalogue["signs"]
                  if e["id"] in ship and (e.get("review") or {}).get("status") != "approved"]
    if unapproved:
        problems.append(f"{len(unapproved)} signs not approved yet (set review.status: approved in the catalogue), "
                        "e.g. " + ", ".join(unapproved[:6]))
    if len(ship) < 2:
        problems.append(f"only {len(ship)} signs have distributable references: record signs in Studio first")
    return ship, left_out, problems


def build_pack(cfg: Config, pack_id: str = "ksl-core", release: bool = False, attempts: int = 16) -> Path:
    catalogue = load_catalogue(catalogue_file(cfg, pack_id))
    sources, missing = assemble(catalogue, load_dictionary(DICTIONARY), load_studio(STUDIO))
    distractors = distractor_segments(Path(DICTIONARY).parent / "distractors.npz")
    return write_pack(catalogue, sources, missing, load_rights(cfg), EngineParams.from_config(cfg),
                      dict(cfg.calibrate.augment), content_dir(cfg) / "packs", release, attempts, distractors)


def write_pack(catalogue: dict, sources: dict, missing: list[str], rights: dict, p: EngineParams, aug: dict,
               out_root: Path, release: bool = False, attempts: int = 16, distractors=()) -> Path:
    pack_id = catalogue["pack_id"]
    if release:
        sources, left_out, problems = release_selection(catalogue, sources, missing, rights)
        if problems:
            raise SystemExit("Not releasable:\n  " + "\n  ".join(problems))
        if left_out:
            print(f"Release leaves out {len(left_out)} signs: "
                  + "; ".join(f"{sid} ({why})" for sid, why in list(left_out.items())[:8])
                  + ("…" if len(left_out) > 8 else ""))
        missing = []
    if len(sources) < 2:
        raise SystemExit("Need references for at least two catalogue signs.")
    if missing:
        print(f"Left out (no reference yet): {', '.join(missing)}")

    report, calibrated = calibrate(catalogue, sources, p, aug, distractors, attempts)
    info = report["signs"]
    ids = [e["id"] for e in catalogue["signs"] if e["id"] in sources]
    refs = build_refs(ids, sources, calibrated, {s: info[s]["tier"] for s in ids},
                      {s: info[s]["tau"] for s in ids}, {s: info[s]["lookalikes"] for s in ids})
    entries = {e["id"]: e for e in catalogue["signs"]}

    cat_json = {"pack_id": pack_id, "units": catalogue["units"], "signs": []}
    ref_json = {"seq_len": calibrated.seq_len, "signs": {}}
    for sid in ids:
        e, src, ref = entries[sid], sources[sid], refs[sid]
        cat_json["signs"].append({
            "id": sid, "gloss": e["gloss"], "aliases": e.get("aliases") or [], "unit": e["unit"], "order": e["order"],
            "tier": ref.tier, "tier_source": info[sid]["tier_source"], "tier_reasons": info[sid]["tier_reasons"],
            "recognise": e["recognise"] if e.get("recognise") is not None else ref.tier != "watch",
            "hint": e.get("hint") or "", "facial_grammar": bool(e.get("facial_grammar")),
            "tags": {k: src.tags.get(k) for k in TAG_KEYS}, "lookalikes": ref.lookalikes,
            "source": {"kind": src.kind, "takes": len(src.takes), "signers": src.signers},
            "review": e.get("review") or {"status": "draft"},
        })
        signer = src.signers[0] if src.signers else ""
        ref_json["signs"][sid] = {"takes": [take_json(t, signer) for t in ref.takes], "shadow_take": ref.shadow_take,
                                  "tau": round(ref.tau, 4), "tol": ref.tol.to_json()}

    bad = forbidden_keys(cat_json) + forbidden_keys(ref_json)
    if bad:
        raise SystemExit(f"Refusing to write a pack with forbidden fields: {bad[:5]}")
    cat_bytes, ref_bytes = _dump(cat_json), _dump(ref_json)

    # Self-check: the headset recomputes this from the file as written (after rounding) and compares.
    first = ref_json["signs"][ids[0]]["takes"][0]
    check = float(segment_descriptor(take_segment(json.loads(_dump(first))), calibrated.weights).sum())
    dict_rights = rights.get("ksl_dictionary") or {}
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "pack_id": pack_id, "version": catalogue["version"],
        "spec_version": SPEC_VERSION, "runtime_version": RUNTIME_VERSION,
        "distribution": "release" if release else "internal",
        "build": {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **_git()},
        "rights": {"sources": sorted({s.kind for s in sources.values()}),
                   "ksl_dictionary": {"permission": dict_rights.get("permission", "none"),
                                      "reference": dict_rights.get("reference") or ""},
                   "studio": {"consent_form": (rights.get("studio") or {}).get("consent_form") or ""}},
        "engine": calibrated.to_json(),
        "calibration": {"method": "simulated", "attempts_per_sign": report["attempts_per_sign"],
                        "rates": report["rates"]["pooled"]},
        "files": {"catalogue": {"path": "catalogue.json", "sha256": hashlib.sha256(cat_bytes).hexdigest()},
                  "references": {"path": "references.json", "sha256": hashlib.sha256(ref_bytes).hexdigest()}},
        "content_hash": hashlib.sha256(cat_bytes + ref_bytes).hexdigest(),
        "selfcheck": {"sign_id": ids[0], "take": 0, "descriptor_sum": round(check, 4), "tolerance": 1e-2},
    }
    out = out_root / f"{pack_id}-{catalogue['version']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalogue.json").write_bytes(cat_bytes)
    (out / "references.json").write_bytes(ref_bytes)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    tiers = [s["tier"] for s in cat_json["signs"]]
    print(f"{pack_id} {catalogue['version']} ({manifest['distribution']}): {len(ids)} signs "
          f"({tiers.count('scored')} scored, {tiers.count('lenient')} lenient, {tiers.count('watch')} watch-only), "
          f"{size:.1f} MB -> {out}")
    if not release:
        print("Internal pack, for development. For the hosted app, approve signs (`ashiria approve`) and build "
              "with --release.")
    return out

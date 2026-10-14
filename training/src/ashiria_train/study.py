"""`ashiria study-data`: what the dev-only study sheet shows (apps/quest-learn, /dev/study).

For each catalogue sign: its review status and tier, the local KSL Dictionary video it came from (development
machine only; never served by the app), and the reference take as plain joint positions for a preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue import catalogue_file, load_catalogue
from .config import Config, load_config, path, training_relative
from .store import DICTIONARY, STUDIO, assemble, load_dictionary, load_studio

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"


def _video_paths() -> dict[str, str]:
    """Dictionary vocabulary word -> absolute path of its local video."""
    refs_file = training_relative(DICTIONARY)
    if not refs_file.exists():
        return {}
    dcfg = load_config(DICTIONARY_CONFIG)
    manifest = path(dcfg, "manifests") / "videos.csv"
    if not manifest.exists():
        return {}
    paths = pd.read_csv(manifest, usecols=["clip_id", "path"], dtype=str).set_index("clip_id")["path"]
    raw = path(dcfg, "raw_videos")
    out = {}
    for s in json.loads(refs_file.read_text())["signs"]:
        rel = paths.get(s["clip_id"])
        if isinstance(rel, str) and (raw / rel).exists():
            out[s["word"]] = str((raw / rel).resolve())
    return out


def study_data(cfg: Config, pack_id: str = "ksl-core") -> dict:
    catalogue = load_catalogue(catalogue_file(cfg, pack_id))
    sources, _ = assemble(catalogue, load_dictionary(DICTIONARY), load_studio(STUDIO))
    videos = _video_paths()
    units = {u["id"]: u["title"] for u in catalogue["units"]}
    signs = []
    for e in catalogue["signs"]:
        src = sources.get(e["id"])
        take = src.takes[src.shadow_take] if src else None
        word = (e.get("sources") or {}).get("dictionary")
        signs.append({
            "id": e["id"], "gloss": e["gloss"], "unit": units.get(e["unit"], e["unit"]), "tier": e.get("tier", "auto"),
            "tier_note": e.get("tier_note", ""), "review": (e.get("review") or {}).get("status", "draft"),
            "source": src.kind if src else None, "two_handed": bool(src and src.tags.get("hands") == "two"),
            "video": videos.get(word) if word else None,
            "take": None if take is None else {
                "fps": round(float(take.segment.fps), 3), "frames": len(take.segment),
                "hand_rel": np.round(take.segment.hand_rel, 4).ravel().tolist(),
                "wrist_pos": np.round(take.segment.wrist_pos, 3).ravel().tolist(),
                "present": take.segment.present.ravel().tolist()},
        })
    return {"pack_id": catalogue["pack_id"], "version": catalogue["version"], "signs": signs}


def video_roots() -> list[Path]:
    """Folders the dev server may stream videos from."""
    return [path(load_config(DICTIONARY_CONFIG), "raw_videos").resolve(), training_relative("data").resolve()]


def dictionary_index() -> list[dict]:
    """Every extracted KSL Dictionary clip for the dev-only sign browser: id, sign, aliases, category, description."""
    dcfg = load_config(DICTIONARY_CONFIG)
    manifest = path(dcfg, "manifests") / "videos.csv"
    landmarks = path(dcfg, "landmarks")
    if not manifest.exists():
        return []
    df = pd.read_csv(manifest, dtype=str).fillna("")
    raw = path(dcfg, "raw_videos")
    out = []
    for _, r in df.iterrows():
        if not (landmarks / f"{r['clip_id']}.npz").exists() or not (raw / r["path"]).exists():
            continue
        out.append({"clip_id": r["clip_id"], "sign": r["label"], "aliases": r.get("aliases", ""),
                    "category": r.get("category", ""), "description": r.get("description", ""),
                    "video": str((raw / r["path"]).resolve())})
    return sorted(out, key=lambda x: x["sign"])


def clip_landmarks(clip_id: str) -> dict:
    """MediaPipe image landmarks of one clip (x, y in 0..1 of the frame), frame k at k / fps seconds."""
    if not clip_id.isalnum():
        raise ValueError("bad clip id")
    file = path(load_config(DICTIONARY_CONFIG), "landmarks") / f"{clip_id}.npz"
    with np.load(file) as d:
        return {"fps": float(d["fps"]),
                "hand": np.round(d["hand_image"][..., :2], 4).tolist(),        # [T][2 slots: right, left][21][x, y]
                "hand_present": d["hand_present"].astype(int).tolist(),
                "hand_score": np.round(d["hand_score"], 2).tolist(),
                "pose": np.round(d["pose_image"][:, :25, :2], 4).tolist(),     # upper body only
                "pose_present": d["pose_present"].astype(int).tolist()}

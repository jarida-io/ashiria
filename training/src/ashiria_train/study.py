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

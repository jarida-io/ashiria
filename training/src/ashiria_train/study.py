"""`ashiria study-data`: what the dev-only study sheet shows (apps/quest-learn, /dev/study).

For each catalogue sign: its review status and tier, the local KSL Dictionary video it came from (development
machine only; never served by the app), and the reference take as plain joint positions for a preview.
`ashiria study-index`, `study-landmarks` and `study-reference` feed the sign browser (/dev/study/browse): every sign
in the dictionary's data sheet, with MediaPipe landmarks and the reference Ashiria would cut from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue import catalogue_file, load_catalogue
from .config import Config, load_config, path, training_relative
from .dictionary import folder_index, resolve_copy
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
    """Every sign in the KSL Dictionary data sheet, for the dev-only sign browser.

    One entry per sign name and video: names that share a video each get their own entry (with `same_video` listing
    the others), and a name with several videos gets one entry per video. `clip_id` is None when the scrape has no
    usable copy of the video. `flags` come from the last `ashiria audit-dictionary`.
    """
    dcfg = load_config(DICTIONARY_CONFIG)
    manifest, sheet = path(dcfg, "manifests") / "videos.csv", training_relative(dcfg.dictionary["source_csv"])
    if not manifest.exists() or not sheet.exists():
        return []
    landmarks, raw = path(dcfg, "landmarks"), path(dcfg, "raw_videos")
    m = pd.read_csv(manifest, dtype=str).fillna("")
    # `ashiria import-dictionary` keeps one saved copy per video; find it among each video's copies.
    clip_by_path = {r["path"]: r for _, r in m[m["label"] != ""].iterrows()}
    folders = folder_index(raw)
    v1 = {c: w for w, c in _v1_clips().items()}
    from .dictionary_audit import audit_flags
    flags = audit_flags()

    df = pd.read_csv(sheet, dtype=str).fillna("")
    df = df[df["sign"].str.strip() != ""].copy()
    df["n"] = pd.to_numeric(df["number"], errors="coerce").fillna(10**9).astype(int)
    df = df.sort_values("n")
    out = []
    for src, group in df.groupby("video_src", sort=False):
        copies = (resolve_copy(raw, folders, p, n) for p, n in zip(group["video_path"], group["n"]))
        clip = next((clip_by_path[c] for c in copies if c in clip_by_path), None) if src else None
        ok = clip is not None and (landmarks / f"{clip['clip_id']}.npz").exists() and (raw / clip["path"]).exists()
        names = list(dict.fromkeys(s.strip() for s in group["sign"]))
        for name in names:
            rows = group[group["sign"].str.strip() == name]
            out.append({
                "sign": name, "category": rows["category"].mode().iloc[0] if len(rows) else "",
                "description": next((d for d in rows["description"] if d), "") or next((d for d in group["description"] if d), ""),
                "note": next((x for x in rows["note"] if x), ""),
                "same_video": [x for x in names if x != name],
                "clip_id": clip["clip_id"] if ok else None,
                "video": str((raw / clip["path"]).resolve()) if ok else None,
                "v1": v1.get(clip["clip_id"]) if ok else None,
                "flags": flags.get(clip["clip_id"] if ok else src, [] if ok else ["broken"]),
            })
    # Number the videos of names that have more than one, so the list can tell them apart.
    count: dict[str, int] = {}
    for e in out:
        count[e["sign"].lower()] = count.get(e["sign"].lower(), 0) + 1
    seen: dict[str, int] = {}
    for e in sorted(out, key=lambda e: (e["clip_id"] is None, e["sign"].lower())):
        k = e["sign"].lower()
        seen[k] = seen.get(k, 0) + 1
        e["variant"] = f"{seen[k]} of {count[k]}" if count[k] > 1 else ""
    return sorted(out, key=lambda e: (e["sign"].lower(), e["variant"]))


def _v1_clips() -> dict[str, str]:
    """v1 vocabulary word -> the dictionary clip its reference came from."""
    refs_file = training_relative(DICTIONARY)
    if not refs_file.exists():
        return {}
    return {s["word"]: s["clip_id"] for s in json.loads(refs_file.read_text())["signs"]}


def clip_reference(clip_id: str) -> dict | None:
    """One dictionary clip cut into a reference the way Ashiria cuts every reference (segmenter, then prepare), as
    plain joint positions for the browser's preview. None if the clip has no usable hand."""
    from .canonical import from_mediapipe, gap_fill, trim
    from .params import EngineParams
    from .pipeline import cut_reference

    if not clip_id.isalnum():
        raise ValueError("bad clip id")
    dcfg = load_config(DICTIONARY_CONFIG)
    with np.load(path(dcfg, "landmarks") / f"{clip_id}.npz") as d:
        seg, _ = from_mediapipe(d)
    f = dcfg.features
    if seg is not None:
        seg = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
    ref = cut_reference(seg, EngineParams.from_config(dcfg), f.rest_height) if seg is not None and len(seg) >= 4 else None
    if ref is None:
        return None
    return {"fps": round(float(ref.fps), 3), "frames": len(ref), "two_handed": bool(ref.present[:, 1].mean() > 0.3),
            "hand_rel": np.round(ref.hand_rel, 4).ravel().tolist(), "wrist_pos": np.round(ref.wrist_pos, 3).ravel().tolist(),
            "present": ref.present.ravel().tolist()}


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

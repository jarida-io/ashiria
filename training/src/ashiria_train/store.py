"""Where each catalogue sign's reference takes come from.

Two stores, both local and git-ignored:
- the KSL Dictionary drafts: exports/dictionary/references/references.json (`ashiria references`)
- Studio takes recorded on Quest: exports/studio/store.json (`ashiria import-studio`)

Per sign, kept Studio takes replace the dictionary draft entirely; the two are never mixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .canonical import Segment
from .config import training_relative
from .signs import Take

DICTIONARY = "exports/dictionary/references/references.json"
STUDIO = "exports/studio/store.json"


@dataclass
class SignSource:
    kind: str                  # studio | ksl_dictionary
    takes: list[Take]
    tags: dict
    signers: list[str] = field(default_factory=list)
    shadow_take: int = 0


def segment_from_json(d: dict) -> Segment:
    return Segment(np.asarray(d["hand_rel"], np.float64), np.asarray(d["wrist_pos"], np.float64),
                   np.asarray(d["present"], np.float64), float(d["fps"]))


def load_dictionary(file: str | Path = DICTIONARY) -> dict[str, SignSource]:
    """Dictionary drafts keyed by vocabulary word."""
    file = training_relative(file)
    if not file.exists():
        return {}
    out = {}
    for s in json.loads(file.read_text())["signs"]:
        take = Take(segment_from_json(s), int(s["key_frame"]), "ksl_dictionary", take_id=f"dict-{s['clip_id']}")
        # The same take as other camera viewpoints would see it (references.VIEW_TAKES); the shadow stays the first.
        views = [Take(segment_from_json(v), int(v["key_frame"]), "ksl_dictionary", take_id=f"dict-{s['clip_id']}-view{k}")
                 for k, v in enumerate(s.get("views", []))]
        out[s["word"]] = SignSource("ksl_dictionary", [take, *views], s["tags"], ["ksl-dictionary"])
    return out


def load_studio(file: str | Path = STUDIO) -> dict[str, SignSource]:
    """Kept Studio takes keyed by sign id (format written by studio.py)."""
    file = training_relative(file)
    if not file.exists():
        return {}
    out = {}
    for sid, s in json.loads(file.read_text())["signs"].items():
        takes = [Take(segment_from_json(t), int(t["key_frame"]), "studio", take_id=t["take_id"]) for t in s["takes"]]
        if takes:
            out[sid] = SignSource("studio", takes, s["tags"], sorted({t["signer"] for t in s["takes"]}),
                                  int(s.get("shadow_take", 0)))
    return out


def assemble(catalogue: dict, dictionary: dict[str, SignSource], studio: dict[str, SignSource]
             ) -> tuple[dict[str, SignSource], list[str]]:
    """Catalogue order; returns (sources by sign id, ids with no reference at all)."""
    found, missing = {}, []
    for entry in catalogue["signs"]:
        sid = entry["id"]
        word = (entry.get("sources") or {}).get("dictionary")
        src = studio.get(sid) or (dictionary.get(word) if word else None)
        if src is None:
            missing.append(sid)
        else:
            found[sid] = src
    return found, missing


def distractor_segments(file: str | Path) -> list[Segment]:
    """Prepared non-vocabulary signs written by `ashiria references` (none if the file is absent)."""
    file = training_relative(file)
    if not file.exists():
        return []
    with np.load(file) as d:
        return [Segment(d["hand_rel"][i].astype(np.float64), d["wrist_pos"][i].astype(np.float64),
                        d["present"][i].astype(np.float64), float(d["fps"][i])) for i in range(len(d["fps"]))]

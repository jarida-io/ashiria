"""Everyday phrases (content/phrases/<pack>.yaml) as dictionary signs, and sentences for training and testing the
reading of signs in a row: the phrases in KSL order, plus random runs of signs (a model that only ever sees one
sentence pattern fails on real signing), each signed by joining the signs' recordings (sentences.py).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import yaml

from .config import training_relative

PHRASES = "../content/phrases/ksl-core.yaml"
RANDOM_SHARE = 0.25


@lru_cache(maxsize=1)
def load_phrases() -> list[dict]:
    return yaml.safe_load(training_relative(PHRASES).read_text())["phrases"]


@lru_cache(maxsize=1)
def sign_clips() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Single signs by name (lowercase; "name 1", "name 2" are variants of "name") and recorded phrases by name,
    each with all its clips."""
    import re

    from .study import dictionary_index

    single: dict[str, list[str]] = {}
    recorded: dict[str, list[str]] = {}
    for e in dictionary_index():
        if not e.get("clip_id"):
            continue
        name = e["sign"].strip()
        if e["category"] == "Phrases":
            recorded.setdefault(name, []).append(e["clip_id"])
        else:
            single.setdefault(re.sub(r"\s+\d+$", "", name.lower()), []).append(e["clip_id"])
    return single, recorded


def phrase_clips(phrase: dict) -> list[list[str]] | None:
    """Each gloss sign's clips (any variant can sign it), or None if a sign has no recording."""
    single, _ = sign_clips()
    out = [single.get(t.lower()) for t in phrase["gloss"]]
    return None if any(not c for c in out) else out


def training_sentences(ids: list[str], rng: np.random.Generator, count: int) -> list[list[int]]:
    """Runs of reference indices to sign in a row: the phrases (a random variant of each sign) and, for
    RANDOM_SHARE of them, 2 to 4 random signs."""
    index = {c: i for i, c in enumerate(ids)}
    phrases = [[[index[c] for c in clips if c in index] for clips in pc] for p in load_phrases()
               if (pc := phrase_clips(p))]
    phrases = [p for p in phrases if all(p)]
    out = []
    for _ in range(count):
        if rng.random() < RANDOM_SHARE or not phrases:
            out.append([int(i) for i in rng.choice(len(ids), int(rng.integers(2, 5)), replace=False)])
        else:
            p = phrases[int(rng.integers(len(phrases)))]
            out.append([int(opts[int(rng.integers(len(opts)))]) for opts in p])
    return out

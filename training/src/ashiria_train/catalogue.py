"""The sign catalogue: content/catalogue/<pack>.yaml, committed, edited by the Deaf co-creators.

It says which signs the app teaches, in which unit and order, how strictly each is checked (`tier`:
auto, or a co-creator override), the team's own hint, and where its reference comes from. It holds
no personal data and none of the KSL Dictionary's text. `ashiria catalogue` creates it from a
vocabulary file and, on later runs, only adds new words: existing entries are never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .config import Config, training_relative

TIER_CHOICES = ("auto", "scored", "lenient", "watch")
REVIEW_STATUS = ("draft", "approved", "rejected")
NUMBERS = ("zero one two three four five six seven eight nine ten").split()
HEADER = """\
# Ashiria sign catalogue: what the app teaches. Edit freely; `ashiria catalogue` only adds new words.
#   tier:   auto (computed from tracking checks) | scored | lenient | watch (shown, never scored)
#   hint:   the team's own words, shown after two misses (never text from the KSL Dictionary)
#   review: set status to approved (with by/date) once the Deaf co-creators have checked the sign
#   sources.dictionary: the KSL Dictionary sign used as a draft until Studio takes replace it
"""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def sign_id(word: str, unit_title: str) -> str:
    if len(word) == 1 and word.isalpha() and "spell" in unit_title.lower():
        return f"letter-{word}"
    if word in NUMBERS:
        return f"number-{word}"
    return slug(word)


def parse_vocabulary(file: str | Path) -> list[dict]:
    """Units from '# Unit N: title (notes)' lines and the words under each."""
    units: list[dict] = []
    for line in training_relative(file).read_text().splitlines():
        m = re.match(r"#\s*Unit\s+(\d+)\s*:\s*([^(]+)", line)
        if m:
            title = m.group(2).strip()
            units.append({"id": slug(title), "title": title[0].upper() + title[1:], "order": int(m.group(1)), "words": []})
            continue
        word = line.split("#", 1)[0].strip().lower()
        if word:
            if not units:
                units.append({"id": "signs", "title": "Signs", "order": 1, "words": []})
            units[-1]["words"].append(word)
    return units


def new_entry(word: str, unit: dict, order: int) -> dict:
    return {
        "id": sign_id(word, unit["title"]),
        "gloss": word.upper() if len(word) == 1 else word,
        "aliases": [],
        "unit": unit["id"],
        "order": order,
        "tier": "auto",
        "tier_note": "",
        "hint": "",
        "facial_grammar": False,
        "recognise": None,
        "sources": {"dictionary": word},
        "review": {"status": "draft", "by": None, "date": None},
    }


def catalogue_file(cfg: Config, pack_id: str) -> Path:
    return training_relative(cfg.paths.get("content", "../content")) / "catalogue" / f"{pack_id}.yaml"


def update_catalogue(cfg: Config, vocab_file: str | Path, pack_id: str = "ksl-core") -> Path:
    file = catalogue_file(cfg, pack_id)
    cat = load_catalogue(file) if file.exists() else {"pack_id": pack_id, "version": "0.1.0", "units": [], "signs": []}
    known_units = {u["id"] for u in cat["units"]}
    known_signs = {s["id"] for s in cat["signs"]}
    added = []
    for unit in parse_vocabulary(vocab_file):
        if unit["id"] not in known_units:
            cat["units"].append({k: unit[k] for k in ("id", "title", "order")})
            known_units.add(unit["id"])
        for k, word in enumerate(unit["words"], 1):
            entry = new_entry(word, unit, k)
            if entry["id"] not in known_signs:
                cat["signs"].append(entry)
                known_signs.add(entry["id"])
                added.append(entry["id"])
    validate(cat)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(HEADER + yaml.safe_dump(cat, sort_keys=False, allow_unicode=True, width=120))
    print(f"{file}: {len(cat['signs'])} signs in {len(cat['units'])} units; added {len(added)}"
          + (f" ({', '.join(added[:8])}{'…' if len(added) > 8 else ''})" if added else ""))
    return file


def load_catalogue(file: str | Path) -> dict:
    cat = yaml.safe_load(Path(file).read_text())
    validate(cat)
    return cat


def validate(cat: dict) -> None:
    problems = []
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(cat.get("version", ""))):
        problems.append("version must look like 1.2.3")
    units = {u["id"] for u in cat["units"]}
    seen = set()
    for s in cat["signs"]:
        sid = s.get("id", "?")
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", sid):
            problems.append(f"{sid}: ids are lowercase words joined by hyphens")
        if sid in seen:
            problems.append(f"{sid}: duplicate id")
        seen.add(sid)
        if s.get("unit") not in units:
            problems.append(f"{sid}: unknown unit {s.get('unit')!r}")
        if s.get("tier") not in TIER_CHOICES:
            problems.append(f"{sid}: tier must be one of {', '.join(TIER_CHOICES)}")
        if (s.get("review") or {}).get("status") not in REVIEW_STATUS:
            problems.append(f"{sid}: review.status must be one of {', '.join(REVIEW_STATUS)}")
    if problems:
        raise ValueError("catalogue problems:\n  " + "\n  ".join(problems))

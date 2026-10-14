"""Automatic tiers (configs/tiers.yaml): which signs the headset scores strictly, leniently, or not at all."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import training_relative

TIERS_FILE = "configs/tiers.yaml"


def load_rules(file: str | Path = TIERS_FILE) -> dict:
    return yaml.safe_load(training_relative(file).read_text())


def assign_tier(entry: dict, tags: dict, stats: dict, rules: dict) -> tuple[str, str, list[str]]:
    """(tier, source, reasons). `stats` holds simulation results: top1, top3, verify_pass,
    lookalike_false_pass, segmentation_miss (any may be missing)."""
    if entry.get("tier", "auto") != "auto":
        return entry["tier"], "co-creator", [entry.get("tier_note") or "chosen by the co-creators"]

    def above(key: str, limit) -> bool:
        return limit is not None and key in stats and stats[key] > limit

    def below(key: str, limit) -> bool:
        return limit is not None and key in stats and stats[key] < limit

    w, l = rules.get("watch", {}), rules.get("lenient", {})
    two = tags.get("hands") == "two"
    watch = [reason for ok, reason in (
        (tags.get("dominant_tracked", 1.0) < w.get("dominant_tracked_below", 0), "dominant hand often not tracked"),
        (w.get("hands_close_two_handed") and two and tags.get("hands_close"), "hands overlap"),
        (w.get("facial_grammar") and entry.get("facial_grammar"), "facial grammar"),
        (below("top3", w.get("sim_top3_below")), "rarely recognised"),
        (above("segmentation_miss", w.get("segmentation_miss_above")), "attempts often missed"),
    ) if ok]
    if watch:
        return "watch", "auto", watch
    lenient = [reason for ok, reason in (
        (l.get("near_face") and tags.get("near_face"), "near the face"),
        (l.get("two_handed") and two, "two hands"),
        (l.get("handshape_change") and tags.get("movement") == "handshape change", "handshape changes"),
        (entry["id"] in (l.get("known_hard") or []), "hard handshape for hand tracking"),
        (below("verify_pass", l.get("verify_pass_below")), "correct attempts often fail"),
        (above("lookalike_false_pass", l.get("lookalike_false_pass_above")), "look-alikes often pass"),
        (below("top1", l.get("sim_top1_below")), "often confused"),
    ) if ok]
    if lenient:
        return "lenient", "auto", lenient
    return "scored", "auto", ["passes every automatic check"]

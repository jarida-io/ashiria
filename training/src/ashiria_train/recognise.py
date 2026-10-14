"""Recognise: "which sign was that?" for Free Sign (spec/RUNTIME_SPEC.md §5).

Each candidate's distance is divided by its tau (the typical distance of a correct attempt), so
z ≈ 1 means "as close as a normal correct attempt". An absolute gate on z says "not sure" for
anything unlike every candidate; softmax confidence alone would not, because with only a few
learned signs one of them always wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonical import Segment
from .matching import descriptor, segment_features
from .params import EngineParams
from .signs import DOM, SignRef


@dataclass
class RecogniseResult:
    status: str                       # match | ambiguous | not_sure | no_hands
    best: str | None = None
    runner_up: str | None = None
    best_z: float = float("inf")
    runner_up_z: float = float("inf")
    margin: float = float("inf")
    confidence: float = 0.0
    ranking: list = field(default_factory=list)  # [(sign_id, distance, z)] best first


def recognise(attempt: Segment, candidates: list[SignRef], p: EngineParams) -> RecogniseResult:
    """`attempt` must be prepared. Ties keep the candidates' order."""
    rp = p.recognise
    if not candidates or (attempt.present[:, DOM] > 0.5).mean() < rp.min_present:
        return RecogniseResult("no_hands")
    desc = descriptor(segment_features(attempt), p.weights)
    dist = np.array([c.closest(desc, p.dtw_band)[0] for c in candidates])
    z = dist / np.array([c.tau for c in candidates])
    order = np.argsort(z, kind="stable")
    logits = -z / rp.temperature
    conf = np.exp(logits - logits.max())
    conf /= conf.sum()
    b = order[0]
    r = RecogniseResult("match", best=candidates[b].id, best_z=float(z[b]), confidence=float(conf[b]),
                        ranking=[(candidates[i].id, float(dist[i]), float(z[i])) for i in order[:rp.top]])
    if len(order) > 1:
        ru = order[1]
        r.runner_up, r.runner_up_z = candidates[ru].id, float(z[ru])
        r.margin = r.runner_up_z - r.best_z
    if r.best_z > rp.accept_z:
        r.status = "not_sure"
    elif r.margin < rp.min_margin_z and r.runner_up_z <= rp.accept_z:
        r.status = "ambiguous"
    return r

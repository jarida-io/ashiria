"""`ashiria steps-fit`: learn the step generator's calls (two hands, starting handshape, palm direction, location,
kinds of movement) from the KSL Dictionary's descriptions, scored on held-out clips against the hand-written rules.

Only the fitted weights are kept (exports/steps/model.json, on this computer like every export); no description text
is. steps.py applies them with plain numpy.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import training_relative
from .steps import MODEL, MOVE_KINDS, measure, sign_steps
from .steps_check import described_clips, read_description

MIN_CLASS = 12   # rarer handshapes and places are left out
FOLDS = 5
PRECISION = 0.7  # a call is put in a caption only when, on held-out clips, calls that sure agree this often


def collect(limit: int | None = None) -> pd.DataFrame:
    """One row per described clip: the measurements, what the description says, and what the rules say."""
    from .study import clip_pose

    rows = []
    for _, r in described_clips().head(limit).iterrows():
        pose = clip_pose(r["clip"])
        x = measure(pose) if pose is not None else None
        if x is None:
            continue
        said = read_description(r["description"])
        said["palm"] = "side" if said["palm"] == "each_other" else said["palm"]
        rules = sign_steps(pose, use_model=False)["facts"]
        rows.append({"clip": r["clip"], "label": r["label"], **{f"x_{k}": v for k, v in x.items()},
                     **{f"said_{k}": said[k] for k in ("two", "shape", "palm", "loc")}, "said_moves": said["moves"],
                     **{f"rule_{k}": rules[k] for k in ("two", "shape", "palm", "loc")}, "rule_moves": rules["moves"]})
    return pd.DataFrame(rows)


def _pipe(balanced: bool = False):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced" if balanced else None))


def _export(pipe, classes) -> dict:
    scaler, lr = pipe[1], pipe[2]
    return {"classes": [c.item() if hasattr(c, "item") else c for c in classes],
            "mean": scaler.mean_.round(6).tolist(), "scale": scaler.scale_.round(6).tolist(),
            "coef": lr.coef_.round(6).tolist(), "intercept": lr.intercept_.round(6).tolist()}


def fit(df: pd.DataFrame) -> tuple[dict, dict]:
    """Fitted models and their held-out agreement with the descriptions, next to the rules' agreement."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    models, scores = {}, {}
    for key, inputs in (("two", "x_two"), ("shape", "x_shape"), ("palm", "x_palm"), ("loc", "x_loc")):
        d = df[df[f"said_{key}"].notna()]
        counts = d[f"said_{key}"].value_counts()
        d = d[d[f"said_{key}"].isin(counts[counts >= MIN_CLASS].index)]
        X, y = np.stack(d[inputs].to_numpy()), d[f"said_{key}"].to_numpy()
        if key == "two":
            y = y.astype(bool)
        proba = cross_val_predict(_pipe(), X, y, cv=cv, method="predict_proba")
        pipe = _pipe().fit(X, y)
        held = pipe.classes_[proba.argmax(1)]
        right, sure = held == y, proba.max(1)
        say = _sure_enough(right, sure)
        models[key] = {**_export(pipe, pipe.classes_), "say_above": say}
        stated = sure >= say
        scores[key] = {"described": len(y), "rules": round(float(np.mean(d[f"rule_{key}"].to_numpy() == y)), 3),
                       "model": round(float(right.mean()), 3), "classes": int(len(pipe.classes_)),
                       "stated": round(float(stated.mean()), 3),
                       "stated_right": round(float(right[stated].mean()) if stated.any() else 0.0, 3)}

    moves, X = {}, np.stack(df["x_move"].to_numpy())
    for kind in MOVE_KINDS:
        y = np.array([kind in m for m in df["said_moves"]])
        p = cross_val_predict(_pipe(balanced=True), X, y, cv=cv, method="predict_proba")[:, 1]
        # Claim a movement only when claims that sure are right often enough; it is better to leave it to the avatar
        # than to name the wrong one.
        best = _sure_enough(y, p, grid=np.linspace(0.3, 0.98, 35))
        pipe = _pipe(balanced=True).fit(X, y)
        moves[kind] = {**_export(pipe, pipe.classes_), "threshold": round(float(best), 3)}
        rule = np.array([kind in m for m in df["rule_moves"]])
        scores[kind] = {"described": int(y.sum()),
                        "rules": {"found": _rate(rule, y), "false": _rate(rule, ~y), "f1": round(_f1(y, rule), 3)},
                        "model": {"found": _rate(p >= best, y), "false": _rate(p >= best, ~y), "f1": round(_f1(y, p >= best), 3)},
                        "precision": {"rules": _rate(y, rule), "model": _rate(y, p >= best)}}
    return {"models": models, "moves": moves}, scores


def _sure_enough(right: np.ndarray, sure: np.ndarray, grid=np.linspace(0.2, 0.98, 40)) -> float:
    """The lowest confidence at which calls are right at least PRECISION of the time (1.01: never sure enough)."""
    for t in grid:
        hit = sure >= t
        if hit.sum() >= 10 and right[hit].mean() >= PRECISION:
            return round(float(t), 3)
    return 1.01


def _rate(pred: np.ndarray, among: np.ndarray) -> float:
    return round(float(pred[among].mean()) if among.any() else 0.0, 3)


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float((y & pred).sum())
    return 2 * tp / max(float(y.sum() + pred.sum()), 1.0)


def run_fit(limit: int | None = None) -> dict:
    t = time.time()
    df = collect(limit)
    print(f"{len(df)} described clips measured ({time.time() - t:.0f} s)")
    model, scores = fit(df)
    out = training_relative(MODEL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**model, "scores": scores, "clips": len(df)}))
    print(f"Held-out agreement with the descriptions (rules -> fitted; captions state a call only when it is right "
          f"{PRECISION:.0%} of the time):")
    for key in ("two", "shape", "palm", "loc"):
        s = scores[key]
        print(f"  {key:6s} {s['rules']:.0%} -> {s['model']:.0%}; stated for {s['stated']:.0%} of signs, right "
              f"{s['stated_right']:.0%} of those  ({s['described']} described, {s['classes']} classes)")
    for kind in MOVE_KINDS:
        s = scores[kind]
        r, m, pr = s["rules"], s["model"], s["precision"]
        print(f"  {kind:6s} found {r['found']:.0%} -> {m['found']:.0%}; claims right {pr['rules']:.0%} -> "
              f"{pr['model']:.0%}  ({s['described']} described)")
    print(f"Wrote {out}")
    return scores

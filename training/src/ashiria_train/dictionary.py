"""Import the KSL Dictionary scrape (KSL_Dictionary_Extractor) as a labels CSV for `ashiria audit`.

The scrape lists every sign twice and many English glosses share one video, so this keeps one
row per unique video and records all of its glosses as aliases. Rights: the videos are the KSL
Dictionary's copyright. Use them for local development only until its owners give permission.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config, path, training_relative

SIGNER = "ksl_dictionary"
MIN_BYTES = 5_000  # smaller files are truncated downloads


def import_dictionary(cfg: Config, source_csv: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(training_relative(source_csv or cfg.dictionary["source_csv"]), dtype=str).fillna("")
    df = df[df["video_path"] != ""].copy()
    df["number"] = df["number"].astype(int)
    df = df.sort_values("number")
    root = path(cfg, "raw_videos")

    rows, missing = [], 0
    for _, group in df.groupby("video_src", sort=False):
        # Every listing saved its own copy; use the first copy that exists and isn't a broken download.
        copies = [Path(*Path(p).parts[1:]).as_posix() for p in group["video_path"]]  # drop "ksl_videos/"
        usable = [c for c in copies if (root / c).exists() and (root / c).stat().st_size >= MIN_BYTES]
        if not usable:
            missing += 1
            continue
        rel = usable[0]
        first = group.iloc[0]
        glosses = list(dict.fromkeys(s.strip() for s in group["sign"] if s.strip()))
        rows.append({
            "path": rel,
            "label": glosses[0].lower(),
            "signer": SIGNER,
            "aliases": "; ".join(glosses),
            "category": group["category"].mode().iloc[0],
            "description": next((d for d in group["description"] if d), ""),
            "note": next((n for n in group["note"] if n), ""),
            "number": int(first["number"]),
        })
    out = pd.DataFrame(rows)
    target = training_relative(cfg.labels["csv"])
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"{len(out)} unique videos ({missing} with no usable copy on disk) -> {target}")
    print(out["category"].value_counts().to_string())
    return out

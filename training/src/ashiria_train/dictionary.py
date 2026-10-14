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


def folder_index(root: Path) -> dict[str, str]:
    """Listing number -> its folder. Folders are named "<number>_<sign>" as the scraper wrote them, but characters some
    filesystems refuse (such as "?") were saved as "_", so the data sheet's paths don't always match. The number does."""
    return {d.name.split("_", 1)[0]: d.name for d in root.iterdir() if d.is_dir() and "_" in d.name}


def resolve_copy(root: Path, folders: dict[str, str], video_path: str, number) -> str | None:
    """The saved copy of one data-sheet row, relative to `root`, or None if nothing was saved."""
    if not video_path:
        return None
    rel = Path(*Path(video_path).parts[1:]).as_posix()  # drop "ksl_videos/"
    if (root / rel).exists():
        return rel
    folder = folders.get(str(number))
    return f"{folder}/{Path(video_path).name}" if folder and (root / folder / Path(video_path).name).exists() else None


def is_video(file: Path) -> bool:
    """A real MP4, not a truncated download or the error page the website sent instead (saved as .mp4)."""
    if file.stat().st_size < MIN_BYTES:
        return False
    with open(file, "rb") as f:
        return f.read(12)[4:8] == b"ftyp"


def import_dictionary(cfg: Config, source_csv: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(training_relative(source_csv or cfg.dictionary["source_csv"]), dtype=str).fillna("")
    df = df[df["video_path"] != ""].copy()
    df["number"] = df["number"].astype(int)
    df = df.sort_values("number")
    root = path(cfg, "raw_videos")
    folders = folder_index(root)

    rows, missing = [], 0
    for _, group in df.groupby("video_src", sort=False):
        # Every listing saved its own copy; use the first that is a real video. Copies at the data sheet's exact path
        # come first, so clips found before folder names were matched by number keep their clip ids.
        exact = [Path(*Path(p).parts[1:]).as_posix() for p in group["video_path"]]
        found = [resolve_copy(root, folders, p, n) for p, n in zip(group["video_path"], group["number"])]
        copies = list(dict.fromkeys([c for c in exact if (root / c).exists()] + [c for c in found if c]))
        usable = [c for c in copies if is_video(root / c)]
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

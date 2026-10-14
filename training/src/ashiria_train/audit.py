"""Step 1: inventory the videos, read labels and signers, and report what the dataset contains."""

from __future__ import annotations

import hashlib
import os
import re

# Broken downloads make OpenCV and FFmpeg print a warning per file; the audit reports them instead.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config, path, training_relative


def clip_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def find_videos(root: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))


def make_label_reader(lcfg: Config, root: Path):
    """Returns read(video) -> (label, signer, extra columns from the labels CSV)."""
    signer_re = re.compile(lcfg["signer_regex"]) if lcfg.get("signer_regex") else None
    lookup = None
    if lcfg["mode"] == "csv":
        table = pd.read_csv(training_relative(lcfg["csv"]), dtype=str)
        lookup = {str(r["path"]).replace("\\", "/"): r for _, r in table.iterrows()}
    name_re = re.compile(lcfg["regex"]) if lcfg["mode"] == "regex" else None

    def read(video: Path) -> tuple[str | None, str | None, dict]:
        rel = video.relative_to(root).as_posix()
        label = signer = None
        extras: dict = {}
        if lookup is not None:
            row = lookup.get(rel)
            if row is not None:
                label = str(row["label"])
                signer = str(row["signer"]) if "signer" in row and pd.notna(row["signer"]) else None
                extras = {k: v for k, v in row.items() if k not in ("path", "label", "signer")}
        elif name_re is not None:
            m = name_re.match(video.stem)
            if m:
                label = m.groupdict().get("label")
                signer = m.groupdict().get("signer")
        else:
            parents = Path(rel).parent.parts
            level = int(lcfg.get("folder_level", 1))
            if len(parents) >= level:
                label = parents[-level]
        if signer_re is not None and signer is None:
            m = signer_re.search(rel)
            signer = m.group("signer") if m else None
        if label is not None:
            label = label.strip()
            label = label.lower() if lcfg.get("lowercase", True) else label
        return label, signer, extras

    return read


def probe(video: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"readable": False}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {
        "readable": frames > 0,
        "fps": round(fps, 3),
        "frames": frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "duration_s": round(frames / fps, 3) if fps > 0 else None,
    }
    cap.release()
    return info


def audit(cfg: Config) -> pd.DataFrame:
    root = path(cfg, "raw_videos")
    if not root.exists():
        raise SystemExit(f"Video folder not found: {root}\nCopy the videos there or set paths.raw_videos in the config.")
    videos = find_videos(root, cfg.video_extensions)
    if not videos:
        raise SystemExit(f"No videos with extensions {cfg.video_extensions} under {root}")

    read_label = make_label_reader(cfg.labels, root)
    rows = []
    for v in tqdm(videos, desc="audit"):
        rel = v.relative_to(root).as_posix()
        label, signer, extras = read_label(v)
        rows.append({"clip_id": clip_id(rel), "path": rel, "label": label, "signer": signer,
                     "size_mb": round(v.stat().st_size / 1e6, 2), **probe(v), **extras})
    df = pd.DataFrame(rows)

    out = path(cfg, "manifests")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "videos.csv", index=False)
    report = render_report(df, root)
    (out / "audit_report.md").write_text(report)
    print(report)
    return df


def render_report(df: pd.DataFrame, root: Path) -> str:
    ok = df[df["readable"].fillna(False).astype(bool)]
    labelled = ok[ok["label"].notna()]
    counts = labelled["label"].value_counts()
    signers = labelled["signer"].dropna()
    dupes = ok[ok.duplicated(["size_mb", "frames"], keep=False)]

    lines = [
        "# Video audit",
        "",
        f"- Folder: `{root}`",
        f"- Videos found: {len(df)} ({len(df) - len(ok)} unreadable)",
        f"- Total size: {df['size_mb'].sum() / 1000:.1f} GB; total duration: {ok['duration_s'].sum() / 60:.1f} min",
        f"- Labelled: {len(labelled)}; unlabelled: {len(ok) - len(labelled)}",
        f"- Distinct labels: {counts.size}",
    ]
    if counts.size:
        lines.append(f"- Clips per label: min {counts.min()}, median {int(counts.median())}, max {counts.max()}")
        lines.append(f"- Labels with fewer than 5 clips: {int((counts < 5).sum())}")
    lines.append(f"- Signers identified: {signers.nunique() if len(signers) else 0}"
                 + ("" if len(signers) else " (none: test accuracy will be optimistic; see labels.signer_regex)"))
    if len(ok):
        lines += [
            f"- Duration (s): min {ok['duration_s'].min():.1f}, median {ok['duration_s'].median():.1f}, max {ok['duration_s'].max():.1f}",
            f"- FPS: {', '.join(f'{k:g} ({v})' for k, v in ok['fps'].round().value_counts().head(5).items())}",
            f"- Resolution: {', '.join(f'{w}x{h} ({n})' for (w, h), n in ok.groupby(['width', 'height']).size().sort_values(ascending=False).head(5).items())}",
        ]
    lines.append(f"- Possible duplicates (same size and frame count): {len(dupes)}")
    if counts.size:
        lines += ["", "## Clips per label", "", "| Label | Clips |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines += ["", "## Sample of parsed rows", "", "| Path | Label | Signer |", "|---|---|---|"]
    lines += [f"| {r.path} | {r.label} | {r.signer} |" for r in df.head(15).itertuples()]
    return "\n".join(lines) + "\n"

"""`ashiria audit-dictionary`: check every video the KSL Dictionary data sheet lists.

Three kinds of check, per video (one video can be listed under several sign names):
- the download: missing, broken (truncated) or unreadable files, and listings whose saved copies differ;
- what MediaPipe sees: how often it finds the hands and the body, whether the hands ever rise into signing space,
  which hand leads, frames that failed to decode, and whether Ashiria can cut a reference from it;
- duplicates: the same file, or nearly the same motion, listed as different videos.
It also checks the data sheet itself (names, descriptions, rows without a video).

Writes `dictionary_audit.csv` (one row per video, with flags) and `dictionary_audit.md` to the dictionary's manifests folder, which is
local and git-ignored like the videos. `ashiria study-index` reads the flags so the sign browser can show them.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, path, training_relative
from .dictionary import folder_index, is_video, resolve_copy

DICTIONARY_CONFIG = "configs/ksl_dictionary.yaml"
FEW_HANDS = 0.5            # hands found in fewer frames than this
FEW_BODY = 0.8
LEAD_MARGIN = 0.3          # shoulder widths, as in canonical.left_dominant_votes
NEAR_DUPLICATE = 0.02      # motion distance (shoulder widths per coordinate) below which two clips look like one recording
DESC_FRAMES = 16

FLAG_TEXT = {
    "missing": "video missing from the download",
    "broken": "every saved copy is an error page or a truncated file, not a video",
    "copies_differ": "the saved copies of this video differ",
    "partial_decode": "some frames failed to decode",
    "few_hands": "MediaPipe finds the hands in under half the frames",
    "few_body": "MediaPipe finds the body in under 80% of frames",
    "no_signing": "the hands stay below mid-torso throughout (a low sign, or no sign shown)",
    "no_reference": "Ashiria can't cut a reference from it",
    "left_leads": "the left hand leads (left-handed sign or mirrored video?)",
    "short": "very short",
    "long": "very long",
    "duplicate": "the same file is listed as another video",
    "near_duplicate": "nearly the same motion as another video",
}


@lru_cache(maxsize=1)
def _engine():
    from .params import EngineParams

    dcfg = load_config(DICTIONARY_CONFIG)
    return dcfg.features, EngineParams.from_config(dcfg)


def clip_check(lm_file: str) -> tuple[dict, np.ndarray | None]:
    """What MediaPipe found in one clip, and a motion descriptor for finding near-duplicates."""
    from .canonical import from_mediapipe, gap_fill, hand_activity, trim
    from .pipeline import cut_reference

    f, params = _engine()
    with np.load(lm_file) as d:
        hp, score, pp = d["hand_present"].astype(bool), d["hand_score"], d["pose_present"].astype(bool)
        hand, pose = d["hand_image"][..., :2].astype(np.float64), d["pose_image"][..., :2].astype(np.float64)
        seg, why = from_mediapipe(d)
    out = {"lm_frames": len(hp), "hands_rate": float(hp.any(1).mean()) if len(hp) else 0.0,
           "right_rate": float(hp[:, 0].mean()) if len(hp) else 0.0, "left_rate": float(hp[:, 1].mean()) if len(hp) else 0.0,
           "body_rate": float(pp.mean()) if len(pp) else 0.0, "hand_score": float(score[hp].mean()) if hp.any() else 0.0}

    # Signing space: a tracked wrist above the middle of the torso (image y points down).
    shoulders, hips = pose[:, [11, 12], 1].mean(1), pose[:, [23, 24], 1].mean(1)
    raised = (hand[:, :, 0, 1] < (shoulders + 0.5 * (hips - shoulders))[:, None]) & hp & pp[:, None]
    out["signing_rate"] = float(raised.any(1).mean()) if len(hp) else 0.0

    lead, ref = "", None
    if seg is not None:
        a = hand_activity(seg)
        if np.isfinite(a).all():
            lead = "left" if a[1] - a[0] > LEAD_MARGIN else "right" if a[0] - a[1] > LEAD_MARGIN else "both"
        else:
            lead = "right" if np.isfinite(a[0]) else "left" if np.isfinite(a[1]) else ""
        s = trim(gap_fill(seg, f.gap_fill_frames), f.rest_height, f.trim_margin_frames)
        if s is not None and len(s) >= 4:
            ref = cut_reference(s, params, f.rest_height)
            why = "" if ref is not None else "no attempt found"
        else:
            why = "no hands above rest"
    out.update(lead=lead, reference=ref is not None, reference_why=why)

    # Motion descriptor: both hands relative to the shoulder midpoint, in shoulder widths, at 16 evenly spaced frames.
    desc = None
    if pp.any() and hp.any():
        mid = pose[:, [11, 12]].mean(1)
        width = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1)
        ok = pp & (width > 1e-3)
        if ok.any():
            k = np.flatnonzero(ok)
            near = k[np.abs(np.arange(len(ok))[:, None] - k[None]).argmin(1)]
            rel = (hand - mid[near][:, None, None]) / width[near][:, None, None, None]
            rel[~hp] = 0.0
            idx = np.linspace(0, len(hp) - 1, DESC_FRAMES).round().astype(int)
            desc = np.concatenate([rel[idx].ravel(), hp[idx].ravel().astype(np.float64)])
    return out, desc


def _is_html(file: Path) -> bool:
    with open(file, "rb") as f:
        return f.read(64).lstrip().lower().startswith((b"<!doctype html", b"<html"))


def _sha1(file: Path) -> str:
    return hashlib.sha1(file.read_bytes()).hexdigest()


def audit_dictionary(workers: int = 8) -> pd.DataFrame:
    dcfg = load_config(DICTIONARY_CONFIG)
    raw, landmarks, manifests = path(dcfg, "raw_videos"), path(dcfg, "landmarks"), path(dcfg, "manifests")
    sheet = pd.read_csv(training_relative(dcfg.dictionary["source_csv"]), dtype=str).fillna("")
    manifest = pd.read_csv(manifests / "videos.csv", dtype=str).fillna("")
    by_path = manifest.set_index("path")
    clip_by_path = {r["path"]: r["clip_id"] for _, r in manifest[manifest["label"] != ""].iterrows()}

    sheet["n"] = pd.to_numeric(sheet["number"], errors="coerce").fillna(10**9).astype(int)
    sheet = sheet.sort_values("n")
    listed = sheet[sheet["video_src"] != ""]

    # 1. The download, per video.
    folders = folder_index(raw)
    rows = []
    for src, group in listed.groupby("video_src", sort=False):
        exact = [Path(*Path(p).parts[1:]).as_posix() for p in group["video_path"] if p]
        found = [resolve_copy(raw, folders, p, n) for p, n in zip(group["video_path"], group["n"])]
        copies = list(dict.fromkeys(exact))
        on_disk = list(dict.fromkeys([c for c in exact if (raw / c).exists()] + [c for c in found if c]))
        usable = [c for c in on_disk if is_video(raw / c)]
        hashes = {_sha1(raw / c) for c in usable}
        names = list(dict.fromkeys(s.strip() for s in group["sign"] if s.strip()))
        chosen = next((c for c in usable if c in clip_by_path), usable[0] if usable else None)
        clip = clip_by_path.get(chosen) if chosen else None
        info = by_path.loc[chosen] if chosen and chosen in by_path.index else None
        rows.append({
            "video_src": src, "number": int(group["n"].iloc[0]), "names": "; ".join(names),
            "category": group["category"].mode().iloc[0], "listings": len(group),
            "copies_listed": len(set(copies)), "copies_on_disk": len(on_disk), "copies_usable": len(usable),
            "distinct_files": len(hashes), "sha1": min(hashes) if hashes else "", "clip_id": clip or "",
            "path": chosen or "", "copies_html": sum(_is_html(raw / c) for c in on_disk),
            "fps": float(info["fps"]) if info is not None and info["fps"] else np.nan,
            "frames": float(info["frames"]) if info is not None and info["frames"] else np.nan,
            "width": float(info["width"]) if info is not None and info["width"] else np.nan,
            "height": float(info["height"]) if info is not None and info["height"] else np.nan,
            "duration_s": float(info["duration_s"]) if info is not None and info["duration_s"] else np.nan,
        })
    df = pd.DataFrame(rows)

    # 2. What MediaPipe sees, per clip (in parallel: cutting references takes ~0.1 s a clip).
    todo = [(i, str(landmarks / f"{c}.npz")) for i, c in enumerate(df["clip_id"]) if c and (landmarks / f"{c}.npz").exists()]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(clip_check, [f for _, f in todo], chunksize=16))
    descs = {}
    for (i, _), (stats, desc) in zip(todo, results):
        for k, v in stats.items():
            df.loc[i, k] = v
        if desc is not None:
            descs[i] = desc

    # 3. Duplicates: the same file under two videos, or (nearly) the same motion.
    df["duplicate_of"] = ""
    for _, g in df[df["sha1"] != ""].groupby("sha1"):
        if len(g) > 1:
            for i in g.index:
                df.loc[i, "duplicate_of"] = "; ".join(df.loc[j, "names"].split("; ")[0] for j in g.index if j != i)
    df["near_duplicate_of"], df["motion_distance"] = "", np.nan
    if len(descs) > 1:
        keys = np.array(sorted(descs))
        X = np.stack([descs[k] for k in keys])
        sq = (X ** 2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0) / X.shape[1])
        np.fill_diagonal(dist, np.inf)
        nn = dist.argmin(1)
        for a, b in enumerate(nn):
            i, j = keys[a], keys[b]
            df.loc[i, "motion_distance"] = dist[a, b]
            if dist[a, b] < NEAR_DUPLICATE and df.loc[i, "sha1"] != df.loc[j, "sha1"]:
                df.loc[i, "near_duplicate_of"] = df.loc[j, "names"].split("; ")[0]

    df["flags"] = df.apply(_flags, axis=1)
    manifests.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifests / "dictionary_audit.csv", index=False)
    report = render_report(df, sheet)
    (manifests / "dictionary_audit.md").write_text(report)
    print(report)
    print(f"-> {manifests / 'dictionary_audit.csv'}\n-> {manifests / 'dictionary_audit.md'}")
    return df


def _flags(r) -> str:
    if not r["copies_usable"]:
        return "broken" if r["copies_on_disk"] else "missing"
    out = []
    if r["distinct_files"] > 1:
        out.append("copies_differ")
    if not r["clip_id"] or pd.isna(r.get("lm_frames")):
        return "; ".join(out + ["no_reference"])
    if r["frames"] and r["lm_frames"] < 0.9 * r["frames"]:
        out.append("partial_decode")
    if r["hands_rate"] < FEW_HANDS:
        out.append("few_hands")
    if r["body_rate"] < FEW_BODY:
        out.append("few_body")
    if r["signing_rate"] < 0.05:
        out.append("no_signing")
    if not r["reference"]:
        out.append("no_reference")
    if r["lead"] == "left":
        out.append("left_leads")
    if r["duration_s"] < 1.0:
        out.append("short")
    if r["duration_s"] > 10.0:
        out.append("long")
    if r["duplicate_of"]:
        out.append("duplicate")
    if r["near_duplicate_of"]:
        out.append("near_duplicate")
    return "; ".join(out)


def audit_flags() -> dict[str, list[str]]:
    """clip_id or video_src -> flag codes, from the last `ashiria audit-dictionary` (empty if it hasn't run)."""
    file = path(load_config(DICTIONARY_CONFIG), "manifests") / "dictionary_audit.csv"
    if not file.exists():
        return {}
    df = pd.read_csv(file, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        flags = [f for f in r["flags"].split("; ") if f]
        out[r["clip_id"] or r["video_src"]] = flags
    return out


def render_report(df: pd.DataFrame, sheet: pd.DataFrame) -> str:
    n = len(df)
    flag_counts = pd.Series([f for fs in df["flags"] for f in fs.split("; ") if f]).value_counts()
    seen = df[df["lm_frames"].notna()] if "lm_frames" in df else df.iloc[:0]
    pct = lambda x: f"{100 * x:.0f}%"

    names = sheet["sign"].str.strip()
    lines = [
        "# KSL Dictionary video audit", "",
        "Local and git-ignored: describes the dictionary's videos. Regenerate with `ashiria audit-dictionary`.", "",
        "## The data sheet", "",
        f"- Rows: {len(sheet)}; sign names: {names[names != ''].nunique()}; videos listed: {df['video_src'].nunique()}",
        f"- Rows without a video link: {int((sheet['video_src'] == '').sum())}; rows whose video the scraper didn't save "
        f"(another row with the same video did): {int(((sheet['video_src'] != '') & (sheet['video_path'] == '')).sum())}",
        f"- Rows per video: median {int(df['listings'].median())}, max {int(df['listings'].max())} "
        "(the scrape lists every sign more than once, and many names share a video)",
        f"- Videos listed under more than one name: {int((df['names'].str.count('; ') > 0).sum())}",
        f"- Names with more than one video: {int(sheet.drop_duplicates(['sign', 'video_src']).groupby(names).size().gt(1).sum())}",
        f"- Rows with no description: {int((sheet['description'].str.strip() == '').sum())}",
        f"- Names with stray characters (leading or trailing punctuation or spaces, double spaces): "
        f"{', '.join(sorted({s for s in sheet['sign'] if re.search(r'^[^A-Za-z0-9]|[^A-Za-z0-9?!.)]$|  ', s)})[:20]) or 'none'}",
        "", "## The download", "",
        f"- Videos usable: {int((df['copies_usable'] > 0).sum())} of {n}",
        f"- Missing (no file saved): {int(flag_counts.get('missing', 0))}; broken (every copy is the website's error page or "
        f"a truncated file): {int(flag_counts.get('broken', 0))}",
        f"- Files saved: {int(df['copies_on_disk'].sum())} for {n} videos ({int(df['copies_html'].sum())} of them the website's "
        f"404 page); videos whose saved copies differ: {int(flag_counts.get('copies_differ', 0))}",
    ]
    if len(seen):
        res = seen.groupby(["width", "height"]).size().sort_values(ascending=False)
        lines += [
            f"- Resolution: {', '.join(f'{int(w)}x{int(h)} ({c})' for (w, h), c in res.head(4).items())}",
            f"- Frame rate: {', '.join(f'{k:g} fps ({v})' for k, v in seen['fps'].round().value_counts().head(4).items())}",
            f"- Length: median {seen['duration_s'].median():.1f} s (shortest {seen['duration_s'].min():.1f} s, longest {seen['duration_s'].max():.1f} s)",
            "", "## What MediaPipe sees", "",
            f"- Clips checked: {len(seen)}",
            f"- Hands found: median {pct(seen['hands_rate'].median())} of frames; under half the frames in "
            f"{int(flag_counts.get('few_hands', 0))} clips",
            f"- Body found: median {pct(seen['body_rate'].median())}; under 80% in {int(flag_counts.get('few_body', 0))} clips",
            f"- Hands stay below mid-torso throughout: {int(flag_counts.get('no_signing', 0))} clips (low signs, or no sign shown)",
            f"- Leading hand: {', '.join(f'{k} {v}' for k, v in seen['lead'].replace('', 'none').value_counts().items())}",
            f"- Frames that failed to decode (over 10%): {int(flag_counts.get('partial_decode', 0))} clips",
            f"- Reference cut: {int(seen['reference'].astype(bool).sum())} of {len(seen)} clips",
            "", "## Duplicates", "",
            f"- The same file listed as different videos: {int(flag_counts.get('duplicate', 0))} videos",
            f"- Nearly the same motion as another video (distance under {NEAR_DUPLICATE}): {int(flag_counts.get('near_duplicate', 0))} videos",
        ]
    lines += ["", "## Flags", "", "| Flag | Videos | Meaning |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {FLAG_TEXT.get(k, '')} |" for k, v in flag_counts.items()]
    for flag, title in (("missing", "Missing videos"), ("broken", "Broken downloads"), ("no_signing", "Hands stay low"),
                        ("duplicate", "Same file as another video"), ("near_duplicate", "Nearly the same motion"),
                        ("left_leads", "Left hand leads"), ("few_hands", "Hands found in under half the frames")):
        hit = df[df["flags"].str.contains(rf"\b{flag}\b")]
        if not len(hit):
            continue
        lines += ["", f"## {title} ({len(hit)})", ""]
        for r in hit.sort_values("names").head(60).itertuples():
            extra = {"duplicate": f" = {r.duplicate_of}", "near_duplicate": f" ~ {r.near_duplicate_of}",
                     "few_hands": f" ({pct(r.hands_rate)})"}.get(flag, "")
            lines.append(f"- {r.names.split('; ')[0]}{extra} ({r.video_src})")
        if len(hit) > 60:
            lines.append(f"- ... and {len(hit) - 60} more (see dictionary_audit.csv)")
    return "\n".join(lines) + "\n"

"""Step 2: run MediaPipe hand and pose landmarkers on every video.

Writes one compressed .npz per clip (arrays per processed frame):
  fps            processed frames per second
  hand_world     [T,2,21,3] hand-centred metres, camera axes
  hand_image     [T,2,21,3] normalised image x, y and relative depth
  hand_present   [T,2]      slot 0 = signer's right hand, slot 1 = left
  hand_score     [T,2]      MediaPipe handedness confidence
  pose_world     [T,33,3]   hip-centred metres, camera axes
  pose_image     [T,33,4]   normalised image x, y, z and visibility
  pose_present   [T]
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe's native INFO/WARNING lines

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import joints as J
from .config import Config, path

HAND_MODEL = "hand_landmarker.task"


def assign_slots(dets: list[tuple], pose_image: np.ndarray | None) -> dict[int, int]:
    """Map hand detections to slots {slot: detection index} for footage that is not mirrored.

    Uses the body pose's wrists when available; otherwise MediaPipe's handedness label, which names the hand itself
    on ordinary video ("Right" is the signer's right: checked on the dictionary, 198 of 200 detections agree with the
    pose). MediaPipe's documentation says the label assumes a mirrored image; this version doesn't.
    """
    if not dets:
        return {}
    if pose_image is not None:
        targets = pose_image[[J.POSE_RIGHT_WRIST, J.POSE_LEFT_WRIST], :2]
        cost = np.array([[np.linalg.norm(d[0][J.WRIST, :2] - t) for t in targets] for d in dets])
        if len(dets) == 1:
            return {int(np.argmin(cost[0])): 0}
        straight, crossed = cost[0, 0] + cost[1, 1], cost[0, 1] + cost[1, 0]
        return {0: 0, 1: 1} if straight <= crossed else {0: 1, 1: 0}
    if len(dets) == 2 and dets[0][2] == dets[1][2]:
        # Same label twice: the signer's right hand is on the left of the image.
        order = np.argsort([d[0][J.WRIST, 0] for d in dets])
        return {J.RIGHT: int(order[0]), J.LEFT: int(order[1])}
    return {(J.RIGHT if d[2] == "Right" else J.LEFT): k for k, d in enumerate(dets)}


def _landmarkers(models_dir: str, e: dict):
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(models_dir, HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=e["min_hand_detection_confidence"],
            min_hand_presence_confidence=e["min_hand_presence_confidence"],
            min_tracking_confidence=e["min_tracking_confidence"],
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(models_dir, e["pose_model"])),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    return hand, pose


def process_clip(clip: str, video: str, out_file: str, models_dir: str, e: dict) -> dict:
    """Extract one video. Runs in a worker process; new landmarkers per clip so no tracking state leaks."""
    import cv2
    import mediapipe as mp

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps < 1:
        src_fps = 30.0
    step = max(1, int(round(src_fps / e["target_fps"])))

    frames = {k: [] for k in ("hand_world", "hand_image", "hand_present", "hand_score", "pose_world", "pose_image", "pose_present")}
    hand, pose = _landmarkers(models_dir, e)
    truncated = False
    with hand, pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if len(frames["pose_present"]) >= e["max_frames"]:
                    truncated = True
                    break
                if e["videos_are_mirrored"]:
                    frame = cv2.flip(frame, 1)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(i * 1000.0 / src_fps)
                hr = hand.detect_for_video(image, ts)
                pr = pose.detect_for_video(image, ts)

                pose_world = np.zeros((J.NUM_POSE, 3), np.float32)
                pose_image = np.zeros((J.NUM_POSE, 4), np.float32)
                if pr.pose_landmarks:
                    pose_image = np.array([[p.x, p.y, p.z, p.visibility or 0.0] for p in pr.pose_landmarks[0]], np.float32)
                    pose_world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]], np.float32)

                dets = [
                    (
                        np.array([[p.x, p.y, p.z] for p in hr.hand_landmarks[k]], np.float32),
                        np.array([[p.x, p.y, p.z] for p in hr.hand_world_landmarks[k]], np.float32),
                        hr.handedness[k][0].category_name,
                        hr.handedness[k][0].score,
                    )
                    for k in range(len(hr.hand_landmarks))
                ]
                hand_world = np.zeros((2, J.NUM_JOINTS, 3), np.float32)
                hand_image = np.zeros((2, J.NUM_JOINTS, 3), np.float32)
                hand_present = np.zeros(2, bool)
                hand_score = np.zeros(2, np.float32)
                for slot, k in assign_slots(dets, pose_image if pr.pose_landmarks else None).items():
                    hand_image[slot], hand_world[slot], _, hand_score[slot] = dets[k]
                    hand_present[slot] = True

                for key, value in (("hand_world", hand_world), ("hand_image", hand_image), ("hand_present", hand_present),
                                   ("hand_score", hand_score), ("pose_world", pose_world), ("pose_image", pose_image),
                                   ("pose_present", bool(pr.pose_landmarks))):
                    frames[key].append(value)
            i += 1
    cap.release()

    n = len(frames["pose_present"])
    stats = {"clip_id": clip, "frames": n, "fps": src_fps / step, "truncated": truncated}
    if n == 0:
        return {**stats, "status": "no frames decoded"}
    arrays = {k: np.stack(v) for k, v in frames.items()}
    tmp = out_file + ".tmp.npz"
    np.savez_compressed(tmp, fps=np.float32(src_fps / step), **arrays)
    os.replace(tmp, out_file)
    present = arrays["hand_present"]
    return {
        **stats,
        "status": "ok",
        "any_hand": round(float(present.any(1).mean()), 3),
        "both_hands": round(float(present.all(1).mean()), 3),
        "pose": round(float(arrays["pose_present"].mean()), 3),
    }


def extract(cfg: Config, limit: int | None = None, force: bool = False, workers: int | None = None) -> pd.DataFrame:
    manifest_dir = path(cfg, "manifests")
    manifest = pd.read_csv(manifest_dir / "videos.csv")
    manifest = manifest[manifest["readable"].fillna(False).astype(bool) & manifest["label"].notna()]
    models_dir = path(cfg, "mediapipe_models")
    for name in (HAND_MODEL, cfg.extract.pose_model):
        if not (models_dir / name).exists():
            raise SystemExit(f"Missing {models_dir / name}. Run `make models` first.")

    out_dir = path(cfg, "landmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    root = path(cfg, "raw_videos")
    jobs = [
        (r.clip_id, str(root / r.path), str(out_dir / f"{r.clip_id}.npz"))
        for r in manifest.itertuples()
        if force or not (out_dir / f"{r.clip_id}.npz").exists()
    ]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(manifest)} labelled videos, {len(jobs)} to extract")

    e = dict(cfg.extract)
    results = []
    with ProcessPoolExecutor(max_workers=workers or e["workers"]) as pool:
        futures = {pool.submit(process_clip, c, v, o, str(models_dir), e): c for c, v, o in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            try:
                results.append(fut.result())
            except Exception as exc:  # one broken video must not stop the run
                results.append({"clip_id": futures[fut], "status": f"error: {exc}"})

    log = manifest_dir / "extraction.csv"
    df = pd.DataFrame(results)
    if log.exists() and not df.empty:
        old = pd.read_csv(log)
        df = pd.concat([old[~old["clip_id"].isin(df["clip_id"])], df], ignore_index=True)
    elif log.exists():
        df = pd.read_csv(log)
    df.to_csv(log, index=False)
    if len(df):
        ok = df[df["status"] == "ok"]
        print(f"ok: {len(ok)}/{len(df)}; hands visible in {ok['any_hand'].mean():.0%} of frames on average; "
              f"pose found in {ok['pose'].mean():.0%}")
    return df

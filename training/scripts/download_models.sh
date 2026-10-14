#!/usr/bin/env bash
# Downloads the MediaPipe models that `ashiria extract` runs on every video frame.
set -euo pipefail
cd "$(dirname "$0")/.."

dest=models/mediapipe
base=https://storage.googleapis.com/mediapipe-models
mkdir -p "$dest"

curl -fL --progress-bar -o "$dest/hand_landmarker.task" \
  "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -fL --progress-bar -o "$dest/pose_landmarker_full.task" \
  "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

ls -lh "$dest"

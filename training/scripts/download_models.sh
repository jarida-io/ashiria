#!/usr/bin/env bash
# Downloads the MediaPipe models that `ashiria extract` runs on every video frame, and checks them
# against scripts/models.sha256 (the versions every landmark file so far was made with).
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
if ! shasum -a 256 -c scripts/models.sha256; then
  echo "WARNING: Google has published new model versions. Landmarks extracted with them won't exactly match" \
       "earlier extractions. Re-extract everything, or restore the old files." >&2
  exit 1
fi

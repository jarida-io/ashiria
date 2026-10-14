# Feature spec v1

The contract between the training pipeline (Python) and the headset (C#). The model only works if both sides turn hands into numbers in exactly the same way. `features.py` and `canonical.py` are the reference implementation. Every exported model records `spec_version` in `model_meta.json`, and the app must refuse a model whose version it doesn't implement.

## 1. Conventions

- **Reference frame:** origin at the head, x = signer's right, y = up, z = signer's forward. This is left-handed, the same as Unity's axes.
- **Units:** hand shapes in metres; hand locations in **shoulder widths** from the head.
- **Hand slots:** index 0 and 1 in every array. Before canonicalisation, 0 = signer's right hand, 1 = left. After it (§5.1), 0 = dominant hand, 1 = the other hand.
- **Time:** 30 frames per second in, a fixed number of frames out (§5.4).

## 2. Model input and output

- Input `features`: float32 `[1, 32, 197]`, raw values from §6. Normalisation is built into the model.
- Output `logits`: float32 `[1, C]`. Class names are in `labels.json`, in output order.
- Probabilities: `softmax(logits / temperature)`, with `temperature` from `model_meta.json`.

## 3. Joint layout

Canonical layout: MediaPipe's 21 hand landmarks.

| # | Joint | Quest (OpenXR) index |
|---|---|---|
| 0 | wrist | 1 |
| 1–4 | thumb CMC, MCP, IP, tip | 2, 3, 4, 5 |
| 5–8 | index MCP, PIP, DIP, tip | 7, 8, 9, 10 |
| 9–12 | middle MCP, PIP, DIP, tip | 12, 13, 14, 15 |
| 13–16 | ring MCP, PIP, DIP, tip | 17, 18, 19, 20 |
| 17–20 | pinky MCP, PIP, DIP, tip | 22, 23, 24, 25 |

On Quest, use `OVRSkeleton` with the OpenXR hand skeleton (`XRHand_*`, 26 joints). The palm (0) and the four finger metacarpals (6, 11, 16, 21) are unused. OpenXR places each joint at the base of the bone it names, which is why `INDEX_PROXIMAL` (7) is the index MCP knuckle. The same mapping is in `joints.json`.

## 4. From each source to a segment

A **segment** is a run of frames with, for each frame and hand slot:
- `hand_rel` `[21,3]`: joints minus the wrist, rotated into reference-frame axes, in metres;
- `wrist_pos` `[3]`: the wrist in the reference frame, in shoulder widths;
- `present`: 0 or 1;
- plus `fps` for the whole segment.

### 4.1 Training videos (Python, `canonical.from_mediapipe`)

1. **Body frame from MediaPipe pose world landmarks.** Frames without a pose borrow the nearest frame that has one.
   - x = normalise(right shoulder − left shoulder).
   - y = the part of (ear midpoint − shoulder midpoint) perpendicular to x, normalised. Ears, not the nose: the nose sits in front of the shoulders and would tilt "up" forward by about 20°, while the headset's up is gravity.
   - z = cross(y, x), flipped if needed so it points the way the face does, i.e. (nose − ear midpoint) · z > 0.
2. **Origin and scale.** Origin = nose. Scale = shoulder distance in metres.
3. **Hand shape.** `hand_rel` = rotate (hand world landmarks − hand wrist) into the frame.
4. **Hand location.** `wrist_pos` = rotate (pose wrist − nose) into the frame, then divide by shoulder width. Pose landmark 16 goes to slot 0, landmark 15 to slot 1.
5. **Slots.** Detected hands go to the nearest pose wrist. Without a pose, MediaPipe's handedness label decides, remembering it assumes a mirrored image.

### 4.2 Headset (C#)

1. **Head frame each frame.** Origin = centre-eye position. y = world up. z = head forward with its vertical part removed, normalised. x = cross(y, z) in Unity.
2. **Scale.** The user's shoulder width, default 0.38 m. It can be calibrated by asking the user to hold both hands out at shoulder width.
3. **Hand shape.** `hand_rel` = for the 21 mapped joints, (joint − wrist) expressed in (x, y, z).
4. **Hand location.** `wrist_pos` = (wrist − centre eye) in (x, y, z), divided by the scale.
5. **Presence.** `present` = 1 only when all of these hold:
   - the hand is tracked;
   - its data is valid and high confidence;
   - it isn't an inferred Wide Motion Mode pose;
   - no system gesture is in progress.
6. **Slots.** Slot 0 = right hand, 1 = left, then §5.1.

**Known approximations.** Augmentation during training is sized to cover these:
- the nose versus the centre eye;
- shoulders versus head yaw;
- MediaPipe versus Quest joint placement.

## 5. Canonicalisation (shared)

### 5.1 Dominant hand

If the signer is left-dominant, mirror:
- negate x in `hand_rel` and `wrist_pos`;
- swap slots 0 and 1.

Who counts as left-dominant:
- **Training:** decided per signer by majority vote of which wrist travels further.
- **Headset:** from the user's handedness setting.

### 5.2 Gap fill

For each slot, a run of at most 5 untracked frames between two tracked frames is filled. Both `hand_rel` and `wrist_pos` are interpolated linearly, and `present` is set to 1.

### 5.3 Trim

1. A frame is **active** if some present hand has `wrist_pos.y` ≥ −1.8.
2. Keep from the first to the last active frame, plus 2 frames of margin on each side.
3. If no frame is active, fall back to the frames where a hand is present.

On the headset the same rule runs over a buffered attempt window, starting after the prompt and ending when the hands drop or hold still.

### 5.4 Resample to 32 frames

1. Source times = linspace(0, L − 1, 32).
2. In untracked frames, positions take the nearest tracked frame's value before interpolating.
3. Positions then interpolate linearly; `present` takes the nearest frame.
4. New fps = 31 / ((L − 1) / fps). So dt = 1 / new fps, and speed information is preserved.

## 6. Per-frame features (197 floats)

Two hand blocks of 97, then 3 inter-hand values. A missing hand's block is all zeros.

**Hand axes**, from `hand_rel`:
- y = normalise(middle MCP − wrist);
- n = cross(index MCP − wrist, pinky MCP − wrist);
- x = normalise(cross(y, n));
- z = cross(x, y), which is the palm normal;
- palm = |middle MCP − wrist|.

| Offset in block | Count | Feature |
|---|---|---|
| 0 | 60 | Joints 1–20 in hand axes, divided by palm: (x·p, y·p, z·p) / palm, in joint order |
| 60 | 15 | Bend angles (radians) at each finger's three joints, thumb first. For chain wrist, j1, j2, j3, tip: the angle between consecutive bones at j1, j2 and j3 |
| 75 | 4 | Spread angles between first bones: thumb (MCP − CMC) vs index (PIP − MCP); index vs middle; middle vs ring; ring vs pinky |
| 79 | 5 | Fingertip distances / palm: thumb–index, thumb–middle, thumb–ring, thumb–pinky, index–middle |
| 84 | 3 | Palm normal z in the reference frame |
| 87 | 3 | Hand direction y in the reference frame |
| 90 | 3 | `wrist_pos` |
| 93 | 3 | Wrist velocity: (wrist_pos[t] − wrist_pos[t−1]) / dt when the hand is present in both frames, otherwise 0. Frame 0 is 0 |
| 96 | 1 | Present (1) |

- **Dominant hand:** block at offset 0.
- **Other hand:** block at offset 97.
- **Offset 194:** other wrist − dominant wrist when both are present, otherwise 0.

Numerics: use ε = 1e-6 when normalising (v / max(|v|, ε)). Compute every angle as atan2(|u × v|, u · v), never arccos: it stays accurate in 32-bit floats for nearly straight fingers.

## 7. Parity tests

`ashiria export` writes `golden.json` with a few test clips. Each clip has:
- the canonical segment, before resampling;
- the expected features;
- the expected logits.

The Unity edit-mode tests must:
1. resample the segment and compute features, matching within 1e-3;
2. run the model in Sentis, matching the logits within 5e-3.

## 8. Changing this spec

Any change to §3–§6 means all of these:
- bump `SPEC_VERSION` in `features.py`;
- update the C# implementation;
- rebuild the cache;
- retrain and re-export the model.

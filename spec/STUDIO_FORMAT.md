# Studio take format, v1

Ashiria Studio (the internal Quest recorder) saves one JSON file per take. `ashiria import-studio` turns
takes into reference signs that replace the KSL Dictionary drafts, sign by sign.

A take keeps **both** the raw tracking and the canonical frames the app computed:

- **Raw world poses** let us re-run canonicalisation if the spec changes (head frame, shoulder width,
  joint use) without another recording session with the co-creators.
- **The app's canonical frames** give a free on-device parity check: the importer recomputes them in
  Python and warns if they differ.
- **Joint rotations** keep a skinned shadow hand possible later (packs carry positions only).

## Where takes live

- On the headset: `Application.persistentDataPath/studio/<session_id>/<sign_id>__<n>.json`. One file per
  take, so a crash loses at most one take.
- Pulled over USB into `training/data/studio/` (git-ignored). Takes are biometric data of identifiable
  team members: they never go into git or any cloud service, and are deleted from the headset after import.

## File

```json
{
  "format": "ashiria.studio-take", "format_version": 1,
  "take_id": "uuid", "session_id": "2026-11-02-a",
  "sign_id": "letter-a",               // a catalogue id, or "_none" for deliberate non-signs
  "pack_id": "ksl-core", "pack_version": "0.1.0",
  "signer": "S01",                     // pseudonymous id only, never a name
  "role": "reference",                 // reference | test (held-out attempts for calibration)
  "handedness": "right",               // right | left
  "consent_ref": "ashiria-studio-consent-v1",
  "shoulder_width_m": 0.38, "shoulder_calibrated": false,
  "rest_height": -1.5,                 // the signer's rest level (shoulder widths), from onboarding
  "device": "Quest 3S", "app_version": "0.1.0",
  "spec_version": 1, "runtime_version": 1, "joint_layout": "openxr26",
  "tick_hz": 30, "recorded_at": "2026-11-02T10:15:00Z",
  "trim": {"start": 12, "end": 88},    // frames kept, end exclusive (Studio's auto cut, adjustable)
  "status": "keep",                    // keep | reject
  "frames": {
    "n": 120,
    "t": [0.0, 0.0333, ...],                       // seconds since the take started, n values
    "head_pos": [...], "head_rot": [...],         // n×3 and n×4 (x, y, z, w), Unity world space
    "right": {
      "tracked": [...], "high_conf": [...], "inferred": [...], "system_gesture": [...],  // n values, 0/1
      "pos": [...],                                // n×26×3 metres, Unity world space, OpenXR joint order
      "rot": [...]                                 // n×26×4 (x, y, z, w)
    },
    "left": { "...": "same as right" },
    "canonical": {"hand_rel": [...], "wrist_pos": [...], "present": [...]}   // n×2×21×3, n×2×3, n×2
  }
}
```

All arrays are flat, row-major, in the order the shapes are written. World space is Unity's
(left-handed: x right, y up, z forward), in metres. Joint positions must be world-space poses
(`OVRBone.Transform.position` or the Interaction SDK's `IHand.GetJointPose`), never local positions.

## Rules

- **Fixed tick.** The app samples the latest tracking at a fixed 30 Hz (an accumulator in `Update`),
  exactly as the live attempt recorder does. `t` records the real sample times.
- **Hold static signs for about a second.** Tags (static or moving, one or two hands) come from the take,
  and a hold shorter than the rise into it reads as movement, which switches on the movement check.
- **Presence.** A hand counts as present only when it is tracked with high confidence, its pose is not
  inferred, and it is not performing the system gesture (spec §4.2).
- **Canonical frame.** Head-yaw axes (y = world up, z = head forward flattened), origin at the centre eye
  plus `NoseOffset` (0, −0.045, 0.025) m in those axes, wrist positions divided by `shoulder_width_m`.
  Left-handed signers are mirrored so slot 0 is always the dominant hand.

## Import (`ashiria import-studio <dir>`)

1. Validate each file (format, version, array sizes, known sign id).
2. Rebuild canonical frames from the raw poses (`quest.py`). Warn if they differ from the app's by more
   than 1e-3.
3. Keep frames `trim.start` to `trim.end`. Cut them like a live attempt (`pipeline.cut_reference`, with
   the take's `rest_height`).
4. Quality: the dominant hand is present in at least 80% of kept frames; 0.3–5 s long.
5. Per sign, with three or more takes: drop a take whose mean DTW distance to the others is more than
   twice the median (`--keep-outliers` keeps them). The shadow take is the medoid.
6. Write `training/exports/studio/store.json` (reference takes by sign id) and
   `training/exports/studio/attempts.json` (`role: test` and `_none` takes, for calibration).

`ashiria pack` then uses Studio takes for every sign that has them, instead of the dictionary draft.

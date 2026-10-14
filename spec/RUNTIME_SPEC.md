# Runtime spec, v1

How the headset turns live hand tracking into "you signed it" / "bend your ring finger" / "that was
HELLO". This builds on `FEATURE_SPEC.md` (frames, joints, features), which is unchanged.

- **Python reference:** `training/src/ashiria_train/` (`params.py`, `pipeline.py`, `segmenter.py`,
  `signs.py`, `verify.py`, `recognise.py`, `quest.py`).
- **C# port:** `sdk/unity/com.jarida.ashiria.engine/Runtime/`.
- **Parity:** `spec/golden/runtime_v1.json` and the synthetic mini pack `spec/golden/pack_v1/`, checked by
  `dotnet test sdk/unity/com.jarida.ashiria.engine/Tests~/dotnet` and the Unity EditMode tests.

Every number below is a default in `EngineParams`. Packs carry the values they were built with
(manifest `engine`), and the headset uses the pack's values.

## 1. Frames

The app samples tracking at a **fixed 30 Hz tick**: an accumulator in `Update` takes the latest hand pose
once per 1/30 s. Each tick, `QuestCanonical.Frame` builds one canonical frame:

- **Axes:** head yaw only (y = world up, z = head forward flattened). Looking straight up or down keeps the
  previous tick's axes.
- **Origin:** the centre eye plus `NoseOffset` (0, −0.045, 0.025) m in those axes. This approximates the nose
  tip, the origin of the video references.
- **Scale:** wrists are in shoulder widths (0.38 m unless calibrated).
- **Presence:** a hand counts only when tracked with high confidence, not pose-inferred, and not doing
  the system gesture.
- **Left-handed users** are mirrored, so slot 0 is always the dominant hand.

## 2. Prepare

`Pipeline.Prepare(segment, params, restHeight)`: gap fill (≤ 5 frames) → trim to signing space (wrist
above `restHeight`, ±2 frames) → active window (cut the rise from and drop to rest: keep frames above 30%
of the dominant wrist's height range, if that range is at least 0.3 shoulder widths) → resample to 32
frames.

References were cut the same way when the pack was built. `cut_reference` streams the recording through
the segmenter (§3), then prepares it. So a held sign ends at its hold in both references and attempts.

**Rest height is per user.** At onboarding, take the median resting wrist height and add 0.25, clamped to
[−1.9, −1.2] shoulder widths. A seated user rests higher than the standing signers in the videos (−1.8).

## 3. Attempt segmentation

`AttemptSegmenter.Push(handRel, wrist, present)` once per tick. It returns an `Attempt` when one ends.

| State | Moves on when |
|---|---|
| idle | Hand up (any present wrist ≥ rest height) for 0.10 s → active, keeping 0.20 s of pre-roll |
| active | Hand down or lost for 0.13 s → **drop**. Dominant hand still for 0.6 s → **hold**. 4 s → **timeout** |
| cooldown (after hold or timeout) | Hand below rest for 0.13 s, or wrist ≥ 0.3 shoulder widths from where it was held, not downwards → idle |

**Still** means that, over the last 0.6 s, the wrist stays within 0.12 shoulder widths of its median. The
15 bends must also stay within 0.35 rad (mean absolute deviation from their medians). At most 10% of
frames may break this. Attempts with under 0.35 s of hand-up frames are dropped.

The cooldown rule stops one held sign firing twice. Lowering the hand after a sign is not a new attempt.

## 4. Verify

`Verifier.Verify(prepared, sign, params, toleranceScale, lookalikes)` returns:
- a status: `pass`, `fail`, `unscored` (watch-only signs) or `no_hands` (dominant hand in < 60% of frames);
- a score from 0 to 1, with 0.5 as the pass line;
- components: error ÷ tolerance, where ≤ 1 passes;
- signed finger errors;
- hints.

1. **Pick the take.** Align the attempt to each of the sign's takes with DTW (band 8) and keep the nearest.
2. **Key pose.** Take the attempt frames aligned to that take's key frame, plus 2 either side, and use the
   per-component median of their features.
3. **Check components** against the take's own key pose. Tolerances are per sign, from the pack, × the
   user's `toleranceScale` (strictness):

| Component | Error | Notes |
|---|---|---|
| shape | per finger, the largest bend difference ÷ that finger's tolerance; the worst finger | |
| spread | largest difference between neighbouring fingers | off when 0 |
| orientation | max(palm-normal angle, hand-direction angle), degrees | |
| location | key-frame wrist distance, forward/back counting half | |
| movement | 2 × DTW between the wrist paths (relative to the key-frame wrist) | moving signs only |
| other_hand | other wrist relative to the dominant one; 2.0 if the other hand is missing | two-handed signs only |
| overall | DTW distance ÷ (tau × distance_z) | |
| lookalike | 0.9 × distance ÷ the nearest look-alike's distance | fires when a look-alike is clearly closer |

**Hints**, most severe first, are codes the app turns into words:

| Code | Extra field |
|---|---|
| `tracking.lost` | |
| `finger.bend_more`, `finger.straighten` | finger |
| `spread.more`, `spread.less` | only when the shape is right |
| `palm.face`, `hand.point` | direction |
| `move` | direction |
| `movement.bigger`, `movement.smaller`, `movement.path` | |
| `other_hand.missing`, `other_hand.place` | direction |
| `overall.different` | |
| `lookalike` | sign_id |

Directions are canonical (`dominant_side`, `other_side`, `up`, `down`, `forward`, `back`). They read
correctly for left-handed users, whose data is mirrored.

## 5. Recognise (Free Sign)

`Recogniser.Recognise(prepared, candidates, params)`:

- **Distances.** Each candidate's distance is the DTW to its nearest take. z = distance ÷ tau, where tau is
  the 90th percentile of correct simulated attempts, so z ≈ 1 is a normal correct attempt. Confidence is
  softmax(−z / T).
- **Decision:**
  - `no_hands` if the dominant hand is present in < 50% of frames;
  - `not_sure` if best z > `accept_z`;
  - `ambiguous` (show both) if runner-up z − best z < `min_margin_z` and runner-up z ≤ `accept_z`;
  - otherwise `match`.
- **Why an absolute gate:** with only a few learned signs, one of them always wins a softmax.
- **Candidates:** the signs whose catalogue `recognise` is true (all but watch-only by default).

## 6. Calibration and tiers

`ashiria calibrate`, run by `ashiria pack`, simulates "someone else" signing each reference. It applies
small rotations, body-size and finger-shape changes, jitter, speed changes and dropouts. Hands rise from
rest first and hold or drop afterwards. Every attempt then goes through the real segmenter and
`prepare`.

**What it sets:**
- **tau** per sign.
- **Free Sign thresholds.** They maximise correct matches while keeping wrong matches ≤ 5% and out-of-set
  attempts accepted ≤ 10%. Out-of-set means other vocabulary signs when only 15 are learned, 300
  non-vocabulary dictionary signs, and random movement.
- **The temperature.**

**Tiers** (`configs/tiers.yaml`) are decided in this order:
1. A co-creator override always wins.
2. **Watch-only:** tracking is poor, hands overlap, the face carries meaning, or the sign is rarely found.
3. **Lenient** (about 1.4× wider, look-alike guard kept): near the face, two hands, handshape change, a
   known hard handshape, or the scored check fails too many correct attempts.
4. **Scored:** everything else.

**Simulated results, 14 Oct 2026** (72 dictionary drafts, one signer, so optimistic):

| Measure | Result |
|---|---|
| Free Sign | 98% correct, 0–1% wrong, 6% of out-of-set attempts accepted |
| Verify, correct attempts that pass | 91% |
| Verify, look-alikes that pass | 1% |

The same command will calibrate from real Quest attempts (Studio `role: test` takes) once they exist.
Real numbers will be lower.

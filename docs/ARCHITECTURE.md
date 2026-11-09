# Architecture

Ashiria reads hands, never pixels. On Quest the headset tracks your hand joints; on video or a webcam, MediaPipe
finds the same joints. From there every app uses one engine, so a sign is read the same way everywhere.

```
Quest hand tracking (25 joints a hand)         MediaPipe on video or a webcam (21 joints, plus the body)
              │                                               │
              └──────────────► canonical frames ◄─────────────┘
                     body-centred: origin at the nose, units of shoulder widths, x to the signer's right,
                     y up, z the way they face; 30 frames a second; left-handed signers mirrored
                                      │
                         attempt segmenter: idle → hands up → ends on a hold, a drop or a timeout
                                      │
                   prepare: gap fill → trim at the user's rest height → active window → resample
                          │                                            │
          Verify (Practice): is this sign X?                Recognise: which sign is this?
          compare with X's reference takes (DTW), then      nearest reference (course signs), or the
          handshape, spread, palm, place, movement and      embedding model over every dictionary sign;
          the other hand → pass, or hints per finger        an answer, or "not sure"
```

## The engine

One design, three implementations, kept identical by shared test vectors:

| Implementation | Where | Used by |
|---|---|---|
| Python (the reference) | [`training/src/ashiria_train/`](../training/src/ashiria_train) | Building references, calibration, packs, training |
| TypeScript | [`sdk/web/ashiria-engine/`](../sdk/web/ashiria-engine) | The WebXR app and the web tools |
| C# | [`sdk/unity/com.jarida.ashiria.engine/`](../sdk/unity/com.jarida.ashiria.engine) | A Unity fallback |

- **Features** ([spec](../spec/FEATURE_SPEC.md)): each hand gives 97 numbers a frame (joints in the hand's own frame,
  finger bends and spreads, fingertip gaps, palm normal and direction, wrist position and velocity, presence), plus
  where one hand is relative to the other: 197 in all.
- **Runtime** ([spec](../spec/RUNTIME_SPEC.md)): the frame builder, segmenter, Verify and Recognise, with their
  parameters.
- **Parity:** `training` writes golden vectors from synthetic hands into [`spec/golden/`](../spec/golden); the
  TypeScript and C# tests must reproduce them.

## Content packs

What the app teaches arrives as a content pack ([spec](../spec/CONTENT_PACK.md)), built by `ashiria pack` from the
catalogue and the references:

| File | Holds |
|---|---|
| `manifest.json` | Versions, rights, engine parameters (calibrated), file hashes, a self-check the app recomputes |
| `catalogue.json` | Units, signs (tier, tags, look-alikes, review), phrases (signs in KSL order) |
| `references.json` | Each sign's reference takes (hand poses only) and its tolerances |
| `coach.json` | The avatar coach's motion for each sign, and its step captions |

The app refuses a pack whose hashes or self-check don't match. A production build bundles only *release* packs,
which hold only approved signs whose sources have permission.

## The app

[`apps/quest-learn/`](../apps/quest-learn) is an Immersive Web SDK app: entities and systems over three.js, in a
WebXR session with hand tracking and passthrough.

| System | Does |
|---|---|
| Hand sampler | WebXR joints and head pose → canonical frames at a fixed 30 Hz → attempts |
| Shadow hands | The current sign's reference, animated where your hands should be; fingers that are off light up |
| Avatar coach | A figure facing you that signs the current sign step by step, slowly, with captions, from the pack |
| Lesson | The lesson panel: Practice (Verify with hints), Free Sign (Recognise), phrases part by part, settings |
| App state | The pack, the lesson's signs and phrases, settings and progress, kept in the browser on the headset |
| Studio | Development only: every attempt saved as a reference take on the developer's Mac |

## The recogniser and the web tools

The recogniser ([Recognition](RECOGNITION.md)) is an embedding model over all 2,646 KSL Dictionary signs, exported to
ONNX and run in the browser with ONNX Runtime Web. It reads each attempt as it is and mirrored, so a left-handed
signer or a mirrored camera needs no setting. It powers the development webcam preview today.

Development pages, served by the dev server to this machine only (the dictionary's videos never leave it):

| Page | Shows |
|---|---|
| `/dev/study/` | The study sheet: each course sign's video next to its reference hands, with Approve and Reject |
| `/dev/study/browse` | Every dictionary sign: video with MediaPipe's landmarks, the cut reference, the avatar, step captions, audit flags and the recogniser's results |
| `/dev/study/results` | The model check: dictionary clips played through the web app's own pipeline, next to what it read |
| Start page → *Start with my webcam* | The emulator's hands driven by your webcam, with the recogniser reading your signs |

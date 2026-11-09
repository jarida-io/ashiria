# Ashiria

**Practise Kenyan Sign Language with your own hands.** Ashiria is a hands-first KSL practice app for Meta Quest:
you fit your hands into translucent *shadow hands*, an avatar coach shows each sign step by step, and the app
checks every finger and tells you what to fix. Everything runs on the headset, and your hand data never leaves it.

*Ashiria* is Swahili for "to signal" or "to indicate".

> **Status (November 2026):** Ashiria Learn is our entry to the Meta VR Start Developer Competition 2026, built with
> Meta's Immersive Web SDK (WebXR) and submitted as a hosted link. It is a learning aid for hearing family, friends
> and colleagues of Deaf people, not an interpreter.

## What it does

| | |
|---|---|
| **Shadow hands** | The sign you're learning, animated where your own hands should be. Fingers that are off turn orange. |
| **Per-finger feedback** | Each attempt is checked handshape, palm, place and movement, with hints like "bend your ring finger". |
| **Avatar coach** | A signer stands in front of you and signs slowly, one step at a time, with a caption for each step. |
| **Course** | 75 signs in six units (fingerspelling, numbers, everyday signs, meeting people, asking again, needs). |
| **Phrases** | 18 everyday phrases in KSL word order ("NAME YOUR WHAT?"), practised sign by sign. |
| **Practice loop** | Copy the shadow, then sign from memory; mastery builds over several days, with a streak. |
| **Free Sign** | Sign anything you've learned; the app tells you what it saw, or says it isn't sure. |
| **Private by design** | No account, no analytics, nothing uploaded. Progress stays on the headset ([privacy](docs/privacy.md)). |

A development web preview reads signs from a laptop webcam with a recogniser that knows all 2,646 signs in the KSL
Dictionary's videos ([results](docs/RECOGNITION.md)).

## Quick start

You need Node.js 22 and, for the training tools, Python 3.12 and `make`.

```bash
cd apps/quest-learn && npm install
```
```bash
npm run dev
```

This starts the dev server (https://localhost:8081) and opens the app in a browser with Meta's headset emulator;
**Start** enters the emulated headset. To use a Quest, open the dev server's network address in the Quest Browser
on the same Wi-Fi. The app needs a content pack, which is built from local sign data that isn't in this repository:
see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Run the tests:

```bash
make -C training test
```
```bash
node --test sdk/web/ashiria-engine/test/ && (cd apps/quest-learn && npm run typecheck && node --test src/*.test.ts)
```

## How it works

```
Quest hand tracking (or MediaPipe on a camera)
   → body-centred frames at 30 Hz → attempt cutter (hands up … hands down)
   → Verify: compare with the sign's reference, finger by finger → pass, or hints
   → Recognise: which learned sign is this? → answer, or "not sure"
Content pack: catalogue (signs, units, phrases) + reference hand poses + the avatar coach's motion and steps
```

One engine, written once as a Python reference and ported to TypeScript (and C#), checked against shared test
vectors so every app reads signs the same way. More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository

| Folder | What's in it |
|---|---|
| [`apps/quest-learn/`](apps/quest-learn) | Ashiria Learn: the WebXR app (Immersive Web SDK, TypeScript, three.js), plus dev-only tools |
| [`sdk/web/ashiria-engine/`](sdk/web/ashiria-engine) | The TypeScript engine: frames, segmenter, verify, recognise, content packs |
| [`sdk/unity/`](sdk/unity/com.jarida.ashiria.engine) | The same engine in C#, kept as a Unity fallback |
| [`training/`](training) | Python: video landmarks → references, the engine's reference implementation, calibration, content packs, the recogniser |
| [`spec/`](spec) | Shared contracts: [features](spec/FEATURE_SPEC.md), [runtime](spec/RUNTIME_SPEC.md), [content packs](spec/CONTENT_PACK.md), [Studio takes](spec/STUDIO_FORMAT.md), synthetic test vectors |
| [`content/`](content) | The course: [catalogue](content/catalogue/ksl-core.yaml) (signs, steps, phrases, reviews), [phrase list](content/phrases/ksl-core.yaml), [rights](content/rights.yaml) |
| [`docs/`](docs) | [Documentation](docs/README.md) |

## Data and rights

Sign videos show real people, so they stay on the developer's machine: no video, video frame or landmark file is in
this repository, and none is uploaded anywhere. The course's reference hand poses are derived, with its owners'
permission, from the [KSL Dictionary](https://ksldictionary.com)'s videos; the app ships hand poses only. The
phrases follow Kenyan teaching material and most still need a Deaf KSL signer's review before release. Details in
[docs/DATA_AND_RIGHTS.md](docs/DATA_AND_RIGHTS.md).

## Licence

Code and documentation: [Apache 2.0](LICENSE). Sign data (videos, landmarks, references, models trained on them)
is not part of this repository and is not covered by this licence.

Ashiria is made by [Jarida](https://jarida.io). It is a new app, started in October 2026; Jarida's
earlier KSL work, the 2025 Android app KSL Translator, is at
[jarida-io/ksl-translator-2025](https://github.com/jarida-io/ksl-translator-2025).

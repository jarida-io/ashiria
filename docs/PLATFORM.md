# Platform and roadmap

Ashiria starts as one app, Ashiria Learn on Quest, on a base built to serve more: one sign catalogue, one engine and
one data pipeline, so a sign recorded, reviewed or improved once improves every app.

## The apps

| App | Device | For | What it does | Status |
|---|---|---|---|---|
| **Ashiria Learn** | Quest (WebXR) | Hearing people learning KSL | Shadow hands, an avatar coach with step captions, per-finger feedback, phrases sign by sign, Free Sign, progress and streaks | Meta VR Start 2026 entry |
| **Ashiria Studio** | Quest (development builds) | The team | Record reference signs in the headset, three takes each, saved to the developer's machine | Internal tool |
| **Ashiria Interpret** | Web, then phone and Quest | Conversations between signers and non-signers | Read signs from a camera and show what was signed | A development web preview reads single signs; reading signs in a row is experimental ([recognition](RECOGNITION.md)) |
| **Ashiria Contribute** | Phone | Signers who want to help | Record consented sign videos for review, so the references come from many signers, not one | Planned |

## The shared base

| Part | What it is | In this repo |
|---|---|---|
| Sign catalogue | Units, signs, step captions, phrases, reviews; shipped as versioned content packs | `content/`, [spec](../spec/CONTENT_PACK.md) |
| Engine | Canonical frames, the attempt segmenter, Verify and Recognise: Python reference, TypeScript and C# ports | `training/`, `sdk/web/`, `sdk/unity/`, [spec](../spec/RUNTIME_SPEC.md) |
| Models | The recogniser (ONNX), trained and tested in `training/` | [Recognition](RECOGNITION.md) |
| Data pipeline | Videos → landmarks → references and packs → training → evaluation → export | `training/` |

## How far recognition goes, honestly

| Stage | Reads | Status |
|---|---|---|
| Course signs, one at a time | The signs you are learning, with per-finger feedback | In Ashiria Learn |
| Any dictionary sign, one at a time | 2,646 signs from one signer's videos | Development preview; untested on other signers |
| Signs in a row | Short phrases | Experimental: works on joined recordings, untested on real running signing |
| Sentences with facial grammar | Questions, negation and other grammar carried by the face | Research: hand tracking alone can't see it |

Ashiria is a learning and practice aid, not an interpreter: Deaf interpreters stay essential. Its references come from
one signer, so every recognition result is described with that limit.

## Roadmap

| Phase | Delivers |
|---|---|
| **Competition** (to 18 Nov 2026) | Ashiria Learn: six units, phrases, the avatar coach, Free Sign, hosted as a WebXR link |
| **After** | Deaf KSL signers reviewing the phrases and step captions; Studio recordings by more signers; accuracy measured on people other than the dictionary's signer |
| **Then** | Interpret on the web and phone; Contribute, with consent and review, to grow the references beyond one signer |

# Ashiria platform

Ashiria helps people learn, practise and **recognise** Kenyan Sign Language, built with Deaf KSL signers. It is several apps on one shared base. The Quest learning app is the Meta VR Start entry (18 Nov 2026), and the platform continues after the hackathon.

*Draft for discussion: 14 Oct 2026.*

---

## 1. The shared base

Every app uses the same five parts, so a sign recorded, reviewed or improved once benefits every app.

| # | Part | What it is | Lives in |
|---|---|---|---|
| 1 | **Sign catalogue** | One record per sign (fields below), shipped as versioned **content packs** (e.g. `ksl-core-v1`) that apps download or bundle | `content/` |
| 2 | **Hand-sign engine** | The input contract (feature spec) plus four functions, built once per platform (below) | `spec/`, `sdk/unity/`, `sdk/android/`, `training/` (Python reference) |
| 3 | **Models** | Trained recognisers exported to ONNX (Sentis on Quest, ONNX Runtime or LiteRT on Android), each with metadata and golden test vectors | built by `training/`, shipped in content packs |
| 4 | **Data pipeline** | Ingest (dictionary, phone contributions, Studio recordings) → landmarks → Deaf review → references and content packs → training → evaluation → export | `training/` |
| 5 | **Contribution service** | Our own **PocketBase** server (self-hosted): contributor accounts, uploads, consent records, the review queue for Deaf validators, and content-pack hosting. Apps talk to its REST API | `services/` |

**Catalogue fields:** ID, glosses, category, description, tier (scored, lenient or watch-only), tags (hands, movement, near face), source and rights, reviewer approval, plus its reference recordings and video.

**Engine functions, one implementation per platform (Python, C#, Kotlin):**
- **Verify:** is this sign X? Reference matching with per-finger feedback.
- **Recognise:** which sign is this? Nearest reference now, a trained model later.
- **Tune:** per-user tolerances, kept on the user's device.

**Engine inputs, through adapters:**
- Quest hand tracking (26 joints);
- MediaPipe on camera video, on the phone and in training.

**Rules that apply to every app:**
- Every contribution has recorded consent, and a Deaf reviewer approves every sign before it ships.
- Content only ships when its rights are clear: the KSL Dictionary needs its owners' permission.
- Quest users' hand data never leaves the headset (Meta's policy). Personal tuning is the exception, because it stays on the device.

## 2. The apps

| App | Device | For | What it does | When |
|---|---|---|---|---|
| **Ashiria Learn** | Quest | Learners | Learn with shadow hands, practise with per-finger feedback, **Free Sign** (sign anything you've learned and it tells you what it saw), quizzes, Room Words, sign wall | **Hackathon entry**, then store launch |
| **Ashiria Studio** | Quest | The team and approved Deaf contributors | Record a sign three times, replay it as shadow hands, label it, upload it for review. This is how the catalogue's 3D references get made | Hackathon: internal tool. Public only with Meta's written approval |
| **Ashiria Contribute** | Android | Deaf signers, volunteers, validators | Record short sign videos with consent. Validators review and approve. Grows from the 2025 phone app (its Signer/Validator roles and submit flow) | After the hackathon |
| **Ashiria Interpret** | Android first, Quest later | Conversations | Recognise signs and fingerspelling from the camera and show text or speech. Text to shadow hands in the other direction. Quest version: read another person's signs through the headset cameras | After the hackathon |

**Possible simplification:** merge Contribute and Interpret into one Android app with two modes, which means one codebase and one store listing.

## 3. Recognition ("inferring"), honestly staged

| Stage | Recognises | How | Data it needs |
|---|---|---|---|
| **v1: hackathon** | About 50 signs taught in Learn, one at a time | Compare the attempt with every reference; best match plus confidence; "not sure" below a threshold | References only (dictionary drafts, then Studio recordings) |
| **v2: 2027** | Hundreds of signs, more robust across people | Trained model (`training/`) on contributed multi-signer data | Studio recordings first, then phone contributions once the Android app exists, all reviewed |
| **v3: research** | Short sentences, continuous signing | Sequence models; needs facial grammar too | Much more data; not promised |

**Framing:** "Ashiria recognises signs you sign to it." It's a learning and communication aid, not an interpreter. Deaf interpreters stay essential, and saying so protects the project's trust with the Deaf community.

## 4. How data flows (the flywheel)

```
Phone contributions (videos, consented) ──┐
Studio recordings (team, in headset) ─────┼─▶ review by Deaf validators ─▶ catalogue + references ─▶ content packs ─▶ all apps
KSL Dictionary (only with permission) ────┘                │
                                                          └─▶ training data ─▶ models ─▶ content packs
Learn on Quest: personal tuning stays on the headset; nothing uploaded.
```

## 5. Repository layout (monorepo)

```
ashiria/
  docs/                 PLATFORM.md (this), decisions
  spec/                 shared contracts: FEATURE_SPEC.md, joints.json (exist), catalogue schema
  content/              sign catalogue and content-pack manifests (generated packs are not committed)
  training/             Python pipeline (exists): ingest, extract, references, train, export
  sdk/unity/            C# engine: adapters, features, verify, recognise, tune, Sentis runner
  sdk/android/          Kotlin engine: MediaPipe adapter, features, verify, recognise, model runner
  apps/quest-learn/     Unity project: Ashiria Learn, plus Studio as a contributor-only build
  apps/android/         Contribute and Interpret
  services/             contribution and content-pack backend
```

## 6. Roadmap

**Priority order (agreed 14 Oct):** shared infrastructure and the Quest app first, Android last.

| Phase | When | Delivers |
|---|---|---|
| **1. Hackathon** | Now to 18 Nov | Shared infrastructure: `spec` v1, catalogue for about 50 signs, references (dictionary drafts → Studio recordings), C# engine (verify and recognise). **Ashiria Learn** with Free Sign. **Studio** as an internal tool that saves on the headset (files pulled over USB). The competition build stays offline with no backend |
| **2. Platform** | Dec 2026 to Feb 2027 | PocketBase service (contributor accounts, uploads, review queue, content packs), Studio uploads, content-pack updates in Learn, Quest store launch |
| **3. Android (last)** | 2027 | Contribute and Interpret (from the 2025 phone app), phone contributions, first trained model (recognition v2), camera-based reading of another signer on Quest |

## 7. Decisions for you

1. **Names:** Learn, Studio, Contribute, Interpret. Keep them, rename them, or merge some?
2. **Android:** one app with two modes (Contribute and Interpret), or two apps?
3. **Free Sign in the hackathon build:** must-have or nice-to-have?
4. **Studio:** stay internal, or ask Meta about opening it to approved community contributors?
5. ~~Backend~~ **Decided:** our own PocketBase server. It isn't needed for the competition build.
6. **Languages:** English glosses only, or English and Swahili?
7. ~~Priority~~ **Decided:** shared infrastructure and the Quest app first, Android last.

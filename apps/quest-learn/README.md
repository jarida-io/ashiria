# Ashiria Learn (WebXR)

Practise Kenyan Sign Language with your hands on Meta Quest. Built with Meta's
[Immersive Web SDK](https://iwsdk.dev) (TypeScript, three.js, WebXR) and published as a web link: the Meta VR
Start 2026 entry.

Sign reading runs in [`@ashiria/engine`](../../sdk/web/ashiria-engine), the TypeScript port of the shared engine,
tested against the Python reference. This app reads WebXR, runs the lessons and draws.

## Run

```bash
npm install
```
```bash
npm run dev
```

- **Emulator:** `npm run dev` opens a managed browser with the headset emulator. **Start** enters it. Emulated
  hands are posed, not tracked, so don't tune tolerances with them.
- **Quest:** open the network address that `npm run dev:status` prints, on the same Wi-Fi, and accept the local
  certificate.
- **Webcam:** *Start with my webcam* on the start page drives the emulator's hands from your webcam and reads your
  signs with the recogniser (development only).
- **Content:** in development the app loads the newest pack from `../../content/packs/`
  (`make -C ../../training pack`). Development packs are served to this Mac and your headset only.

## What's where

| File | What it does |
|---|---|
| `src/landing.ts` | The start page: what you need, Start, Start over (shared headsets), privacy |
| `src/hand-sampler.ts` | WebXR hand joints and head pose → canonical frames at 30 Hz → the attempt segmenter |
| `src/lesson.ts` | Practice (Verify with per-finger hints), Free Sign (Recognise), Studio; the lesson panel |
| `src/flow.ts` | The practice loop and progress as pure functions: watch, match, recall; phrases part by part (`flow.test.ts`) |
| `src/shadow-hands.ts` | The target sign played where your hands should be; wrong fingers turn orange |
| `src/avatar-coach.ts`, `src/avatar/` | The avatar coach: a figure that signs the current sign step by step, with captions |
| `src/app-state.ts`, `src/content.ts` | The pack, mode, settings and progress (on this device only) |
| `src/studio.ts` | Studio takes saved through the dev server (development builds only) |
| `src/webcam/` | Development: MediaPipe in the browser, a steadier tracker, webcam-driven emulator hands, the recogniser |
| `src/results/` | The model check page |
| `public/ui/lesson.uikitml` | The lesson panel (follows your head; poke or pinch its buttons) |
| `vite.ashiria.ts` | Dev-only routes (packs, Studio, study pages, recogniser); the build guard |

## Development pages

With `npm run dev` running, on this Mac only (the dictionary's videos never leave it):

| Address | For |
|---|---|
| `https://localhost:8081/dev/study/` | Each course sign next to its reference hands. **Approve**, **Reject** and **Back to draft** record your review in the catalogue |
| `https://localhost:8081/dev/study/browse` | Every dictionary sign: landmarks over the video, the reference, the avatar with steps, recogniser results |
| `https://localhost:8081/dev/study/results` | The model check: dictionary clips read by this app's own pipeline, with frames and totals |

## Studio (recording reference signs)

Open `https://<mac>:8081/?studio` on a development build. Pick a sign with Prev and Next, then sign it. Each
attempt the segmenter cuts is saved as a take in `training/data/studio/<date>/`. Hold static signs for about a
second. Then:

```bash
make -C ../../training studio-import STUDIO=data/studio/<date>
```
```bash
make -C ../../training pack
```

## Tests

```bash
npm run typecheck
```
```bash
node --test src/*.test.ts
```

## Release

1. `../../training/.venv/bin/ashiria pack --release`: only approved signs and phrases, and dictionary-derived
   signs only with the permission recorded in `content/rights.yaml`.
2. Copy it into `public/packs/` (git-ignored). The build refuses any pack that isn't a release pack.
3. `npm run build`, then host `dist/` over HTTPS (Vercel or GitHub Pages). `npm run build` also regenerates
   `public/privacy.html` from `docs/privacy.md`.

**Data:** hand tracking is processed in the page and dropped. Progress and settings stay in this browser, and
*Delete my data* erases them. Nothing is uploaded.

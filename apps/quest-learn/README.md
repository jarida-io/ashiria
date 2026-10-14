# Ashiria Learn (WebXR)

The Meta VR Start 2026 entry: practise Kenyan Sign Language with hand tracking on Meta Quest. It is built with
Meta's [Immersive Web SDK](https://iwsdk.dev) (TypeScript, three.js, WebXR), and shipped as a packaged web app
(PWA) later. For the competition, WebXR entries are submitted as a **hosted link** (GitHub Pages or Vercel).

All recognition runs in [`@ashiria/engine`](../../sdk/web/ashiria-engine), the TypeScript port of the shared
engine, parity-tested against the Python reference. This app only reads WebXR and draws.

| File | What it does |
|---|---|
| `src/hand-sampler.ts` | WebXR hand joints and head pose → fixed 30 Hz canonical frames → attempt segmenter |
| `src/shadow-hands.ts` | The target sign's reference, animated where your hands should be; wrong fingers turn orange |
| `src/lesson.ts` | Practice (Verify with hints), Free Sign (Recognise), Studio; the lesson panel |
| `src/studio.ts` | Studio takes saved through the dev server (development builds only) |
| `src/app-state.ts` | Pack, mode, settings and progress (on this device only) |
| `public/ui/lesson.uikitml` | The lesson panel (follows your head, poke or pinch its buttons) |
| `vite.ashiria.ts` | Dev-only routes for packs and Studio; build guard that refuses non-release packs |

## Run

```bash
npm install
```
```bash
npm run dev
```

- Opens the emulator in a managed browser. The panel's **Start** button enters the emulated headset.
- **On the Quest:** open the network URL that `npx @iwsdk/cli dev status` prints, on the same Wi-Fi, and accept
  the local certificate.
- **Content:** in development the app loads the newest pack from `../../content/packs/`. Build one with
  `make -C ../../training pack`. Development packs may contain KSL Dictionary drafts: they are served to this Mac
  and your headset only, and never bundled.

## Study sheet (approve and learn signs)

With `npm run dev` running, open `https://localhost:8081/dev/study/` on the Mac. For each sign:
- the local KSL Dictionary video plays next to our reference hands;
- **Approve**, **Reject** and **Back to draft** record your review in `content/catalogue/ksl-core.yaml`
  (`ashiria approve`);
- the release pack ships only approved signs.

**Sign browser:** `https://localhost:8081/dev/study/browse` shows every extracted KSL Dictionary sign (2,548), with
MediaPipe's hand and body landmarks drawn over the video. It has search, category filter, slow motion and joint
numbers.

Both pages are development only: the videos never leave this Mac and are never built into the app.

## Studio (recording reference signs)

Open the app with `?studio` on a development build (`https://<mac>:8081/?studio`). Pick a sign with Prev/Next,
then sign it. Every attempt the segmenter cuts is saved as a take in `training/data/studio/<date>/`. Hold static
signs for about a second. Then:

```bash
make -C ../../training studio-import STUDIO=data/studio/<date>
```
```bash
make -C ../../training pack
```

## Release

1. Build a release pack: `../../training/.venv/bin/ashiria pack --release`. It ships only distributable signs
   (dictionary-derived poses, which need the permission recorded in `content/rights.yaml`, plus Studio and team takes)
   that you have approved (`review.status: approved` in `content/catalogue/ksl-core.yaml`).
2. Copy it into `public/packs/`. The build refuses any pack that isn't a release pack, and `public/packs/` is
   git-ignored.
3. Run `npm run build` and host `dist/` on HTTPS (Vercel recommended; see `iwsdk-hosting` in `.claude/skills/`).
   The link is what judges open. Packaging as a PWA for the Horizon Store (`iwsdk-pwa-packaging`) comes after
   the competition.

**Data:** hand tracking is processed in the page and dropped. Progress and settings stay in this browser's
storage, and "Delete my data" erases them. Nothing is uploaded.

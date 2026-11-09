# Development

## Setup

| Part | Needs | Set up with |
|---|---|---|
| The app | Node.js 22 (`apps/quest-learn/.nvmrc`) | `cd apps/quest-learn && npm install` |
| Training tools | Python 3.12, `make` | `make -C training setup` (add `setup-train` for PyTorch and ONNX, `models` for MediaPipe's models) |
| C# engine (optional) | .NET 10 | `dotnet test sdk/unity/com.jarida.ashiria.engine/Tests~/dotnet` |

## Sign data (local only)

The app needs a content pack, built from sign data that is never committed ([data and rights](DATA_AND_RIGHTS.md)).
With the KSL Dictionary's videos on your machine (`training/data/raw/ksl_dictionary/`, or `paths.raw_videos` in
`training/configs/ksl_dictionary.yaml`):

```bash
cd training
make dictionary                                   # import labels → audit → MediaPipe landmarks → references (configs/vocabulary_v2.txt)
.venv/bin/ashiria audit-dictionary                # every video: downloads, tracking, duplicates (a report)
make pack                                         # calibrate, then write content/packs/ksl-core-<version>/
```

## Running the app

```bash
cd apps/quest-learn
npm run dev            # dev server at https://localhost:8081 and the headset emulator in a managed browser
npm run dev:runtime    # the dev server alone
```

- **Emulator:** **Start** enters the emulated headset. Emulated hands are posed, not tracked, so never tune
  tolerances with them.
- **Quest:** open the dev server's network address in the Quest Browser on the same Wi-Fi, and accept the local
  certificate.
- **Webcam:** in development, the start page's *Start with my webcam* drives the emulator's hands from your webcam
  and reads your signs with the recogniser.
- **Studio:** add `?studio` to the address (development only). Every attempt is saved as a take on this machine;
  `make -C training studio-import STUDIO=data/studio/<date>` turns them into references.

Development pages, on this machine only:

| Address | For |
|---|---|
| `/dev/study/` | Watching each course sign next to its reference hands; Approve, Reject, Back to draft |
| `/dev/study/browse` | Every dictionary sign: landmarks over the video, reference, avatar, steps, recogniser results |
| `/dev/study/results` | The model check: clips read by the web app's own pipeline, with frames and totals |

## Tests

```bash
make -C training test                                      # Python: engine, content, packs, steps
node --test sdk/web/ashiria-engine/test/                   # TypeScript engine, against spec/golden
cd apps/quest-learn && npm run typecheck && node --test src/*.test.ts
```

After changing the Python engine, regenerate the golden vectors (`make -C training golden`) and keep the
TypeScript and C# tests passing.

## The `ashiria` command

Run from `training/` as `.venv/bin/ashiria <step>` (or `make <step>` for the common ones).

| Group | Steps |
|---|---|
| Dictionary data | `import-dictionary`, `audit-dictionary`, `extract`, `check`, `references` |
| Course content | `catalogue`, `approve`, `calibrate`, `pack` (`--release` for hosting), `import-studio`, `golden` |
| Recogniser | `embed`, `embed-evaluate`, `embed-signs`, `embed-views`, `sign-groups`, `lookalike-rounds`, `sentences-eval`, `export-recogniser` |
| Step captions | `steps-check`, `steps-fit` |
| Dev pages (JSON) | `study-index`, `study-data`, `study-pose`, `study-reference`, `study-landmarks`, `study-avatar` |

`ashiria <step> --help` describes each one. [Recognition](RECOGNITION.md#reproducing) shows a training session
from start to export.

**Long runs.** Training takes about 1–1.5 hours for 30 epochs on an Apple-silicon Mac.

- Start long jobs at normal priority. zsh runs background jobs at lower priority, so use `setopt NO_BG_NICE`.
- Run them from a frozen copy of the code: copy `training/src/ashiria_train` to `training/.snap/` and set
  `PYTHONPATH=training/.snap`. The data loader re-imports the code every epoch, so an edit mid-run would crash it.

## Releasing

1. Approve what should ship, then `ashiria pack --release`. It ships only approved signs and phrases with recorded
   permission.
2. Copy the pack into `apps/quest-learn/public/packs/`. The build refuses any pack that isn't a release pack.
3. `npm run build`, then host `dist/` over HTTPS (Vercel or GitHub Pages). The privacy policy is published with it.

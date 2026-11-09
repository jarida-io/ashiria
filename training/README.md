# Ashiria training tools

The Python side of Ashiria, run as `ashiria <step>` from this folder. It does three jobs:

1. **Course content.** KSL Dictionary videos (or Studio recordings) → reference signs → a calibrated content pack
   the app loads.
2. **The recogniser.** Every dictionary sign → an embedding model (ONNX) that reads any of 2,646 signs, tested on
   simulated attempts.
3. **The engine's reference.** The TypeScript and C# engines must reproduce this one on the golden vectors
   (`ashiria golden`).

The models never see pixels. Videos become MediaPipe hand and body landmarks, then the engine's canonical frames,
the same frames the headset makes from hand tracking ([features](../spec/FEATURE_SPEC.md),
[architecture](../docs/ARCHITECTURE.md)).

## Setup

```bash
make setup         # Python 3.12 virtualenv in .venv: data pipeline, content and packs
make setup-train   # adds PyTorch, ONNX and scikit-learn, for the recogniser
make models        # MediaPipe's hand and pose models into models/mediapipe/
make test
```

## Course content

| Command | What it does |
|---|---|
| `make dictionary` | Imports the dictionary's labels, audits the videos, extracts landmarks and cuts a reference for each sign in `configs/vocabulary_v2.txt` (into `exports/dictionary/references/`) |
| `ashiria check` | Checks the extracted landmarks: hand slots and the hand and body axes agree |
| `ashiria audit-dictionary` | Every dictionary video: downloaded, tracked well, duplicated (a report) |
| `make catalogue` | Adds new vocabulary signs to `content/catalogue/ksl-core.yaml` as drafts; never changes existing entries |
| `ashiria approve <ids>` | Records a review (`--status approved`, `rejected` or `draft`), as the study sheet's buttons do |
| `make calibrate` | Simulated attempts through the real segmenter → each sign's threshold and tier, and Free Sign's thresholds |
| `make pack` | Calibrates, then writes `content/packs/ksl-core-<version>/` ([format](../spec/CONTENT_PACK.md)). `ashiria pack --release` ships only approved signs and phrases, and dictionary-derived signs only with recorded permission |
| `make studio-import STUDIO=data/studio/<date>` | Studio takes from the headset → `exports/studio/store.json`; they replace those signs' dictionary references in the next pack |
| `ashiria studio-fake` | Fake Studio takes from the dictionary references, to test the import without a headset |
| `make golden` | Regenerates `spec/golden/` for the TypeScript and C# tests |

Calibration uses simulated attempts from one signer's videos, so its numbers are optimistic. Studio already saves
test attempts (`role: test`); calibrating from them is not built yet.

## The recogniser

| Command | What it does |
|---|---|
| `ashiria embed` | Trains the embedding model on simulated attempts of every dictionary sign, then tests it (`runs/embed-<time>/`) |
| `ashiria embed-evaluate`, `embed-signs` | Re-tests a run; per-sign results by kind, for the sign browser |
| `ashiria embed-views` | Accuracy from a grid of camera positions, and with a mirrored camera |
| `ashiria sign-groups` | Every sign by kind (fingerspelling, number, word, phrase) and into look-alike groups, with what tells each pair apart |
| `ashiria lookalike-rounds` | Fine-tunes a run in rounds on the signs it mixes up; stops when a round doesn't help |
| `ashiria sentences-eval` | Reading signs in a row: random runs, the phrase list, the dictionary's recorded phrases |
| `ashiria export-recogniser --run runs/<run>` | ONNX model and reference embeddings → `exports/recogniser/`, which the dev server gives the web reader |

Results, limits and a full training session are in [Recognition](../docs/RECOGNITION.md).

An older classifier pipeline (`audit`, `extract`, `build`, `baseline`, `train`, `evaluate`, `export`, with
`configs/default.yaml`) is still here for a future dataset with many signers per sign. One video per sign can't
train a classifier, which is why the recogniser learns embeddings instead.

## Step captions

| Command | What it does |
|---|---|
| `ashiria steps-check` | How often captions made from the measured motion agree with the dictionary's descriptions |
| `ashiria steps-fit` | Fits motion-based calls (handshape, palm, place, movement) to the descriptions, tested on held-out clips |

Course signs use the written `steps:` in the catalogue ([content](../docs/CONTENT.md#step-captions)).

## Long runs

- Training takes about 1–1.5 hours for 30 epochs on an Apple-silicon Mac.
- zsh starts background jobs at lower priority, which starves them on a busy machine. Run `setopt NO_BG_NICE`
  first.
- Run from a frozen copy of the code (`cp -R src/ashiria_train .snap/` and `PYTHONPATH=.snap`). The data loader
  re-imports the code, so editing it mid-run crashes the run.

## Data

- The dictionary's videos live in `data/raw/ksl_dictionary/`, or wherever `paths.raw_videos` in
  `configs/ksl_dictionary.yaml` points.
- Everything under `data/`, `exports/` and `runs/` stays on this machine: it is git-ignored and never goes to a
  cloud service. Landmark files are small, so the videos can move to an external drive after extraction.
- What may ship, and why: [data and rights](../docs/DATA_AND_RIGHTS.md).

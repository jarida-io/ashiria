# Ashiria training pipeline

Two jobs:
1. **Content for the headset.** Reference signs (from the KSL Dictionary drafts or Studio recordings) → a
   calibrated content pack. This is what the competition build uses.
2. **A sign model** (later). Sign videos → a small offline model for Quest.

It is also the **reference implementation of the engine**. The C# port in `sdk/unity/` must reproduce it on
the golden vectors (`ashiria golden`).

```
videos ─audit─▶ manifest ─extract─▶ MediaPipe hand + body landmarks ─build─▶ canonical segments
       ─train─▶ SignNet (1D conv + Transformer, ~2M params) ─export─▶ ONNX + metadata ─▶ Unity Sentis
```

The model never sees video pixels. On the headset there is no camera image of the user's hands, only tracked joints. So training turns each video into hand-joint sequences, and the headset produces the same sequences from Quest hand tracking. The rules for that are in [`spec/FEATURE_SPEC.md`](../spec/FEATURE_SPEC.md), which every app shares.

## Setup (once)

```bash
make setup         # Python 3.12 virtualenv in .venv: enough for audit, extract, check and references
make setup-train   # adds PyTorch, ONNX and scikit-learn for baseline, train, evaluate and export
make models        # MediaPipe hand and pose models into models/mediapipe/
make test
```

After extracting, run `ashiria check` before building anything. It confirms that the hand slots and the hand/body coordinate axes agree on your footage.

## Steps

| Step | Command | Writes |
|---|---|---|
| 1. Audit | `make audit` | `data/manifests/videos.csv`, `data/manifests/audit_report.md` |
| 2. Extract | `make extract` | `data/landmarks/<clip_id>.npz`, `data/manifests/extraction.csv` |
| 3. Build | `make build` | `data/cache/segments.npz`, `index.csv`, `labels.json` |
| 4. Baseline | `make baseline` | nearest-neighbour accuracy, printed |
| 5. Train | `make train NAME=v1` | `runs/v1/best.pt`, `metrics.csv` |
| 6. Evaluate | `make evaluate RUN=runs/v1` | `runs/v1/report.md`, `eval.json`, `confusion.csv` |
| 7. Export | `make export RUN=runs/v1` | `exports/v1/model.onnx`, `model_meta.json`, `labels.json`, `golden.json` |

**Before you trust a model:**
- **Check the audit report first.** It shows how labels and signers were read from your file names. Fix `labels` in `configs/default.yaml` until they're right.
- **Test the extraction on a few clips** with `.venv/bin/ashiria extract --limit 20` before running all of them.
- **Expect lower accuracy on the headset.** The baseline and test accuracy are measured on videos. The real check is a small set of Quest recordings run through the same steps.

## KSL Dictionary videos (old work)

`data/raw/ksl_dictionary/` holds the videos scraped from ksldictionary.com in May 2025. They were moved here from `~/Documents/KSL_Dictionary_Extractor/ksl_videos`, which is now a link to this folder.

What the folder contains:
- **2,694 unique videos.** The site lists every sign twice, so there are 8,894 files.
- **One video per sign, from one source.**
- **Per-sign metadata:** English glosses (aliases), category, and handshape and movement descriptions.

That makes it a **reference library, not a training set**. A classifier can't learn a sign from a single example signed by a single person.

```bash
make dictionary      # import labels → audit → extract landmarks → references for configs/vocabulary_v1.txt
```

Then run `ashiria check` (are the hand slots and coordinate axes consistent?).

This writes these files to `exports/dictionary/references/`:
- **`references.csv`:** tags for every sign: one or two hands, static or moving, near the face, hands close
  together, tracking quality. They help pick a vocabulary the headset can check.
- **`references.json`:** for each vocabulary sign, the reference take (cut exactly like a live attempt) and its key
  frame, plus the dictionary's description (local only, never packed).
- **`lookalikes.csv`, `distances.npy`:** each vocabulary sign's nearest neighbours, i.e. the likely confusions.
- **`distractors.npz`:** 300 non-vocabulary signs, so calibration can check that Free Sign says "not sure" to them.

**Rights:** the videos are the KSL Dictionary's copyright. Jerry holds the owners' permission (recorded in `content/rights.yaml`): derived hand poses may ship in packs. Video frames and the dictionary's text never do, and the videos themselves stay local.

## Content for the headset

```
catalogue (content/catalogue/ksl-core.yaml, committed)  +  references (dictionary drafts or Studio takes)
   ─calibrate─▶ tau per sign, Free Sign thresholds, tiers   ─pack─▶ content/packs/ksl-core-<version>/
```

| Command | What it does |
|---|---|
| `make catalogue` | Creates or extends `content/catalogue/ksl-core.yaml` from `configs/vocabulary_v1.txt`. Re-running only adds new words; your edits (tiers, hints, approvals) are kept |
| `make calibrate` | Simulated attempts through the real segmenter → tau, Free Sign thresholds, tiers (`configs/tiers.yaml`). Writes `exports/calibration/ksl-core.json` and `tiers_ksl-core.csv` for review |
| `make pack` | Runs calibration, then writes the pack (`spec/CONTENT_PACK.md`). Add `--release` (`.venv/bin/ashiria pack --release`) for a distributable pack: it ships dictionary-derived signs (permission in `content/rights.yaml`) and Studio takes, and stops if a shipped sign isn't approved |
| `make studio-import STUDIO=data/studio/<session>` | Studio takes pulled from the headset → `exports/studio/store.json`. Studio takes then replace those signs' dictionary drafts in the next pack |
| `.venv/bin/ashiria studio-fake` | Fake Studio takes from the dictionary references, to test the import before the headset arrives |
| `make golden` | Regenerates `spec/golden/` (engine and runtime vectors, mini pack, Studio take) for the C# tests |

Calibration is simulated (one signer, synthetic variation), so its numbers are optimistic. Studio already saves
the team's test attempts (`role: test`, in `exports/studio/attempts.json`). Calibrating from them instead is the
next step, and isn't built yet.

## Where the videos go

- Copy them into `data/raw/videos/`, in any sub-folders.
- Or point `paths.raw_videos` in the config at wherever they are, such as an external drive.

Landmark files are small (roughly 100 KB per clip), so the raw videos don't have to stay on this Mac after extraction.

## Data rules

- These are videos of people, and that makes them personal data under Kenya's Data Protection Act. Only use videos whose signers agreed to this use.
- Keep them, and everything under `data/`, `runs/` and `exports/`, out of git and out of cloud services. The `.gitignore` already excludes them.
- The app ships only the exported model. It never sends hand data anywhere.

## Still to add once we've seen the data

- A "none" class (rest poses and movement between signs), so the app can reject non-signs.
- Quest validation: record a small test set on the headset and report accuracy on it next to the video test set.
- Fine-tuning on Quest recordings if the gap between the two is large.

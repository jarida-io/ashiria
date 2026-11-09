# Recognition

Ashiria reads signs in two ways:

| | Practice and Free Sign (the Quest app) | The recogniser (web preview, future Interpret) |
|---|---|---|
| Question | "Is this sign X?" and "Which of the signs I learned is this?" | "Which of all 2,646 dictionary signs is this?" |
| How | Compare the attempt with each sign's reference takes (DTW), then check fingers, palm, place and movement | An embedding model: the attempt's embedding against every sign's, by cosine similarity |
| Gives | Pass or fail with per-finger hints; an answer or "not sure" | The top guesses with similarities; an answer or "not sure" |
| Tuned by | Calibration: simulated attempts set each sign's threshold and tier (scored, lenient, watch-only) | Training on simulated attempts, then accept thresholds chosen on held-out attempts |

> **All results below come from one signer's videos** (the KSL Dictionary) and simulated variation of them. They show
> how robust the reading is to camera angle, jitter and timing, not how well it reads other people. Signs recorded
> by other signers are the real test, and single-signer models often lose a lot there.

## The recogniser

- **Input:** a sign cut from hands up to hands down, resampled to 64 frames of the 197 engine features
  ([architecture](ARCHITECTURE.md)).
- **Model:** convolution stages and a transformer (about 1.9 million parameters) → a 128-number embedding.
- **Training:** contrastive. Each sign is shown as two different simulated attempts that should land close
  together, and away from every other sign in the batch.
- **References:** every usable KSL Dictionary video, 2,646 signs: words, 422 recorded phrases, numbers and the
  alphabet.
- **Export:** ONNX (7.5 MB) plus the references' embeddings, run in the browser with ONNX Runtime Web. Accepted when
  the best match has similarity ≥ 0.72 and leads the next by ≥ 0.01. On held-out simulated attempts it then
  answers 99% of them, and is right 99.9% of the time when it answers. It still accepts a sign it doesn't know
  as a known one 15% of the time.

**Simulated attempts** turn one recording into "someone else signing it": the body turned a little, hands rotated
about the wrist, a different body size, tracking jitter, faster or slower, cut a little early or late, frames or a
whole hand lost for a moment, fingers bent slightly differently, and a camera off to one side, above or below
(which misjudges depth along its line of sight and the hand's tilt).

## Results

Recognised first among all 2,646 signs, from 7,938 simulated attempts per test (3 per sign):

| Model (16 Oct 2026) | Normal | Harder variation | Camera off to the side | Average |
|---|---|---|---|---|
| Trained with simulated cameras | 98.9% | 98.4% | 98.2% | 98.51% |
| + look-alikes in half of each batch, from scratch | 98.8% | 97.9% | 98.0% | 98.22% |
| + signs cut out of sentences | 99.4% | 98.6% | 99.0% | 99.00% |
| + three look-alike rounds (**exported**) | **99.5%** | **99.1%** | **99.3%** | **99.32%** |

By kind, for the exported model: fingerspelling 100%, numbers 97.5–100%, words 99.1–99.6%, phrases 99.3–99.4%.

**Look-alikes.** `ashiria sign-groups` sorts every sign by kind and into groups the model finds close or mixes
up, and works out what tells each pair apart. Mixed-up pairs mostly differ in movement or location, not handshape.
Putting look-alikes in half of every batch from scratch fixed most old mix-ups but caused as many new ones.
Gentle rounds worked: fine-tune the best model with look-alikes in a fifth of each batch, collect the mix-ups of
every round, and stop when a round doesn't help. All three rounds helped (99.00 → 99.07 → 99.21 → 99.32%), so more
may help further.

**Camera angle.** From 21 fixed camera positions, up to 60° to either side and 30° above or below (wider than
training's 35° and 25°), the camera-trained model read 97.6–98.6% of signs first.

**Mirrored cameras and left-handed signers.** Mirrored attempts read 15.5% right as they are. Reading each attempt
as it is and mirrored, and keeping the better match, gives 98.5% in simulation, with unmirrored attempts unchanged.
The web reader does this. On real mirrored video (below) it reads 91%.

**Through the browser, on the dictionary's own videos** (`/dev/study/results`: MediaPipe in the browser, the web
tracker, the attempt cutter and the exported model; these are training videos, so this checks the pipeline, not
accuracy on new signing):

| Clips | Read right first | Right sign in the top 5 |
|---|---|---|
| Course signs (75) | 99% (AGAIN came second) | 100% |
| Fingerspelling (24), numbers (27) | 100% | 100% |
| Recorded phrases (40 at random) | 100% | 100% |
| Other words (40 at random) | 98% (THROAT came second) | 100% |
| Course signs, video mirrored | 91% | 95% |

## Signs in a row (experimental)

`training/src/ashiria_train/sentences.py` reads several signs signed in a row:

1. It scans the signing with windows of several lengths.
2. Each window offers its closest signs.
3. It picks the best run of signs, charging a cost for each sign read so short windows don't each become a sign.
4. Signs that follow each other in a known phrase get a bonus.

It is tested on sentences joined from single recordings, with short transitions between them:

| Exported model, with the phrase bonus | Signs found | Exactly right |
|---|---|---|
| 60 random runs of 2–4 signs | 75–79% | 50–53% |
| The 46 phrases ([content](CONTENT.md)) | 81% | 54% |

The model trained with signs cut out of sentences read the phrases better: 85% of signs found, and 61% of
phrases exactly right. The look-alike rounds didn't keep the sentence views, so the exported model gave some of
that back. Real running signing is still untested: the dictionary's own phrase videos mostly don't follow our
glosses (signed differently, or as one movement), so they can't test them. Recordings of the phrases signed as
written are the next step.

## Reproducing

```bash
cd training
.venv/bin/ashiria embed --epochs 30 --lookalike-share 0.5 --context-share 0.35   # train; tests itself at the end
.venv/bin/ashiria sign-groups                                                  # kinds and look-alikes
.venv/bin/ashiria lookalike-rounds --start runs/<run> --rounds 3               # fine-tune on look-alikes
.venv/bin/ashiria embed-views                                                  # camera positions, mirrored
.venv/bin/ashiria sentences-eval                                               # signs in a row
.venv/bin/ashiria export-recogniser --run runs/<run>                           # → exports/recogniser
```

Each run writes `report.json` (vocabulary tests against DTW), `signs.json` (every sign, three tests, by kind),
and, from the commands above, `perspectives.json` and `sentences.json` into its folder under `training/runs/`.
The data these need stays local ([data and rights](DATA_AND_RIGHTS.md)).

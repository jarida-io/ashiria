# Data and rights

## The rule

Sign videos show real people, which makes them personal data under Kenya's Data Protection Act. So:

- **Videos stay on the developer's machine.** No video, video frame or image of a signer is in this repository,
  in the app or in any cloud service.
- **Everything derived from the videos stays local too:** landmarks, references, packs, trained models and run
  results (`training/data/`, `training/exports/`, `training/runs/`, `content/packs/` are git-ignored).
- **The app ships hand poses only:** joint positions over time, never pictures.
- **Users' hand data never leaves their device.** Practice runs on the headset; progress and settings stay in the
  browser there ([privacy policy](privacy.md)).

The test vectors in [`spec/golden/`](../spec/golden) come from synthetic hands: they contain no real person.

## The KSL Dictionary

The course's reference signs, the avatar coach's motion and the recogniser's references are derived from the videos
of the [KSL Dictionary](https://ksldictionary.com), which belong to their owners. Jerry Ochieng (Jarida) holds the owners' permission,
recorded in [`content/rights.yaml`](../content/rights.yaml), to develop with them. `ashiria pack --release` refuses to
ship dictionary-derived signs unless that permission is recorded.

What ships, and what doesn't:

| Ships in the app | Never ships |
|---|---|
| Hand poses derived from the videos | The videos, frames or images of the signer |
| The coach's body and hand motion (a figure, not the signer) | The dictionary's text (descriptions, notes) |
| Step captions in our own words | |

The development tools that show the videos (the study sheet, the sign browser and the model check) serve them only to
the developer's own machine.

## People who record signs

Studio takes (signs recorded in the headset) by anyone other than the developer need that person's signed consent,
on the consent form named in `content/rights.yaml`. Takes carry pseudonymous signer ids, never names.

## Phrase sources

The phrase list's word order comes from published Kenyan teaching material ([content](CONTENT.md#phrases)). The
glosses are short facts about word order, written by us, not copied text.

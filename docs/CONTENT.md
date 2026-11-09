# Content

The course lives in [`content/catalogue/ksl-core.yaml`](../content/catalogue/ksl-core.yaml), edited by the team.
Packs are built from it (`ashiria pack`), so the catalogue is the one place to change what Ashiria teaches.

## The course

| Unit | Signs | Phrases |
|---|---|---|
| 1. Spell your name | The 26 letters of the one-handed manual alphabet (J and Z move) | |
| 2. Numbers | 0–10. In KSL five is a closed fist; six to nine are the left fist ("five") with the right hand's number at its knuckles | |
| 3. Everyday signs | hello, thank you, sorry, fine, again, deaf, eat, water, mother, father, baby, happy | |
| 4. Meeting people | good, morning, you, how, me, name, your, what, mine, live, where, sign, learn, now | 8 |
| 5. Asking again | please, slow, understand, not, know | 4 |
| 6. Needs | hungry, want, sick, doctor, call, toilet, tired | 6 |

**Review.** The 49 signs of units 1–3 were approved on 16 Oct 2026. The 26 signs and 18 phrases of units 4–6 are
drafts. A release pack ships only approved signs and phrases.

**Tiers.** Calibration (simulated attempts) sets how strictly each sign is checked: *scored* (full per-finger
check), *lenient* (wider tolerances, look-alike guard kept), or *watch-only* (shown, never scored). Signs that
tracking can't check well, such as fist letters, crossed fingers or hands close together, end up lenient. A sign's
`tier` in the catalogue overrides calibration.

## A sign's entry

Abridged (entries also hold aliases, tier notes and switches):

```yaml
- id: sorry
  gloss: sorry
  unit: everyday-signs
  order: 3
  tier: auto                      # auto | scored | lenient | watch
  hint: ''                        # the team's own words, shown after two misses
  steps:                          # the avatar coach's captions: start, each movement, end
  - Make a fist with your right hand and place it on the middle of your chest, palm facing you.
  - Rub it round in small circles, about 4 cm across, three times.
  - Hold it still for a moment, then lower your hand.
  sources:
    dictionary: sorry             # the KSL Dictionary sign its reference comes from
  review: {status: approved, by: Jerry, date: '2026-10-16'}
```

## Step captions

Every course sign has written steps (`steps:`), in the team's own words, never the dictionary's text. The coach
lays them over the sign's motion: the first while the hands rise into place, the last as they settle, and the ones
between over the sign's movements.

They are written from facts in each sign's dictionary description (handshape, palm, place, hands, kinds of
movement) and from the measured motion (size, direction, count). Measured from video alone, handshape, palm, place
and movement kind never reach 70% agreement with the descriptions (`ashiria steps-fit`, tested on held-out clips).
So signs without written steps get captions from their description's facts plus what the motion shows reliably:
timing, size, direction and rhythm.

## Phrases

[`content/phrases/ksl-core.yaml`](../content/phrases/ksl-core.yaml) lists 46 everyday phrases as glosses: the
signs in the order they are signed. The course teaches 18 of them, part by part. Each part is checked like a sign,
a miss stays on that part, and the phrase is matched, recalled and mastered as a whole.

**Word order.** The glosses follow patterns taught in Kenyan KSL courses:

- time signs first (TOMORROW SCHOOL ME GO);
- the verb after its object;
- question words, NOTHING/NOT, MUST, CAN and FINISH at the end (NAME YOUR WHAT?; MEAT HE EAT FINISH);
- possessives after the noun (NAME MINE …);
- repetition for plurals and emphasis.

Sources: Jaramogi Oginga Odinga University of Science and Technology KSL course examinations; Kilanya (2016),
Kenyatta University; the Maseno AI4KSL paper ([arXiv 2410.18295](https://arxiv.org/abs/2410.18295)); Morgan (2020) on
KSL transitive events.

**Attested or patterned.** A phrase is *attested* when its gloss appears in that material, and *patterned* when we
built it from the patterns. Most are patterned. No linguistic study of KSL yet confirms the question and negation
patterns, and the dictionary's own recordings of some phrases sign them differently. **A Deaf KSL signer should check
every phrase before it ships**, and a release pack ships only approved ones. Phrases marked `question` are signed
with a question face, which hand tracking can't see.

## Adding signs and phrases

1. Add the words to a vocabulary file (`training/configs/vocabulary_v2.txt`, one sign per line under
   `# Unit N: title`).
2. `ashiria references --config configs/ksl_dictionary.yaml --vocab configs/vocabulary_v2.txt` picks each word's
   dictionary clip and cuts its reference.
3. `ashiria catalogue --vocab configs/vocabulary_v2.txt` adds the new signs as drafts. It never changes existing
   entries.
4. Write each sign's `steps:` and `hint:`. For a phrase, add an entry under `phrases:` with its `parts` (sign ids
   in signing order), `source` and `question`.
5. Review in the study sheet (`/dev/study/`), approve with `ashiria approve <ids>`, then `ashiria pack`.

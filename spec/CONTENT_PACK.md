# Content pack format, v1

What the headset loads to teach and check signs. `ashiria pack` builds one from the catalogue
(`content/catalogue/<pack>.yaml`) and the reference stores. The output goes to
`content/packs/<pack_id>-<version>/`, which is git-ignored. `ContentPack.Parse` (C#) loads it.

## Rules

- **Hand poses only.** No video, images or thumbnails, and none of the KSL Dictionary's text (descriptions,
  notes). The builder refuses forbidden keys (`video`, `image`, `clip_id`, `description`, `note`, `name`,
  `email`, …). Demonstration videos (Entrants only) are separate app assets, keyed by sign id.
- **Pseudonymous signer ids only** (`S01`). Never names.
- **Studio replaces dictionary.** A sign's reference comes either from kept Studio takes or from the
  dictionary draft, never a mix.
- **`distribution: internal`** packs may contain dictionary drafts. They are for the development headset only.
- **`ashiria pack --release`** ships only distributable signs:
  - signs with no reference, and dictionary drafts without written permission in `content/rights.yaml`, are
    left out automatically (and listed);
  - every shipped sign must have `review.status: approved` in the catalogue, or the build stops.

  For the competition Jerry holds the dictionary owners' permission (`content/rights.yaml`), so release packs may
  contain dictionary-derived hand poses next to Studio and team-video takes.
- **Deterministic.** Same inputs, same `content_hash` (simulation seed fixed, keys sorted, values rounded).

## Files

### manifest.json

```json
{
  "format": "ashiria.content-pack", "format_version": 1,
  "pack_id": "ksl-core", "version": "0.1.0", "spec_version": 1, "runtime_version": 1,
  "distribution": "internal",
  "build": {"created": "2026-10-14T12:42:00Z", "git": "544c44a", "dirty": false},
  "rights": {"sources": ["ksl_dictionary"],
             "ksl_dictionary": {"permission": "none", "reference": ""},
             "studio": {"consent_form": ""}},
  "engine": { "...": "EngineParams.to_json(): every runtime parameter, with the calibrated Free Sign thresholds" },
  "calibration": {"method": "simulated", "attempts_per_sign": 16, "rates": {"correct": 0.98, "...": "..."}},
  "files": {"catalogue": {"path": "catalogue.json", "sha256": "..."},
            "references": {"path": "references.json", "sha256": "..."}},
  "content_hash": "sha256 of catalogue.json + references.json bytes",
  "selfcheck": {"sign_id": "letter-a", "take": 0, "descriptor_sum": 185.5013, "tolerance": 0.01}
}
```

**The loader refuses** a pack when:
- the format version, spec version or runtime version differs from its own;
- a file hash does not match;
- the self-check fails. The self-check recomputes the descriptor sum of one take from the loaded data. If
  it fails, the engine and the pack builder disagree.

### catalogue.json

```json
{"pack_id": "ksl-core",
 "units": [{"id": "spell-your-name", "title": "Spell your name", "order": 1}],
 "signs": [{
   "id": "letter-a", "gloss": "A", "aliases": [], "unit": "spell-your-name", "order": 1,
   "tier": "lenient", "tier_source": "auto", "tier_reasons": ["hard handshape for hand tracking"],
   "recognise": true, "hint": "", "facial_grammar": false,
   "tags": {"hands": "one", "movement": "static", "near_face": false, "hands_close": false},
   "lookalikes": ["letter-s", "letter-e", "letter-t"],
   "source": {"kind": "ksl_dictionary", "takes": 1, "signers": ["ksl-dictionary"]},
   "review": {"status": "draft", "by": null, "date": null}}]}
```

### references.json

```json
{"seq_len": 32,
 "signs": {"letter-a": {
   "takes": [{"source": "studio", "take_id": "…", "signer": "S01", "fps": 24.8, "frames": 32, "key_frame": 17,
              "hand_rel": ["frames×2×21×3 flat, metres, 5 dp"],
              "wrist_pos": ["frames×2×3 flat, shoulder widths, 4 dp"],
              "present": ["frames×2 flat, 0/1"]}],
   "shadow_take": 0,
   "tau": 1.48,
   "tol": {"finger": [0.61, 0.44, 0.44, 0.44, 0.44], "finger_mean": false, "spread": 0.56,
           "orientation_deg": 50, "location": 0.5, "movement": 0, "other_hand": 0,
           "distance_z": 1.6, "lookalike_guard": true}}}}
```

**Computed on load, not stored:** each take's features, descriptor, key pose and path length.
`SignRef.Bind` does this in C#, and `signs.Take.bind` in Python, so the files carry only recordings.

**`shadow_take`** is the take the app animates as the shadow hand (the medoid of the Studio takes).

**Tolerances** are final per-sign values:
- the tier default (`params.VerifyParams`);
- widened for the thumb (×1.4) and for dictionary-sourced signs (×1.25, fingers and spread);
- widened for fingers that vary across Studio takes (2.5 × their spread, capped at 2×).

The headset only multiplies them by the user's strictness.

## Size and loading

The 72-sign dictionary pack is about 2.5 MB. Parse it off the main thread after the first frame is
shown: `ContentPack.Parse` → `SelfCheck` → ready.

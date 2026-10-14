"""Catalogue, tiers, calibration and content packs, on synthetic signs only."""

import hashlib
import json

import numpy as np
import pytest

from ashiria_train.augment import rotation
from ashiria_train.canonical import steadiest_frame
from ashiria_train.catalogue import load_catalogue, parse_vocabulary, update_catalogue
from ashiria_train.config import Config
from ashiria_train.matching import segment_descriptor
from ashiria_train.pack import FORBIDDEN_KEYS, forbidden_keys, release_problems, take_segment, write_pack
from ashiria_train.params import EngineParams
from ashiria_train.pipeline import cut_reference
from ashiria_train.references import tag
from ashiria_train.signs import Take
from ashiria_train.store import SignSource
from ashiria_train.synthetic import sign_stream
from ashiria_train.tiers import assign_tier, load_rules

P = EngineParams()
AUG = {"rotate_deg": 4, "hand_rotate_deg": 5, "wrist_scale": 0.05, "wrist_shift": 0.04, "joint_noise_m": 0.001,
       "finger_noise_rad": 0.05, "speed": [0.9, 1.1], "edge_crop": 0.0, "frame_drop": 0.0, "hand_drop": 0.0}


def test_vocabulary_units_and_ids(tmp_path):
    vocab = tmp_path / "v.txt"
    vocab.write_text("# Unit 1: spell your name (no I or J)\na\nb\n\n# Unit 2: numbers\ntwo\n# Unit 3: everyday\nthank you\n")
    units = parse_vocabulary(vocab)
    assert [(u["id"], u["title"]) for u in units] == [("spell-your-name", "Spell your name"), ("numbers", "Numbers"),
                                                       ("everyday", "Everyday")]
    cfg = Config({"paths": {"content": str(tmp_path / "content")}})
    file = update_catalogue(cfg, vocab, "test")
    cat = load_catalogue(file)
    assert [s["id"] for s in cat["signs"]] == ["letter-a", "letter-b", "number-two", "thank-you"]
    # Co-creator edits survive a re-run; new words are appended.
    text = file.read_text().replace("hint: ''", "hint: Thumb along the side", 1)
    file.write_text(text)
    vocab.write_text(vocab.read_text() + "hello\n")
    cat = load_catalogue(update_catalogue(cfg, vocab, "test"))
    assert cat["signs"][0]["hint"] == "Thumb along the side" and cat["signs"][-1]["id"] == "hello"


def test_tier_rules():
    rules = load_rules()
    entry = {"id": "hello", "tier": "auto"}
    tags = {"hands": "one", "movement": "moving", "near_face": False, "hands_close": False, "dominant_tracked": 1.0}
    good = {"top1": 0.95, "top3": 1.0, "verify_pass": 0.95, "lookalike_false_pass": 0.0, "segmentation_miss": 0.0}
    assert assign_tier(entry, tags, good, rules)[0] == "scored"
    assert assign_tier(entry, {**tags, "near_face": True}, good, rules)[0] == "lenient"
    assert assign_tier(entry, {**tags, "dominant_tracked": 0.5}, good, rules)[0] == "watch"
    assert assign_tier({"id": "letter-s", "tier": "auto"}, tags, good, rules)[0] == "lenient"
    assert assign_tier({**entry, "tier": "scored", "tier_note": "checked by the team"}, {**tags, "dominant_tracked": 0.5},
                       good, rules) == ("scored", "co-creator", ["checked by the team"])


def _synthetic_signs():
    shapes = {"flat": np.zeros((5, 3)), "fist": np.full((5, 3), 1.2), "point": np.full((5, 3), 1.2)}
    shapes["point"][1] = 0.0
    specs = {"flat": {}, "fist": {}, "point": {}, "wave": {"path": (0.7, 0, 0)},
             "high": {"at": (0.3, -0.3, 0.5), "orient": rotation(0, 40, 0)}}
    catalogue = {"pack_id": "test-pack", "version": "0.0.1",
                 "units": [{"id": "basics", "title": "Basics", "order": 1}], "signs": []}
    sources = {}
    for k, (sid, kw) in enumerate(specs.items(), 1):
        stream = sign_stream(shapes.get(sid, shapes["flat"]), **kw)
        seg = cut_reference(stream, P, P.segmenter.rest_height)
        take = Take(seg, steadiest_frame(seg), "studio", take_id=f"{sid}-1")
        sources[sid] = SignSource("studio", [take], tag(stream), ["S01"])
        catalogue["signs"].append({"id": sid, "gloss": sid, "unit": "basics", "order": k, "tier": "auto",
                                   "review": {"status": "approved"}})
    return catalogue, sources


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    catalogue, sources = _synthetic_signs()
    out = write_pack(catalogue, sources, [], {}, P, AUG, tmp_path_factory.mktemp("packs"), attempts=4)
    return out, catalogue, sources


def test_pack_files_hashes_and_selfcheck(pack):
    out, catalogue, _ = pack
    manifest = json.loads((out / "manifest.json").read_text())
    cat_bytes, ref_bytes = (out / "catalogue.json").read_bytes(), (out / "references.json").read_bytes()
    assert manifest["format"] == "ashiria.content-pack" and manifest["distribution"] == "internal"
    assert manifest["files"]["references"]["sha256"] == hashlib.sha256(ref_bytes).hexdigest()
    assert manifest["content_hash"] == hashlib.sha256(cat_bytes + ref_bytes).hexdigest()
    assert EngineParams.from_json(manifest["engine"]).seq_len == P.seq_len
    refs = json.loads(ref_bytes)
    check = manifest["selfcheck"]
    seg = take_segment(refs["signs"][check["sign_id"]]["takes"][check["take"]])
    assert segment_descriptor(seg, P.weights).sum() == pytest.approx(check["descriptor_sum"], abs=check["tolerance"])
    cat = json.loads(cat_bytes)
    assert [s["id"] for s in cat["signs"]] == [s["id"] for s in catalogue["signs"]]
    assert all(s["tier"] in ("scored", "lenient", "watch") and s["tier_reasons"] for s in cat["signs"])
    assert forbidden_keys(cat) == [] and forbidden_keys(refs) == []
    assert "point" in cat["signs"][1]["lookalikes"] or "flat" in cat["signs"][1]["lookalikes"]


def test_pack_is_deterministic(pack, tmp_path):
    out, catalogue, sources = pack
    again = write_pack(catalogue, sources, [], {}, P, AUG, tmp_path, attempts=4)
    a = json.loads((out / "manifest.json").read_text())
    b = json.loads((again / "manifest.json").read_text())
    assert a["content_hash"] == b["content_hash"]


def test_forbidden_fields_and_release_rules(pack):
    assert forbidden_keys({"signs": [{"takes": [{"clip_id": "x"}]}]}) == ["/signs[0]/takes[0]/clip_id"]
    assert {"video", "description", "clip_id"} <= FORBIDDEN_KEYS
    _, catalogue, sources = pack
    assert release_problems(catalogue, sources, [], {}) == []
    sources = {**sources, "flat": SignSource("ksl_dictionary", sources["flat"].takes, sources["flat"].tags)}
    problems = release_problems(catalogue, sources, ["missing-sign"], {})
    assert len(problems) == 2 and "written permission" in problems[1]
    assert len(release_problems(catalogue, sources, [], {"ksl_dictionary": {"permission": "written"}})) == 0
    draft = {**catalogue, "signs": [{**catalogue["signs"][0], "review": {"status": "draft"}}] + catalogue["signs"][1:]}
    assert "not approved" in release_problems(draft, sources, [], {"ksl_dictionary": {"permission": "written"}})[0]

"""Quest canonicalisation (quest.py) and Studio takes: round trips, import, outliers."""

import json

import numpy as np
import pytest

from ashiria_train import quest
from ashiria_train.config import Config
from ashiria_train.store import load_studio
from ashiria_train.studio import canonical_from_raw, fake_take, import_studio, load_take
from ashiria_train.synthetic import sign_stream

FLAT, FIST = np.zeros((5, 3)), np.full((5, 3), 1.2)


@pytest.mark.parametrize("left_handed", [False, True])
def test_quest_hand_round_trip(left_handed):
    rng = np.random.default_rng(0)
    rel, wrist = rng.normal(0, 0.05, (21, 3)), rng.normal(0, 1, 3)
    rel[0] = 0
    R, ok = quest.head_axes(np.array([0.3, -0.4, 0.8]))
    head = np.array([0.2, 1.5, -0.3])
    joints = quest.to_world(rel, wrist, head, R, 0.4, left_handed)
    back_rel, back_wrist = quest.hand(joints, head, R, 0.4, left_handed)
    assert ok and np.allclose(back_rel, rel) and np.allclose(back_wrist, wrist)


def test_head_axes_and_presence_rules():
    R, ok = quest.head_axes(np.array([0.0, 0.0, 1.0]))
    assert ok and np.allclose(R, np.eye(3))  # x right, y up, z forward
    assert not quest.head_axes(np.array([0.0, 1.0, 0.0]))[1]
    assert quest.usable(True, True, False, False) and not quest.usable(True, True, True, False)
    assert quest.slot_for(is_left_hand=True, left_handed=True) == 0 and quest.slot_for(False, True) == 1
    # Looking straight up keeps the previous axes.
    hands = {"right": {"joints": np.zeros((26, 3)), "tracked": True, "high_confidence": True}}
    last = quest.head_axes(np.array([1.0, 0.0, 0.0]))[0]
    *_, axes = quest.frame([0, 1.6, 0], [0.0, 1.0, 0.0], hands, False, last_axes=last)
    assert np.allclose(axes, last)


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_fake_take_canonicalises_back(tmp_path, handedness):
    seg = sign_stream(FIST, path=(0.4, 0.2, 0), other_at=(-0.3, -1.0, 0.6))
    take = fake_take(seg, "fist", np.random.default_rng(1), handedness=handedness)
    file = tmp_path / "t.json"
    file.write_text(json.dumps(take))
    back = canonical_from_raw(load_take(file))
    on = seg.present > 0.5
    assert (back.present > 0.5).tolist() == on.tolist()
    assert np.abs(back.hand_rel[on] - seg.hand_rel[on]).max() < 1e-5
    assert np.abs(back.wrist_pos[on] - seg.wrist_pos[on]).max() < 1e-5


def test_import_studio(tmp_path):
    content = tmp_path / "content" / "catalogue"
    content.mkdir(parents=True)
    content.joinpath("ksl-core.yaml").write_text(
        "pack_id: ksl-core\nversion: 0.1.0\nunits: [{id: u, title: U, order: 1}]\nsigns:\n"
        + "".join(f"- {{id: {s}, gloss: {s}, unit: u, order: {k}, tier: auto, review: {{status: draft}}}}\n"
                  for k, s in enumerate(["flat", "fist"], 1)))
    cfg = Config({"paths": {"content": str(tmp_path / "content"), "exports": str(tmp_path / "exports")}})
    rng = np.random.default_rng(2)
    takes = tmp_path / "takes"
    takes.mkdir()

    def write(name, seg, sign, **kw):
        takes.joinpath(name).write_text(json.dumps(fake_take(seg, sign, rng, rest_height=-1.5, **kw)))

    for k in range(3):
        write(f"flat{k}.json", sign_stream(FLAT, at=(0.4, -1.0 + 0.03 * k, 0.6)), "flat")
        write(f"fist{k}.json", sign_stream(FIST, path=(0.5, 0, 0)), "fist", handedness="left" if k == 2 else "right")
    write("flat_outlier.json", sign_stream(FIST, at=(-0.5, -0.2, 0.9), path=(0, 0.6, 0)), "flat")
    write("none.json", sign_stream(FLAT, at=(0.1, -0.9, 0.5)), "_none")
    write("test.json", sign_stream(FLAT), "flat", role="test")
    takes.joinpath("broken.json").write_text('{"format": "ashiria.studio-take", "format_version": 9}')
    write("unknown.json", sign_stream(FLAT), "no-such-sign")

    out = import_studio(cfg, takes)
    store = json.loads((out / "store.json").read_text())
    assert set(store["signs"]) == {"flat", "fist"}
    assert len(store["signs"]["flat"]["takes"]) == 3 and len(store["signs"]["flat"]["excluded"]) == 1
    assert len(store["signs"]["fist"]["takes"]) == 3
    assert len(json.loads((out / "attempts.json").read_text())["attempts"]) == 2
    loaded = load_studio(out / "store.json")
    assert loaded["fist"].kind == "studio" and len(loaded["fist"].takes[0].segment) == 32


def test_webxr_layout_take_matches_openxr(tmp_path):
    seg = sign_stream(FLAT, path=(0.3, 0, 0))
    take = fake_take(seg, "flat", np.random.default_rng(4))
    n = take["frames"]["n"]
    web = json.loads(json.dumps(take))
    web["joint_layout"] = "webxr25"
    for side in ("right", "left"):
        web["frames"][side]["pos"] = np.asarray(take["frames"][side]["pos"]).reshape(n, 26, 3)[:, 1:].ravel().tolist()
        web["frames"][side]["rot"] = np.asarray(take["frames"][side]["rot"]).reshape(n, 26, 4)[:, 1:].ravel().tolist()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(take))
    b.write_text(json.dumps(web))
    x, y = canonical_from_raw(load_take(a)), canonical_from_raw(load_take(b))
    assert np.allclose(x.hand_rel, y.hand_rel) and np.allclose(x.wrist_pos, y.wrist_pos)

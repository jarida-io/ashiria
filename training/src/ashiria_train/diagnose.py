"""`ashiria check`: sanity checks on extracted landmarks before anything is built on them.

1. Slot assignment: each hand's wrist should be nearer its own body-pose wrist than the other one.
2. Axis agreement: the palm normal from the hand model and the one from the pose model's coarse
   hand points (wrist, index, pinky) should point the same way. If they don't, the hand model's
   world axes differ from the pose model's, and every orientation feature would be wrong.
"""

from __future__ import annotations

import numpy as np

from . import joints as J
from .config import Config, path
from .features import normalize

POSE_HAND = {J.RIGHT: (J.POSE_RIGHT_WRIST, 20, 18), J.LEFT: (J.POSE_LEFT_WRIST, 19, 17)}  # wrist, index, pinky


def check(cfg: Config, limit: int = 400) -> dict:
    files = sorted(path(cfg, "landmarks").glob("*.npz"))[:limit]
    if not files:
        raise SystemExit("No landmark files yet. Run `ashiria extract` first.")
    cosines = {J.RIGHT: [], J.LEFT: []}
    assigned = correct = frames = any_hand = both_hands = pose_frames = 0
    for f in files:
        with np.load(f) as d:
            hw, hi, hp = d["hand_world"], d["hand_image"], d["hand_present"]
            pw, pi, pp = d["pose_world"], d["pose_image"], d["pose_present"]
        frames += len(hp)
        any_hand += int(hp.any(1).sum())
        both_hands += int(hp.all(1).sum())
        pose_frames += int(pp.sum())
        for s in (J.RIGHT, J.LEFT):
            other = J.LEFT if s == J.RIGHT else J.RIGHT
            mask = hp[:, s] & pp
            if not mask.any():
                continue
            own = np.linalg.norm(hi[mask, s, J.WRIST, :2] - pi[mask, POSE_HAND[s][0], :2], axis=-1)
            alt = np.linalg.norm(hi[mask, s, J.WRIST, :2] - pi[mask, POSE_HAND[other][0], :2], axis=-1)
            assigned += int(mask.sum())
            correct += int((own <= alt).sum())
            h = hw[mask, s]
            n_hand = normalize(np.cross(h[:, J.INDEX[0]] - h[:, J.WRIST], h[:, J.PINKY[0]] - h[:, J.WRIST]))
            w, i, p = (pw[mask, k] for k in POSE_HAND[s])
            n_pose = normalize(np.cross(i - w, p - w))
            cosines[s].extend((n_hand * n_pose).sum(-1).tolist())

    result = {
        "clips": len(files),
        "frames": frames,
        "any_hand": any_hand / max(frames, 1),
        "both_hands": both_hands / max(frames, 1),
        "pose": pose_frames / max(frames, 1),
        "slot_agreement": correct / max(assigned, 1),
    }
    print(f"{len(files)} clips, {frames} frames: a hand in {result['any_hand']:.0%}, both in {result['both_hands']:.0%}, "
          f"body pose in {result['pose']:.0%}")
    print(f"Slot assignment agrees with the body pose in {result['slot_agreement']:.1%} of hand frames")
    for s, name in ((J.RIGHT, "right"), (J.LEFT, "left")):
        c = np.array(cosines[s])
        if len(c):
            result[f"palm_cos_{name}"] = float(np.median(c))
            print(f"Palm normal, hand model vs body model ({name} hand): median cosine {np.median(c):+.2f}, "
                  f"same direction in {(c > 0).mean():.0%} of {len(c)} frames")
    verdict = min(result.get("palm_cos_right", 1), result.get("palm_cos_left", 1))
    if verdict < 0:
        print("PROBLEM: the hand model's axes point the opposite way to the body model's. Fix before building features.")
    elif verdict < 0.5:
        print("WARNING: weak agreement between hand and body orientation. Inspect a few clips before trusting orientation features.")
    else:
        print("OK: hand and body coordinates agree.")
    return result

"""`ashiria calibrate`: per-sign tau, tiers and the Free Sign thresholds, from simulated attempts.

Each simulated attempt starts from one of a sign's reference takes, perturbed like someone else's
signing (viewpoint, body size, finger shape, jitter, speed, dropouts). Hands rise from rest before
it and drop (or hold) after it, and the stream runs through the real attempt segmenter and
`prepare`, exactly as on the headset. From the attempts:

- tau per sign: the 90th percentile of a correct attempt's distance to its own sign;
- Free Sign's accept_z / min_margin_z: as many correct matches as possible with at most 5% wrong
  matches and at most 10% of out-of-set attempts accepted (other vocabulary signs when only some are
  learned, non-vocabulary dictionary signs, random hand movement); temperature by likelihood;
- per sign: recognition top-1/top-3, Verify pass rate, how often its look-alikes pass, segmentation misses;
- tiers (tiers.py) from those numbers and the tags.

All of this is optimistic: one signer and synthetic variation. Real Quest attempts replace it later.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .augment import augment
from .canonical import Segment, resample
from .config import Config, path
from .matching import dtw_many, segment_descriptor
from .params import EngineParams
from .pipeline import cut_reference, prepare
from .segmenter import segment_stream
from .signs import SignRef, Take, make_tolerances
from .store import SignSource
from .synthetic import random_segment
from .tiers import assign_tier, load_rules
from .verify import verify

VIDEO_REST_HEIGHT = -1.8  # the references' frame: standing signers on video
REST_Y = -2.4             # where simulated hands rest (arms down)
LEARNED_SET = 15
LEARNED_TRIALS = 20
LOOKALIKES = 3
MAX_WRONG, MAX_FALSE_ACCEPT = 0.05, 0.10


def simulated_stream(seg: Segment, rng: np.random.Generator, aug: dict, tick_hz: float,
                     drop_after_hold: bool = False) -> Segment:
    """A take as someone else might sign it, at the headset tick rate, with a rise from rest before it and
    either a drop back to rest or a long hold after it. `drop_after_hold`: the hold is followed by a drop too, as
    when the segmenter doesn't end attempts at a hold (the recogniser's whole signs) and the signer lowers their hands."""
    seg = augment(seg, rng, aug)
    n = max(8, int(round((len(seg) - 1) / seg.fps * tick_hz)) + 1)
    seg = resample(seg, n)
    used = seg.present.max(axis=0)  # slots that appear at all

    def frames(hand: np.ndarray, wrists: list[np.ndarray]) -> Segment:
        k = len(wrists)
        return Segment(np.repeat(hand[None], k, 0), np.stack(wrists), np.repeat(used[None], k, 0), seg.fps)

    first, last = seg.wrist_pos[0], seg.wrist_pos[-1]
    rest_first, rest_last = first.copy(), last.copy()
    # Only hands in signing space rise from (and drop to) rest; a passive hand already resting stays put.
    rising = first[:, 1] > VIDEO_REST_HEIGHT
    rest_first[rising, 1] = REST_Y
    rest_last[last[:, 1] > VIDEO_REST_HEIGHT, 1] = REST_Y
    rise = int(rng.integers(6, 13))
    parts = [frames(seg.hand_rel[0], [rest_first] * 10),
             frames(seg.hand_rel[0], [rest_first + (first - rest_first) * (i + 1) / (rise + 1) for i in range(rise)]),
             seg]
    held = rng.random() < 0.3
    if held:
        parts.append(frames(seg.hand_rel[-1], [last] * 25))  # holds the end of the sign
    if not held or drop_after_hold:
        drop = int(rng.integers(5, 11))
        parts.append(frames(seg.hand_rel[-1], [last + (rest_last - last) * (i + 1) / drop for i in range(drop)]
                            + [rest_last] * 10))
    return Segment(*(np.concatenate([getattr(s, a) for s in parts]) for a in ("hand_rel", "wrist_pos", "present")),
                   fps=tick_hz)


def simulate_attempt(take_seg: Segment, rng, aug: dict, p: EngineParams) -> Segment | None:
    """One prepared attempt, or None if the segmenter did not fire (a segmentation miss)."""
    sp = p.segmenter
    attempts = segment_stream(sp, simulated_stream(take_seg, rng, aug, sp.tick_hz, drop_after_hold=sp.hold_s >= sp.max_attempt_s))
    return prepare(attempts[0].segment, p, sp.rest_height) if attempts else None


def _descriptor_bank(refs: list[SignRef]) -> tuple[np.ndarray, np.ndarray]:
    bank = np.stack([t.descriptor for r in refs for t in r.takes]).astype(np.float64)
    owner = np.array([i for i, r in enumerate(refs) for _ in r.takes])
    return bank, owner


def _sign_distances(desc: np.ndarray, bank: np.ndarray, owner: np.ndarray, n_signs: int, band: int) -> np.ndarray:
    d = dtw_many(desc.astype(np.float64), bank, band)
    out = np.full(n_signs, np.inf)
    np.minimum.at(out, owner, d)
    return out


def recognition_rates(z_in: np.ndarray, truth: np.ndarray, z_out: np.ndarray, accept: float, margin: float) -> dict:
    """Outcomes of Free Sign's decision rule on in-set attempts (z [A, S], truth [A]) and out-of-set ones."""
    def decide(z):
        order = np.argsort(z, axis=1, kind="stable")
        rows = np.arange(len(z))
        z1 = z[rows, order[:, 0]]
        z2 = z[rows, order[:, 1]] if z.shape[1] > 1 else np.full(len(z), np.inf)
        not_sure = z1 > accept
        ambiguous = ~not_sure & (z2 - z1 < margin) & (z2 <= accept)
        return order, not_sure, ambiguous

    order, ns, amb = decide(z_in)
    match = ~ns & ~amb
    top2 = (order[:, 0] == truth) | ((order[:, 1] == truth) if z_in.shape[1] > 1 else False)
    out = {"correct": float((match & (order[:, 0] == truth)).mean()),
           "wrong": float((match & (order[:, 0] != truth)).mean()),
           "ambiguous_ok": float((amb & top2).mean()),
           "not_sure": float(ns.mean())}
    if len(z_out):
        _, ns_o, _ = decide(z_out)
        out["false_accept"] = float((~ns_o).mean())
    else:
        out["false_accept"] = 0.0
    return out


def _scenarios(z_all: np.ndarray, truth: np.ndarray, z_distract: np.ndarray, rng) -> list[tuple]:
    """(z_in, truth_in, z_out) for the full vocabulary and for random learned sets."""
    n = z_all.shape[1]
    out = [(z_all, truth, z_distract)]
    if n > LEARNED_SET:
        for _ in range(LEARNED_TRIALS):
            learned = np.sort(rng.choice(n, LEARNED_SET, replace=False))
            remap = -np.ones(n, int)
            remap[learned] = np.arange(LEARNED_SET)
            inside = np.isin(truth, learned)
            z_out = np.concatenate([z_all[~inside][:, learned], z_distract[:, learned]])
            out.append((z_all[inside][:, learned], remap[truth[inside]], z_out))
    return out


def _pooled(scenarios, accept, margin) -> dict:
    rates = [recognition_rates(zi, t, zo, accept, margin) for zi, t, zo in scenarios]
    full, learned = rates[0], rates[1:] or rates[:1]
    return {k: 0.5 * full[k] + 0.5 * float(np.mean([r[k] for r in learned])) for k in full}


def choose_recognition(scenarios) -> tuple[float, float, dict]:
    best, best_key = None, None
    for accept in np.round(np.arange(0.8, 2.01, 0.05), 2):
        for margin in np.round(np.arange(0.0, 0.51, 0.05), 2):
            r = _pooled(scenarios, accept, margin)
            feasible = r["wrong"] <= MAX_WRONG and r["false_accept"] <= MAX_FALSE_ACCEPT
            key = (feasible, r["correct"] + 0.5 * r["ambiguous_ok"] if feasible else -(r["wrong"] + r["false_accept"]))
            if best_key is None or key > best_key:
                best, best_key = (float(accept), float(margin), r), key
    if not best_key[0]:
        print("warning: no setting keeps wrong matches <= 5% and false accepts <= 10%; using the least bad one")
    return best


def fit_temperature(z: np.ndarray, truth: np.ndarray) -> float:
    best = (np.inf, 1.0)
    for t in np.round(np.arange(0.05, 2.01, 0.05), 2):
        logits = -z / t
        logits -= logits.max(axis=1, keepdims=True)
        nll = -(logits[np.arange(len(z)), truth] - np.log(np.exp(logits).sum(axis=1))).mean()
        best = min(best, (nll, float(t)))
    return best[1]


def build_refs(ids: list[str], sources: dict[str, SignSource], p: EngineParams, tiers: dict[str, str],
               taus: dict[str, float], lookalikes: dict[str, list[str]]) -> dict[str, SignRef]:
    refs = {}
    for sid in ids:
        src = sources[sid]
        takes = [Take(t.segment, t.key_frame, t.source, t.take_id).bind(p) for t in src.takes]
        tol = make_tolerances(tiers.get(sid, "scored"), p.verify, takes, moving=src.tags.get("movement") == "moving",
                              two_hands=src.tags.get("hands") == "two")
        refs[sid] = SignRef(sid, tiers.get(sid, "scored"), takes, tol, tau=taus.get(sid, 1.0),
                            lookalikes=lookalikes.get(sid, []), shadow_take=src.shadow_take)
    return refs


def _verify_stats(refs: dict[str, SignRef], attempts: dict[str, list[Segment]], p: EngineParams) -> dict:
    out = {}
    for sid, ref in refs.items():
        if ref.tier == "watch":
            continue
        look = [refs[x] for x in ref.lookalikes if x in refs]
        own = [verify(a, ref, p, lookalikes=look).passed for a in attempts[sid]]
        other = [verify(a, ref, p, lookalikes=look).passed for x in ref.lookalikes for a in attempts.get(x, [])[:8]]
        out[sid] = {"verify_pass": float(np.mean(own)) if own else 0.0,
                    "lookalike_false_pass": float(np.mean(other)) if other else 0.0}
    return out


def calibrate(catalogue: dict, sources: dict[str, SignSource], p: EngineParams, aug: dict,
              distractors: list[Segment] = (), attempts: int = 16, seed: int = 7) -> tuple[dict, EngineParams]:
    """Returns (calibration report, parameters with the calibrated recognition thresholds)."""
    rng = np.random.default_rng(seed)
    p_sim = dataclasses.replace(p, segmenter=dataclasses.replace(p.segmenter, rest_height=VIDEO_REST_HEIGHT))
    entries = {e["id"]: e for e in catalogue["signs"]}
    ids = [e["id"] for e in catalogue["signs"] if e["id"] in sources]
    refs = build_refs(ids, sources, p, {}, {}, {})
    bank, owner = _descriptor_bank([refs[i] for i in ids])
    n = len(ids)

    # Look-alikes: the nearest other signs, reference to reference.
    sign_d = np.stack([np.min([_sign_distances(t.descriptor, bank, owner, n, p.dtw_band) for t in refs[i].takes],
                              axis=0) for i in ids])
    sign_d = np.minimum(sign_d, sign_d.T)
    np.fill_diagonal(sign_d, np.inf)
    lookalikes = {sid: [ids[j] for j in np.argsort(sign_d[i], kind="stable")[:LOOKALIKES]] for i, sid in enumerate(ids)}

    # Simulated attempts at every sign, and at things the app does not teach.
    sim: dict[str, list[Segment]] = {}
    misses: dict[str, float] = {}
    for sid in ids:
        takes = sources[sid].takes
        got = [simulate_attempt(takes[k % len(takes)].segment, rng, aug, p_sim) for k in range(attempts)]
        sim[sid] = [a for a in got if a is not None]
        misses[sid] = 1 - len(sim[sid]) / attempts
    junk = list(distractors) + [cut_reference(random_segment(rng, int(rng.integers(20, 50)), bool(rng.random() < 0.4),
                                                       True, 30.0), p, VIDEO_REST_HEIGHT) for _ in range(50)]
    out_of_set = [a for a in (simulate_attempt(s, rng, aug, p_sim) for s in junk) if a is not None]

    def distances(segs):
        if not segs:
            return np.zeros((0, n))
        return np.stack([_sign_distances(segment_descriptor(s, p.weights), bank, owner, n, p.dtw_band) for s in segs])

    truth = np.array([i for i, sid in enumerate(ids) for _ in sim[sid]])
    d_in = distances([a for sid in ids for a in sim[sid]])
    d_out = distances(out_of_set)

    # tau: 90th percentile of correct attempts' distance to their own sign, clamped around the median.
    tau = np.array([np.percentile(d_in[truth == i, i], 90) if (truth == i).any() else np.nan for i in range(n)])
    med = np.nanmedian(tau)
    tau = np.clip(np.where(np.isnan(tau), med, tau), 0.5 * med, 2.0 * med)
    taus = dict(zip(ids, tau.tolist()))
    z_in, z_out = d_in / tau, d_out / tau

    scenarios = _scenarios(z_in, truth, z_out, rng)
    accept, margin, rates = choose_recognition(scenarios)
    temperature = fit_temperature(z_in, truth)
    learned = [recognition_rates(zi, t, zo, accept, margin) for zi, t, zo in scenarios[1:]] or []
    rank = (z_in < z_in[np.arange(len(truth)), truth][:, None]).sum(axis=1) + 1

    # Tiers: tags and scored-Verify numbers decide; then the final numbers under the chosen tiers.
    rules = load_rules()
    provisional = _verify_stats(refs, sim, p)
    stats, tiers, tier_info = {}, {}, {}
    for i, sid in enumerate(ids):
        mine = rank[truth == i]
        stats[sid] = {"top1": float((mine == 1).mean()) if len(mine) else 0.0,
                      "top3": float((mine <= 3).mean()) if len(mine) else 0.0,
                      "segmentation_miss": misses[sid], **provisional.get(sid, {})}
        tier, source, reasons = assign_tier(entries[sid], sources[sid].tags, stats[sid], rules)
        tiers[sid], tier_info[sid] = tier, {"tier": tier, "tier_source": source, "tier_reasons": reasons}
    final_refs = build_refs(ids, sources, p, tiers, taus, lookalikes)
    final = _verify_stats(final_refs, sim, p)

    calibrated = dataclasses.replace(p, recognise=dataclasses.replace(
        p.recognise, accept_z=accept, min_margin_z=margin, temperature=temperature))
    report = {
        "seed": seed, "attempts_per_sign": attempts, "out_of_set_attempts": len(out_of_set),
        "simulated_rest_height": VIDEO_REST_HEIGHT, "augment": aug,
        "recognise": {"accept_z": accept, "min_margin_z": margin, "temperature": temperature},
        "rates": {"pooled": rates, "full_vocabulary": recognition_rates(z_in, truth, z_out, accept, margin),
                  f"learned_{LEARNED_SET}": {k: float(np.mean([r[k] for r in learned])) for k in rates} if learned else None},
        "signs": {sid: {"tau": round(taus[sid], 4), **tier_info[sid], **{k: round(v, 3) for k, v in stats[sid].items()},
                        "final_verify_pass": round(final.get(sid, {}).get("verify_pass", 0.0), 3) if sid in final else None,
                        "final_lookalike_false_pass": round(final[sid]["lookalike_false_pass"], 3) if sid in final else None,
                        "lookalikes": lookalikes[sid],
                        "lookalike_distances": [round(float(sign_d[i, ids.index(x)]), 3) for x in lookalikes[sid]]}
                  for i, sid in enumerate(ids)},
    }
    return report, calibrated


def run_calibration(cfg: Config, pack_id: str = "ksl-core", attempts: int = 16, seed: int = 7) -> tuple[Path, dict, EngineParams]:
    """Catalogue + reference stores -> exports/calibration/<pack>.json, a tier review CSV and a printed summary."""
    from .catalogue import catalogue_file, load_catalogue
    from .store import DICTIONARY, STUDIO, assemble, distractor_segments, load_dictionary, load_studio

    catalogue = load_catalogue(catalogue_file(cfg, pack_id))
    sources, missing = assemble(catalogue, load_dictionary(DICTIONARY), load_studio(STUDIO))
    if len(sources) < 2:
        raise SystemExit("Need references for at least two catalogue signs: run `ashiria references` (dictionary "
                         "drafts) or `ashiria import-studio` first.")
    if missing:
        print(f"No reference yet for {len(missing)} signs (left out): {', '.join(missing)}")
    p = EngineParams.from_config(cfg)
    distractors = distractor_segments(Path(DICTIONARY).parent / "distractors.npz")
    print(f"Simulating {attempts} attempts × {len(sources)} signs, plus {len(distractors)} dictionary distractors "
          "and 50 random movements...")
    report, calibrated = calibrate(catalogue, sources, p, dict(cfg.calibrate.augment), distractors, attempts, seed)
    report["missing"] = missing
    report["sources"] = {sid: s.kind for sid, s in sources.items()}

    out = path(cfg, "exports") / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    file = out / f"{pack_id}.json"
    file.write_text(json.dumps(report, indent=1))
    rows = ["sign,source,tier,tier_source,reasons,top1,top3,verify_pass,lookalike_false_pass,segmentation_miss,lookalikes"]
    gloss = {e["id"]: e["gloss"] for e in catalogue["signs"]}
    for sid, s in report["signs"].items():
        rows.append(",".join([gloss[sid], report["sources"][sid], s["tier"], s["tier_source"], '"' + "; ".join(s["tier_reasons"]) + '"',
                              f"{s['top1']:.2f}", f"{s['top3']:.2f}", f"{s.get('verify_pass', 0):.2f}",
                              f"{s.get('lookalike_false_pass', 0):.2f}", f"{s['segmentation_miss']:.2f}",
                              '"' + " ".join(gloss[x] for x in s["lookalikes"]) + '"']))
    (out / f"tiers_{pack_id}.csv").write_text("\n".join(rows) + "\n")

    r = report["rates"]
    tiers = [s["tier"] for s in report["signs"].values()]
    print(f"Free Sign thresholds: accept_z {report['recognise']['accept_z']}, min_margin_z {report['recognise']['min_margin_z']}, "
          f"temperature {report['recognise']['temperature']}")
    for name, rates in r.items():
        if rates:
            print(f"  {name:>16}: correct {rates['correct']:.0%}, wrong {rates['wrong']:.0%}, ambiguous-but-right "
                  f"{rates['ambiguous_ok']:.0%}, not sure {rates['not_sure']:.0%}, out-of-set accepted {rates['false_accept']:.0%}")
    print(f"Tiers: {tiers.count('scored')} scored, {tiers.count('lenient')} lenient, {tiers.count('watch')} watch-only")
    passes = [s["final_verify_pass"] for s in report["signs"].values() if s["final_verify_pass"] is not None]
    falses = [s["final_lookalike_false_pass"] for s in report["signs"].values() if s["final_lookalike_false_pass"] is not None]
    if passes:
        print(f"Verify under final tiers: correct attempts pass {np.mean(passes):.0%} (median sign {np.median(passes):.0%}); "
              f"look-alikes pass {np.mean(falses):.0%}")
    print(f"-> {file} and {out / f'tiers_{pack_id}.csv'} (optimistic: one signer, synthetic variation)")
    return file, report, calibrated

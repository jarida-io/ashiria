"""Command line: `ashiria <step>`.

Model training: audit, extract, build, baseline, train, evaluate, export.
Content for the headset: references (dictionary drafts) or import-studio (Studio takes), catalogue,
calibrate, pack.
"""

from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ashiria", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("audit", help="list videos, read labels and signers, write a report")
    p = steps.add_parser("extract", help="MediaPipe hand and pose landmarks for every video")
    p.add_argument("--limit", type=int, help="only this many clips (for a quick test)")
    p.add_argument("--force", action="store_true", help="redo clips that already have landmarks")
    p.add_argument("--workers", type=int)
    p = steps.add_parser("check", help="sanity checks on extracted landmarks (slots, axis agreement)")
    p.add_argument("--limit", type=int, default=400)
    steps.add_parser("build", help="canonical segments, label list and splits")
    steps.add_parser("baseline", help="nearest-neighbour accuracy as a sanity check")
    p = steps.add_parser("train", help="train the sign model")
    p.add_argument("--name", help="run folder name (default: timestamp)")
    p = steps.add_parser("evaluate", help="test accuracy, per-sign results and confusions")
    p.add_argument("--run", required=True, help="run folder, e.g. runs/20261015-101500")
    p = steps.add_parser("export", help="ONNX model, metadata and golden vectors for Unity")
    p.add_argument("--run", required=True)
    p.add_argument("--name", help="export folder name (default: the run's name)")
    p = steps.add_parser("import-dictionary", help="labels CSV from the KSL Dictionary scrape (use configs/ksl_dictionary.yaml)")
    p.add_argument("--source", help="ksl_dictionary_data.csv (default: dictionary.source_csv in the config)")
    p = steps.add_parser("references", help="per-sign reference templates and trackability tags")
    p.add_argument("--vocab", help="vocabulary file, e.g. configs/vocabulary_v1.txt")
    steps.add_parser("golden", help="engine test vectors from synthetic hands (spec/golden/) for the C# port")
    p = steps.add_parser("calibrate", help="simulated attempts -> per-sign tau, tiers and Free Sign thresholds")
    p.add_argument("--pack", default="ksl-core")
    p.add_argument("--attempts", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p = steps.add_parser("pack", help="build the content pack the headset loads (content/packs/)")
    p.add_argument("--pack", default="ksl-core")
    p.add_argument("--release", action="store_true", help="refuse unless every source has permission and review")
    p.add_argument("--attempts", type=int, default=16)
    p = steps.add_parser("import-studio", help="Studio takes (pulled from the headset) -> exports/studio/store.json")
    p.add_argument("folder", help="folder of take .json files, e.g. data/studio/2026-11-02")
    p.add_argument("--keep-outliers", action="store_true")
    p.add_argument("--pack", default="ksl-core")
    p = steps.add_parser("studio-fake", help="fake Studio takes from the dictionary references (for testing)")
    p.add_argument("out", nargs="?", default="data/studio/fake")
    p.add_argument("--signs", nargs="*", help="catalogue ids (default: all)")
    p.add_argument("--takes", type=int, default=3)
    p = steps.add_parser("approve", help="record your review of catalogue signs (approved, rejected or back to draft)")
    p.add_argument("ids", nargs="+", help="sign ids, e.g. letter-a number-two")
    p.add_argument("--status", default="approved", choices=["approved", "rejected", "draft"])
    p.add_argument("--by", default="Jerry")
    p.add_argument("--pack", default="ksl-core")
    p = steps.add_parser("study-data", help="JSON for the dev-only study sheet (catalogue, reviews, videos, previews)")
    p.add_argument("--pack", default="ksl-core")
    p = steps.add_parser("audit-dictionary", help="check every KSL Dictionary video: download, MediaPipe tracking, duplicates")
    p.add_argument("--workers", type=int, default=8)
    steps.add_parser("study-index", help="JSON index of every sign in the dictionary data sheet (dev-only sign browser)")
    p = steps.add_parser("study-avatar", help="JSON avatar (pose and steps) for one catalogue sign")
    p.add_argument("sign_id")
    p = steps.add_parser("study-pose", help="JSON body and hands of one dictionary clip, for the avatar")
    p.add_argument("clip_id")
    p = steps.add_parser("study-reference", help="JSON reference hands cut from one dictionary clip")
    p.add_argument("clip_id")
    p = steps.add_parser("study-landmarks", help="JSON MediaPipe image landmarks of one dictionary clip")
    p.add_argument("clip_id")
    p = steps.add_parser("export-recogniser", help="the recogniser for the web app's Interpret mode (exports/recogniser)")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p = steps.add_parser("embed-evaluate", help="re-test a trained embed run (report.json, then signs.json)")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p.add_argument("--attempts", type=int, default=4)
    p = steps.add_parser("embed-signs", help="per-sign results of an embed run, for the sign browser")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p.add_argument("--attempts", type=int, default=3)
    p = steps.add_parser("embed-views", help="recogniser accuracy from a grid of camera positions and a mirrored camera")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p.add_argument("--attempts", type=int, default=1)
    p = steps.add_parser("sign-groups", help="every sign by kind and into look-alike groups, with what tells them apart")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p = steps.add_parser("sentences-eval", help="reading signs in a row: random runs, the phrases, recorded phrases")
    p.add_argument("--run", help="run folder (default: the newest embed run)")
    p = steps.add_parser("lookalike-rounds", help="fine-tune the recogniser in rounds on the signs it mixes up")
    p.add_argument("--start", required=True, help="run folder to start from")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--share", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p = steps.add_parser("embed", help="experiment: sign embedding over all dictionary signs, compared with DTW")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--lookalike-share", type=float, default=0.5,
                   help="share of each batch given to look-alike neighbourhoods (needs `ashiria sign-groups`; 0 = random)")
    p.add_argument("--context-share", type=float, default=0.0,
                   help="share of training pairs where one view is the sign cut out of a sentence (phrases.py)")
    p.add_argument("--init-run", help="fine-tune this run's model instead of training from scratch")
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (fine-tuning: about 3e-4)")
    p = steps.add_parser("steps-check", help="how often the avatar's step captions agree with the dictionary descriptions")
    p.add_argument("--limit", type=int)
    p.add_argument("--rules", action="store_true", help="the hand-written rules only, without the fitted model")
    p = steps.add_parser("steps-fit", help="fit the step captions' calls to the dictionary descriptions (exports/steps)")
    p.add_argument("--limit", type=int)
    p = steps.add_parser("catalogue", help="create or extend content/catalogue/<pack>.yaml from a vocabulary file")
    p.add_argument("--vocab", default="configs/vocabulary_v1.txt")
    p.add_argument("--pack", default="ksl-core")

    # Accept --config after the step name as well (ashiria audit --config x.yaml).
    for sub in steps.choices.values():
        sub.add_argument("--config", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.step == "audit":
        from .audit import audit
        audit(cfg)
    elif args.step == "extract":
        from .extract import extract
        extract(cfg, limit=args.limit, force=args.force, workers=args.workers)
    elif args.step == "check":
        from .diagnose import check
        check(cfg, limit=args.limit)
    elif args.step == "build":
        from .data import build
        build(cfg)
    elif args.step == "baseline":
        from .baseline import baseline
        baseline(cfg)
    elif args.step == "train":
        from .train import train
        train(cfg, name=args.name)
    elif args.step == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, args.run)
    elif args.step == "export":
        from .export import export
        export(cfg, args.run, name=args.name)
    elif args.step == "import-dictionary":
        from .dictionary import import_dictionary
        import_dictionary(cfg, args.source)
    elif args.step == "references":
        from .references import build_references
        build_references(cfg, args.vocab)
    elif args.step == "golden":
        from .golden import build_camera_golden, build_golden, build_runtime_golden
        build_golden(cfg)
        build_runtime_golden(cfg)
        build_camera_golden(cfg)
    elif args.step == "calibrate":
        from .calibrate import run_calibration
        run_calibration(cfg, args.pack, args.attempts, args.seed)
    elif args.step == "pack":
        from .pack import build_pack
        build_pack(cfg, args.pack, args.release, args.attempts)
    elif args.step == "steps-check":
        from .steps_check import check
        check(args.limit, use_model=not args.rules)
    elif args.step == "steps-fit":
        from .steps_fit import run_fit
        run_fit(args.limit)
    elif args.step == "import-studio":
        from .studio import import_studio
        import_studio(cfg, args.folder, args.keep_outliers, args.pack)
    elif args.step == "studio-fake":
        from .studio import write_fake_takes
        write_fake_takes(cfg, args.out, args.signs, args.takes)
    elif args.step == "approve":
        from .catalogue import set_review
        changed = set_review(cfg, args.pack, args.ids, args.status, args.by)
        print(f"{args.status}: {', '.join(changed)}")
    elif args.step == "study-data":
        import json as _json
        from .study import study_data
        print(_json.dumps(study_data(cfg, args.pack)))
    elif args.step == "study-index":
        import json as _json
        from .study import dictionary_index
        print(_json.dumps(dictionary_index()))
    elif args.step == "study-landmarks":
        import json as _json
        from .study import clip_landmarks
        print(_json.dumps(clip_landmarks(args.clip_id)))
    elif args.step == "audit-dictionary":
        from .dictionary_audit import audit_dictionary
        audit_dictionary(args.workers)
    elif args.step == "study-avatar":
        import json as _json
        from .study import sign_avatar
        print(_json.dumps(sign_avatar(cfg, args.sign_id)))
    elif args.step == "study-pose":
        import json as _json
        from .study import clip_pose
        print(_json.dumps(clip_pose(args.clip_id)))
    elif args.step == "study-reference":
        import json as _json
        from .study import clip_reference
        print(_json.dumps(clip_reference(args.clip_id)))
    elif args.step == "export-recogniser":
        from .recogniser_export import export_recogniser
        export_recogniser(cfg, args.run)
    elif args.step == "embed-evaluate":
        from .embed import evaluate_run
        evaluate_run(cfg, args.run, args.attempts)
    elif args.step == "embed-signs":
        from .embed import sign_results
        sign_results(cfg, args.run, args.attempts)
    elif args.step == "embed-views":
        from .embed import perspective_results
        perspective_results(cfg, args.run, args.attempts)
    elif args.step == "sign-groups":
        from .groups import build_groups
        build_groups(cfg, args.run)
    elif args.step == "sentences-eval":
        from .sentences import evaluate_sentences
        evaluate_sentences(cfg, args.run)
    elif args.step == "lookalike-rounds":
        from .embed import lookalike_rounds
        lookalike_rounds(cfg, args.start, args.rounds, args.share, args.epochs)
    elif args.step == "embed":
        from .embed import run_embed
        run_embed(cfg, epochs=args.epochs, attempts=args.attempts, lookalike_share=args.lookalike_share,
                  context_share=args.context_share, init_run=args.init_run, lr=args.lr)
    elif args.step == "catalogue":
        from .catalogue import update_catalogue
        update_catalogue(cfg, args.vocab, args.pack)


if __name__ == "__main__":
    main()

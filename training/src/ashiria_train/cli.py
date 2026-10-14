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
        from .golden import build_golden, build_runtime_golden
        build_golden(cfg)
        build_runtime_golden(cfg)
    elif args.step == "calibrate":
        from .calibrate import run_calibration
        run_calibration(cfg, args.pack, args.attempts, args.seed)
    elif args.step == "pack":
        from .pack import build_pack
        build_pack(cfg, args.pack, args.release, args.attempts)
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
    elif args.step == "catalogue":
        from .catalogue import update_catalogue
        update_catalogue(cfg, args.vocab, args.pack)


if __name__ == "__main__":
    main()

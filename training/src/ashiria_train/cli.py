"""Command line: `ashiria <step>`. Steps run in order: audit, extract, build, baseline, train, evaluate, export."""

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
    p = steps.add_parser("simulate", help="optimistic Free Sign check: perturbed references vs all references")
    p.add_argument("--attempts", type=int, default=8)

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
        from .golden import build_golden
        build_golden(cfg)
    elif args.step == "simulate":
        from .simulate import simulate
        simulate(cfg, attempts=args.attempts)


if __name__ == "__main__":
    main()

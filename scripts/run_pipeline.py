#!/usr/bin/env python
"""
Thin CLI entry point tying the aem_degradation package together. Mirrors,
stage by stage, what the original flat scripts did as `__main__`:

    prepare_step1.py   -> clean
    build_splits.py    -> split
    model_<x>.py        -> train [--models rf gb ...] (default: all)
    shap_analysis.py    -> interpret [--winner rf]
    compare_zou.py /
    zou_top20_ranking.py -> benchmark [--winner rf]
    model_rf_rowlevel_diagnostic.py,
    kfold_sensitivity.py -> diagnostics
    make_figures.py     -> figures [--winner rf]

    all                 -> clean, train (all families), interpret, benchmark,
                           diagnostics, figures, in that order. `all` does NOT
                           re-split: it reuses the shipped fold assignment, so
                           it reproduces the paper's numbers. --rebuild-splits
                           adds the split stage back.

Every stage reads/writes exactly the same data/results files the original
scripts did (see src/aem_degradation/paths.py) -- nothing here changes the
protocol or any hyperparameter, only how the code is organized and invoked.

Usage examples:
    python scripts/run_pipeline.py clean
    python scripts/run_pipeline.py split
    python scripts/run_pipeline.py train --models rf gb
    python scripts/run_pipeline.py interpret --winner rf
    python scripts/run_pipeline.py benchmark --winner rf
    python scripts/run_pipeline.py diagnostics
    python scripts/run_pipeline.py figures --winner rf
    python scripts/run_pipeline.py all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    sys.exit(
        f"This pipeline requires Python 3.12 or newer (found "
        f"{sys.version_info.major}.{sys.version_info.minor}). The pinned "
        f"numpy/scipy/shap versions in requirements.txt do not build on older "
        f"interpreters."
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_degradation.benchmarking import ZouBenchmark
from aem_degradation.data import DataCleaner
from aem_degradation.diagnostics import KFoldSensitivity, RowLevelDiagnostic
from aem_degradation.figures import FigureGenerator
from aem_degradation.interpretability import SHAPExplainer
from aem_degradation.models import REGISTRY
from aem_degradation.protocol import ExperimentProtocol
from aem_degradation.splits import SplitBuilder

DEFAULT_WINNER = "rf"  # random forest -- see results/rf.json / README


def cmd_clean(_args):
    DataCleaner().run()


def cmd_split(_args):
    SplitBuilder().run()


def cmd_train(args):
    protocol = ExperimentProtocol()
    names = args.models or list(REGISTRY.keys())
    for name in names:
        if name not in REGISTRY:
            raise SystemExit(f"unknown model family: {name} (choices: {list(REGISTRY)})")
        print(f"\n=== training {name} ===")
        REGISTRY[name]().run(protocol)


def cmd_interpret(args):
    protocol = ExperimentProtocol()
    winner = args.winner
    dev_df, test_df = protocol.load_data(log_time=REGISTRY[winner]().log_time)
    SHAPExplainer(winner).run(dev_df, test_df)


def cmd_benchmark(args):
    ZouBenchmark().compare(args.winner)


def cmd_diagnostics(_args):
    print("=== row-level vs. grouped split diagnostic ===")
    RowLevelDiagnostic().run()
    print("\n=== k-fold sensitivity diagnostic ===")
    KFoldSensitivity().run()


def cmd_figures(args):
    FigureGenerator().run(args.winner)


def cmd_all(args):
    cmd_clean(args)
    if getattr(args, "rebuild_splits", False):
        cmd_split(args)
    else:
        print("=== split: skipped, reusing the shipped fold assignment ===")
        print("    data/database_with_splits.csv already holds the exact cv_fold and")
        print("    test_holdout columns behind every number in the paper, so `all`")
        print("    reuses them and reproduces those numbers. Pass --rebuild-splits to")
        print("    regenerate the columns instead; that yields a different, equally")
        print("    valid partition and different scores (see the README).")
    cmd_train(argparse.Namespace(models=None))
    cmd_interpret(args)
    cmd_benchmark(args)
    cmd_diagnostics(args)
    cmd_figures(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("clean", help="raw export -> database_prepared.csv").set_defaults(func=cmd_clean)
    sub.add_parser("split", help="build the three split columns").set_defaults(func=cmd_split)

    t = sub.add_parser("train", help="run the grid-search -> OOF -> locked-test protocol")
    t.add_argument("--models", nargs="+", choices=list(REGISTRY.keys()), default=None,
                   help="which families to train (default: all)")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("interpret", help="per-fold SHAP stability + final SHAP + axiom checks")
    i.add_argument("--winner", default=DEFAULT_WINNER, choices=list(REGISTRY.keys()))
    i.set_defaults(func=cmd_interpret)

    b = sub.add_parser("benchmark", help="compare winner's SHAP ranking against Zou et al. Fig. 3a")
    b.add_argument("--winner", default=DEFAULT_WINNER, choices=list(REGISTRY.keys()))
    b.set_defaults(func=cmd_benchmark)

    sub.add_parser("diagnostics", help="row-level-split and k-fold-sensitivity diagnostics (not the protocol)").set_defaults(func=cmd_diagnostics)

    f = sub.add_parser("figures", help="regenerate the three paper figures")
    f.add_argument("--winner", default=DEFAULT_WINNER, choices=list(REGISTRY.keys()))
    f.set_defaults(func=cmd_figures)

    a = sub.add_parser("all", help="run every stage above in order, for the default winner")
    a.add_argument("--winner", default=DEFAULT_WINNER, choices=list(REGISTRY.keys()))
    a.add_argument("--rebuild-splits", action="store_true",
                   help="also run the split stage, overwriting the shipped fold "
                        "assignment; without this flag `all` reuses it, which is "
                        "what reproduces the paper's numbers")
    a.set_defaults(func=cmd_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

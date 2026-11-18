from __future__ import annotations

import argparse
from pathlib import Path

from force2020_reliability.analysis.manifest import build_pilot_manifest
from force2020_reliability.analysis.plots import make_all_figures
from force2020_reliability.analysis.stage2 import summarize_stage2
from force2020_reliability.analysis.stage2_plots import make_stage2_figures
from force2020_reliability.analysis.summarize import analyze_pilot
from force2020_reliability.config import load_config
from force2020_reliability.data.download import download_official_data
from force2020_reliability.data.las import (
    load_processed,
    parse_las_corpus,
    summarize_classes,
    write_parse_summary,
)
from force2020_reliability.data.stratigraphy import DEFAULT_OUTPUT as STRATIFIED_PATH
from force2020_reliability.data.stratigraphy import load_stratified
from force2020_reliability.models.feature_ablation import run_feature_ablation
from force2020_reliability.models.lightgbm_baseline import run_lightgbm_baseline
from force2020_reliability.models.nested import run_nested_calibration
from force2020_reliability.models.runner import run_pilot
from force2020_reliability.models.segment_eval import run_segment_evaluation
from force2020_reliability.models.segment_postprocess import analyze_segment_smoothing
from force2020_reliability.paths import INTERIM_DIR, ensure_runtime_dirs
from force2020_reliability.splits.generate import generate_all_splits
from force2020_reliability.splits.sensitivity import run_split_sensitivity


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment JSON config (default: configs/pilot.json)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="force2020")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download and verify official data")
    download_parser.set_defaults(handler=lambda _: download_official_data())

    parse_parser = subparsers.add_parser("parse", help="Parse LAS files into parquet")
    _config_argument(parse_parser)
    parse_parser.add_argument("--overwrite", action="store_true")

    splits_parser = subparsers.add_parser("splits", help="Generate all CV manifests")
    _config_argument(splits_parser)

    pilot_parser = subparsers.add_parser("pilot", help="Run pilot cross-validation")
    _config_argument(pilot_parser)
    pilot_parser.add_argument("--models", nargs="+", default=None)
    pilot_parser.add_argument("--max-folds", type=int, default=None)

    analyze_parser = subparsers.add_parser("analyze", help="Aggregate metrics and draw figures")
    analyze_parser.add_argument("--bootstrap", type=int, default=500)

    sensitivity_parser = subparsers.add_parser(
        "sensitivity", help="Run split-seed and spatial-scheme sensitivity"
    )
    sensitivity_parser.add_argument("--rows-per-well", type=int, default=1000)
    sensitivity_parser.add_argument("--models", nargs="+", default=None)

    nested_parser = subparsers.add_parser(
        "nested", help="Run nested selective-threshold calibration"
    )
    nested_parser.add_argument("--rows-per-well", type=int, default=1000)
    nested_parser.add_argument("--target-accuracy", type=float, default=0.90)

    ablation_parser = subparsers.add_parser("ablation", help="Run feature-set ablation")
    ablation_parser.add_argument("--rows-per-well", type=int, default=1000)

    lightgbm_parser = subparsers.add_parser("lightgbm", help="Run LightGBM baseline")
    lightgbm_parser.add_argument("--rows-per-well", type=int, default=1000)

    segment_parser = subparsers.add_parser(
        "segments", help="Evaluate continuous-depth boundary and thickness metrics"
    )
    segment_parser.add_argument("--gap-threshold-m", type=float, default=0.5)
    segment_parser.add_argument("--boundary-tolerance-m", type=float, default=0.5)
    segment_parser.add_argument("--max-folds", type=int, default=None)

    subparsers.add_parser("smoothsegments", help="Evaluate minimum-run smoothing")

    subparsers.add_parser("stage2", help="Summarize sensitivity and nested calibration")

    subparsers.add_parser("manifest", help="Write reproducibility manifest")

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_runtime_dirs()
    args = build_parser().parse_args(argv)
    config = load_config(getattr(args, "config", None))

    if args.command == "download":
        result = download_official_data()
    elif args.command == "parse":
        frame, audit = parse_las_corpus(config, overwrite=args.overwrite)
        summarize_classes(
            frame,
            target_column=config.target_column,
            output_path=INTERIM_DIR / "class_counts.csv",
        )
        result = write_parse_summary(frame, audit, config)
    elif args.command == "splits":
        frame = load_processed()
        result = generate_all_splits(frame, config)
    elif args.command == "pilot":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_pilot(
            frame,
            config,
            models=args.models,
            max_folds=args.max_folds,
        )
    elif args.command == "analyze":
        analysis_result = analyze_pilot(n_bootstrap=args.bootstrap)
        figures = make_all_figures()
        result = {"analysis": analysis_result, "figures": figures}
    elif args.command == "sensitivity":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_split_sensitivity(
            frame,
            config,
            rows_per_well=args.rows_per_well,
            models=args.models,
        )
    elif args.command == "nested":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_nested_calibration(
            frame,
            config,
            rows_per_well=args.rows_per_well,
            target_accuracy=args.target_accuracy,
        )
    elif args.command == "ablation":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_feature_ablation(
            frame,
            config,
            rows_per_well=args.rows_per_well,
        )
    elif args.command == "lightgbm":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_lightgbm_baseline(
            frame,
            config,
            rows_per_well=args.rows_per_well,
        )
    elif args.command == "segments":
        frame = load_stratified() if STRATIFIED_PATH.exists() else load_processed()
        result = run_segment_evaluation(
            frame,
            config,
            gap_threshold_m=args.gap_threshold_m,
            boundary_tolerance_m=args.boundary_tolerance_m,
            max_folds=args.max_folds,
        )
    elif args.command == "smoothsegments":
        result = analyze_segment_smoothing()
    elif args.command == "stage2":
        summaries = summarize_stage2()
        figures = make_stage2_figures()
        result = {"summaries": summaries, "figures": figures}
    elif args.command == "manifest":
        result = build_pilot_manifest()
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
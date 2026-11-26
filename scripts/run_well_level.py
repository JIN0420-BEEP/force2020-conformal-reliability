"""Entry point: well-level conformal experiment across multiple model families."""

import argparse
from dataclasses import replace

import pandas as pd

from force2020_reliability.config import load_config
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.models.well_level_experiment import (
    FEATURE_SETS,
    run_well_experiment,
)
from force2020_reliability.paths import INTERIM_DIR

DATA = INTERIM_DIR / "force2020_labeled_strat.parquet"
DEFAULT_MODELS = [
    "hist_gradient_boosting",
    "random_forest",
    "lightgbm",
    "extra_trees",
    "logistic",
    "knn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-well", type=int, default=1000)
    parser.add_argument("--output", default="well_level")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--schemes", nargs="+", default=["blocks5_buffer20"])
    parser.add_argument("--protocols", nargs="+", default=[
        "random_point", "well_grouped", "spatial_block"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--feature-set", default="full", choices=FEATURE_SETS)
    args = parser.parse_args()

    frame = pd.read_parquet(DATA)
    config = load_config()
    if args.seed is not None:
        config = replace(config, raw={**config.raw, "seed": args.seed})

    filtered, _ = filter_supported_classes(
        frame, target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells)
    sampled = sample_pilot_rows(filtered, args.rows_per_well, config.seed)
    print(f"sampled rows: {len(sampled)} wells: {sampled['well_id'].nunique()}")

    output_dir = run_well_experiment(
        sampled, config,
        model_names=args.models,
        schemes=args.schemes,
        output_name=args.output,
        seed=config.seed,
        protocols=args.protocols,
        feature_set=args.feature_set)
    print("saved to", output_dir)


if __name__ == "__main__":
    main()

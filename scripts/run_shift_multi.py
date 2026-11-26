"""Per-fold distribution shift across multiple seeds and both spatial schemes."""

import argparse
from dataclasses import replace

import pandas as pd

from force2020_reliability.config import load_config
from force2020_reliability.evaluation.distribution_shift import fold_shift
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

DATA = INTERIM_DIR / "force2020_labeled_strat.parquet"
PROTOCOLS = ("random_point", "well_grouped", "spatial_block")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-well", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20261021, 20261022, 20261023])
    parser.add_argument("--schemes", nargs="+", default=["blocks5_buffer20"])
    parser.add_argument("--output", default="well_level/shift_long.csv")
    args = parser.parse_args()

    frame = pd.read_parquet(DATA)
    base_config = load_config()

    rows = []
    for seed in args.seeds:
        config = replace(base_config, raw={**base_config.raw, "seed": seed})
        filtered, _ = filter_supported_classes(
            frame, config.target_column,
            config.min_class_count, config.min_class_wells)
        sampled = sample_pilot_rows(filtered, args.rows_per_well, seed)
        for scheme in args.schemes:
            splits_dir = (
                INTERIM_DIR / "splits" / "sensitivity" /
                f"seed_{seed}" / scheme)
            for protocol in PROTOCOLS:
                for fold in range(config.n_folds):
                    train_pos, test_pos = resolve_fold_members(
                        protocol, fold, sampled, splits_dir)
                    train = sampled.iloc[train_pos]
                    test = sampled.iloc[test_pos]
                    metrics = fold_shift(
                        train[config.numeric_features].to_numpy(),
                        test[config.numeric_features].to_numpy(),
                        train[config.target_column].to_numpy(),
                        test[config.target_column].to_numpy(),
                        seed=seed + fold)
                    metrics.update({
                        "protocol": protocol, "fold": fold,
                        "scheme": scheme, "seed": seed})
                    rows.append(metrics)
                    print(f"shift {seed}/{scheme}/{protocol} f{fold} "
                          f"mmd={metrics['mmd']:.4f}", flush=True)

    table = pd.DataFrame(rows)
    output = RESULTS_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    print("saved", output, len(table), "rows")


if __name__ == "__main__":
    main()

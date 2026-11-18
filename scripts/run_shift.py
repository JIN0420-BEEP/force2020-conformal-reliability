"""Entry point: per-fold distribution shift across the three protocols."""

import pandas as pd

from force2020_reliability.config import load_config
from force2020_reliability.evaluation.distribution_shift import fold_shift
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")


def main() -> None:
    config = load_config()
    frame = pd.read_parquet(INTERIM_DIR / "force2020_labeled_strat.parquet")
    filtered, _ = filter_supported_classes(
        frame, config.target_column,
        config.min_class_count, config.min_class_wells)
    sampled = sample_pilot_rows(filtered, 1000, config.seed)
    splits_dir = (
        INTERIM_DIR / "splits" / "sensitivity" /
        f"seed_{config.seed}" / "blocks5_buffer20")

    rows = []
    for protocol in PROTOCOLS:
        for fold in range(config.n_folds):
            train_pos, test_pos = resolve_fold_members(
                protocol, fold, sampled, splits_dir)
            train, test = sampled.iloc[train_pos], sampled.iloc[test_pos]
            metrics = fold_shift(
                train[config.numeric_features].to_numpy(),
                test[config.numeric_features].to_numpy(),
                train[config.target_column].to_numpy(),
                test[config.target_column].to_numpy(),
                seed=config.seed + fold,
            )
            metrics.update({"protocol": protocol, "fold": fold})
            rows.append(metrics)
            print(f"shift/{protocol} fold={fold} mmd={metrics['mmd']:.4f} "
                  f"js={metrics['label_js']:.4f}", flush=True)

    output = RESULTS_DIR / "conformal" / "shift_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print("saved", output)


if __name__ == "__main__":
    main()

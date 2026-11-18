from __future__ import annotations

import json

import pandas as pd

from force2020_reliability.config import ExperimentConfig
from force2020_reliability.models.runner import (
    PROTOCOLS,
    filter_supported_classes,
    run_protocol,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR


def run_lightgbm_baseline(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    rows_per_well: int = 1000,
) -> pd.DataFrame:
    results_root = RESULTS_DIR / "lightgbm"
    results_root.mkdir(parents=True, exist_ok=True)
    split_dir = (
        INTERIM_DIR / "splits" / "sensitivity" / f"seed_{config.seed}" / "blocks5_buffer20"
    )
    if not split_dir.exists():
        raise FileNotFoundError("Run `force2020 sensitivity` before LightGBM baseline.")

    filtered, _ = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    sampled = sample_pilot_rows(filtered, rows_per_well, config.seed)
    sampled.to_parquet(results_root / "lightgbm_sample.parquet", index=False)
    for protocol in PROTOCOLS:
        run_protocol(
            protocol=protocol,
            model_name="lightgbm",
            frame=sampled,
            config=config,
            output_dir=results_root,
            splits_dir=split_dir,
        )

    metric_frames = []
    for path in sorted(results_root.glob("*/lightgbm/metrics.csv")):
        metrics = pd.read_csv(path)
        metric_frames.append(metrics)
    metrics_frame = pd.concat(metric_frames, ignore_index=True)
    metrics_frame.to_csv(
        results_root / "all_metrics.csv", index=False, encoding="utf-8-sig"
    )
    folds = metrics_frame.loc[metrics_frame["fold"] >= 0]
    summary = (
        folds.groupby("protocol", observed=True)
        .agg(
            n_folds=("fold", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            ece_mean=("ece_top_label", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(results_root / "summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "rows_per_well": rows_per_well,
        "sample_rows": len(sampled),
        "model": "lightgbm",
        "protocols": list(PROTOCOLS),
        "split_seed": config.seed,
    }
    (results_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
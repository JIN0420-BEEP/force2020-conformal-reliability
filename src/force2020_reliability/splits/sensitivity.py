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
from force2020_reliability.splits.generate import (
    generate_all_splits,
    make_spatial_split,
)

SENSITIVITY_SEEDS = (20261021, 20261022, 20261023)
SPATIAL_SCHEMES = {
    "blocks5_buffer20": {"clusters": 5, "buffer_m": 20_000},
    "blocks6_buffer30": {"clusters": 6, "buffer_m": 30_000},
}


def _config_with(
    config: ExperimentConfig,
    seed: int,
    clusters: int,
    buffer_m: float,
) -> ExperimentConfig:
    raw = dict(config.raw)
    raw["seed"] = seed
    raw["spatial_clusters"] = clusters
    raw["spatial_buffer_m"] = buffer_m
    return ExperimentConfig(raw=raw)


def run_split_sensitivity(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    rows_per_well: int = 1000,
    seeds: tuple[int, ...] = SENSITIVITY_SEEDS,
    models: list[str] | None = None,
) -> pd.DataFrame:
    results_root = RESULTS_DIR / "sensitivity"
    results_root.mkdir(parents=True, exist_ok=True)
    split_root = INTERIM_DIR / "splits" / "sensitivity"
    split_root.mkdir(parents=True, exist_ok=True)
    selected_models = models or ["hist_gradient_boosting"]

    filtered, class_counts = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    sampled = sample_pilot_rows(
        filtered,
        max_rows_per_well=rows_per_well,
        seed=config.seed,
    )
    sampled.to_parquet(results_root / "sensitivity_sample.parquet", index=False)
    class_counts.to_csv(
        results_root / "class_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_rows: list[pd.DataFrame] = []
    for seed in seeds:
        primary_config = _config_with(
            config,
            seed=seed,
            clusters=SPATIAL_SCHEMES["blocks5_buffer20"]["clusters"],
            buffer_m=SPATIAL_SCHEMES["blocks5_buffer20"]["buffer_m"],
        )
        primary_split_dir = split_root / f"seed_{seed}" / "blocks5_buffer20"
        generate_all_splits(sampled, primary_config, primary_split_dir)
        for model_name in selected_models:
            for protocol in PROTOCOLS:
                output_dir = results_root / f"seed_{seed}" / "blocks5_buffer20"
                run_protocol(
                    protocol=protocol,
                    model_name=model_name,
                    frame=sampled,
                    config=primary_config,
                    output_dir=output_dir,
                    splits_dir=primary_split_dir,
                )

        secondary_config = _config_with(
            config,
            seed=seed,
            clusters=SPATIAL_SCHEMES["blocks6_buffer30"]["clusters"],
            buffer_m=SPATIAL_SCHEMES["blocks6_buffer30"]["buffer_m"],
        )
        secondary_split_dir = split_root / f"seed_{seed}" / "blocks6_buffer30"
        make_spatial_split(sampled, secondary_config, secondary_split_dir)
        for model_name in selected_models:
            output_dir = results_root / f"seed_{seed}" / "blocks6_buffer30"
            run_protocol(
                protocol="spatial_block",
                model_name=model_name,
                frame=sampled,
                config=secondary_config,
                output_dir=output_dir,
                splits_dir=secondary_split_dir,
            )

    for metrics_path in sorted(results_root.glob("seed_*/blocks*/*/*/metrics.csv")):
        scheme = metrics_path.parent.parent.parent.name
        seed = int(metrics_path.parent.parent.parent.parent.name.removeprefix("seed_"))
        metrics = pd.read_csv(metrics_path)
        metrics["seed"] = seed
        metrics["spatial_scheme"] = scheme
        metric_rows.append(metrics)
    all_metrics = pd.concat(metric_rows, ignore_index=True)
    all_metrics.to_csv(
        results_root / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics = all_metrics.loc[all_metrics["fold"] >= 0]
    summary = (
        fold_metrics.groupby(["model", "protocol", "spatial_scheme"], observed=True)
        .agg(
            n_runs=("fold", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            coverage90_mean=("coverage_at_target", "mean"),
            coverage90_std=("coverage_at_target", "std"),
        )
        .reset_index()
    )
    summary.to_csv(
        results_root / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "rows_per_well": rows_per_well,
        "sample_rows": len(sampled),
        "sample_wells": int(sampled["well_id"].nunique()),
        "seeds": list(seeds),
        "spatial_schemes": SPATIAL_SCHEMES,
        "models": selected_models,
        "protocols": list(PROTOCOLS),
    }
    (results_root / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
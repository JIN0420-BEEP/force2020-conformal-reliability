from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from force2020_reliability.paths import RESULTS_DIR


def summarize_stage2(results_root: Path = RESULTS_DIR) -> dict[str, Path]:
    sensitivity_path = results_root / "sensitivity" / "fold_metrics.csv"
    nested_path = results_root / "nested" / "all_metrics.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    nested = pd.read_csv(nested_path)
    sensitivity_folds = sensitivity.loc[sensitivity["fold"] >= 0].copy()
    nested_folds = nested.loc[nested["fold"] >= 0].copy()
    nested_pooled = nested.loc[nested["fold"] < 0].copy()

    sensitivity_summary = (
        sensitivity_folds.groupby(["protocol", "spatial_scheme"], observed=True)
        .agg(
            n_runs=("fold", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            coverage90_optimistic_mean=("coverage_at_target", "mean"),
            coverage90_optimistic_std=("coverage_at_target", "std"),
        )
        .reset_index()
    )
    nested_summary = (
        nested_folds.groupby(["spatial_scheme", "protocol"], observed=True)
        .agg(
            n_folds=("fold", "size"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            calibrated_coverage_mean=("calibrated_coverage", "mean"),
            calibrated_coverage_min=("calibrated_coverage", "min"),
            calibrated_coverage_max=("calibrated_coverage", "max"),
            calibrated_selective_accuracy_mean=("calibrated_selective_accuracy", "mean"),
            calibrated_selective_accuracy_min=("calibrated_selective_accuracy", "min"),
            calibrated_selective_accuracy_max=("calibrated_selective_accuracy", "max"),
            calibrated_threshold_mean=("calibrated_threshold", "mean"),
            calibrated_threshold_std=("calibrated_threshold", "std"),
            n_target_met=("calibrated_selective_accuracy", lambda values: int((values >= 0.90).sum())),
        )
        .reset_index()
    )

    output_dir = results_root / "stage2_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sensitivity_summary": output_dir / "sensitivity_summary.csv",
        "nested_summary": output_dir / "nested_summary.csv",
        "nested_pooled": output_dir / "nested_pooled.csv",
    }
    sensitivity_summary.to_csv(paths["sensitivity_summary"], index=False, encoding="utf-8-sig")
    nested_summary.to_csv(paths["nested_summary"], index=False, encoding="utf-8-sig")
    nested_pooled.to_csv(paths["nested_pooled"], index=False, encoding="utf-8-sig")
    metadata = {
        "sensitivity_runs": len(sensitivity_folds),
        "nested_outer_folds": len(nested_folds),
        "target_accuracy": 0.90,
        "nested_targets_met": int((nested_folds["calibrated_selective_accuracy"] >= 0.90).sum()),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    metadata_path = output_dir / "summary.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"] = metadata_path
    return paths
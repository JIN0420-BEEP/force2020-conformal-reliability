from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from force2020_reliability.evaluation.segments import (
    evaluate_prediction_segments,
    smooth_predictions_min_run,
)
from force2020_reliability.paths import RESULTS_DIR

DEFAULT_MIN_RUNS_M = (1.0, 3.0, 5.0, 10.0)


def analyze_segment_smoothing(
    results_root: Path = RESULTS_DIR / "segments",
    min_runs_m: tuple[float, ...] = DEFAULT_MIN_RUNS_M,
    gap_threshold_m: float = 0.5,
    boundary_tolerance_m: float = 0.5,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results_root.glob("*/*/fold_*.parquet")):
        protocol = path.parent.parent.name
        model = path.parent.name
        fold = int(path.stem.removeprefix("fold_"))
        predictions = pd.read_parquet(path)
        for min_run_m in min_runs_m:
            smoothed = smooth_predictions_min_run(
                predictions,
                min_run_m=min_run_m,
                gap_threshold_m=gap_threshold_m,
            )
            evaluation = smoothed.copy()
            evaluation["y_pred"] = evaluation["y_pred_smooth"]
            evaluation = evaluation.drop(columns=["y_pred_smooth"])
            metrics = evaluate_prediction_segments(
                evaluation,
                gap_threshold_m=gap_threshold_m,
                boundary_tolerance_m=boundary_tolerance_m,
            )
            metrics = metrics.assign(
                protocol=protocol,
                model=model,
                fold=fold,
                min_run_m=min_run_m,
            )
            frames.append(metrics)
            print(
                f"smoothing/{protocol}/{model} fold={fold} min_run={min_run_m:.1f}m "
                f"boundary_f1={metrics['boundary_f1'].mean():.3f}",
                flush=True,
            )

    well_metrics = pd.concat(frames, ignore_index=True)
    well_metrics.to_csv(
        results_root / "smoothed_well_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        well_metrics.groupby(["protocol", "model", "min_run_m"], observed=True)
        .agg(
            n_wells=("well_id", "nunique"),
            point_accuracy_mean=("point_accuracy", "mean"),
            boundary_precision_mean=("boundary_precision", "mean"),
            boundary_recall_mean=("boundary_recall", "mean"),
            boundary_f1_mean=("boundary_f1", "mean"),
            boundary_f1_median=("boundary_f1", "median"),
            boundary_mae_m_mean=("boundary_mae_m", "mean"),
            thickness_mae_m_mean=("thickness_mae_m", "mean"),
            thickness_relative_error_mean=("thickness_relative_error", "mean"),
            segment_count_abs_error_mean=("segment_count_abs_error", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        results_root / "smoothed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "min_runs_m": list(min_runs_m),
        "gap_threshold_m": gap_threshold_m,
        "boundary_tolerance_m": boundary_tolerance_m,
        "prediction_files": len(list(results_root.glob("*/*/fold_*.parquet"))),
    }
    (results_root / "smoothing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
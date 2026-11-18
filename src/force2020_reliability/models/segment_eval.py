from __future__ import annotations

import json

import pandas as pd

from force2020_reliability.config import ExperimentConfig
from force2020_reliability.evaluation.segments import evaluate_prediction_segments
from force2020_reliability.models.baselines import make_model
from force2020_reliability.models.runner import filter_supported_classes
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

SEGMENT_MODELS = ("hist_gradient_boosting", "lightgbm")
SEGMENT_PROTOCOLS = ("well_grouped", "spatial_block")


def _test_wells_for_fold(protocol: str, fold: int, seed: int) -> set[str]:
    role_file = (
        "well_roles.csv" if protocol == "well_grouped" else "spatial_well_roles.csv"
    )
    roles = pd.read_csv(
        INTERIM_DIR
        / "splits"
        / "sensitivity"
        / f"seed_{seed}"
        / "blocks5_buffer20"
        / role_file
    )
    selected = roles.loc[
        (roles["evaluation_fold"] == fold) & (roles["role"] == "test")
    ]
    return set(selected["well_id"])


def run_segment_evaluation(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    gap_threshold_m: float = 0.5,
    boundary_tolerance_m: float = 0.5,
    max_folds: int | None = None,
) -> pd.DataFrame:
    results_root = RESULTS_DIR / "segments"
    results_root.mkdir(parents=True, exist_ok=True)
    split_dir = INTERIM_DIR / "splits" / "sensitivity" / f"seed_{config.seed}" / "blocks5_buffer20"
    sample = pd.read_parquet(RESULTS_DIR / "sensitivity" / "sensitivity_sample.parquet")
    full, _ = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    features = config.numeric_features
    target = config.target_column
    metric_frames: list[pd.DataFrame] = []
    n_folds = config.n_folds if max_folds is None else min(max_folds, config.n_folds)

    for protocol in SEGMENT_PROTOCOLS:
        for model_name in SEGMENT_MODELS:
            model_dir = results_root / protocol / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            for fold in range(n_folds):
                train_index, _ = resolve_fold_members(
                    protocol,
                    fold,
                    sample,
                    splits_dir=split_dir,
                )
                train = sample.iloc[train_index]
                test_wells = _test_wells_for_fold(protocol, fold, config.seed)
                test = full.loc[full["well_id"].isin(test_wells)].copy()
                if train.empty or test.empty:
                    raise ValueError(f"{protocol}/{model_name} fold {fold} has empty rows")
                model = make_model(model_name, config.seed + fold)
                model.fit(train[features], train[target].to_numpy(dtype=int))
                predictions = model.predict(test[features])
                prediction_frame = pd.DataFrame(
                    {
                        "well_id": test["well_id"].to_numpy(),
                        "row_id": test["row_id"].to_numpy(),
                        "DEPTH_MD": test["DEPTH_MD"].to_numpy(),
                        "y_true": test[target].to_numpy(dtype=int),
                        "y_pred": predictions.astype(int),
                    }
                )
                prediction_frame.to_parquet(
                    model_dir / f"fold_{fold}.parquet", index=False
                )
                well_metrics = evaluate_prediction_segments(
                    prediction_frame,
                    gap_threshold_m=gap_threshold_m,
                    boundary_tolerance_m=boundary_tolerance_m,
                )
                well_metrics = well_metrics.assign(
                    protocol=protocol,
                    model=model_name,
                    fold=fold,
                    train_rows=len(train),
                    train_wells=int(train["well_id"].nunique()),
                )
                metric_frames.append(well_metrics)
                print(
                    f"segments/{protocol}/{model_name} fold={fold} "
                    f"boundary_f1={well_metrics['boundary_f1'].mean():.3f} "
                    f"thickness_mae={well_metrics['thickness_mae_m'].mean():.2f}m",
                    flush=True,
                )

    all_well_metrics = pd.concat(metric_frames, ignore_index=True)
    all_well_metrics.to_csv(
        results_root / "well_segment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        all_well_metrics.groupby(["protocol", "model"], observed=True)
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
    summary.to_csv(results_root / "summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "models": list(SEGMENT_MODELS),
        "protocols": list(SEGMENT_PROTOCOLS),
        "gap_threshold_m": gap_threshold_m,
        "boundary_tolerance_m": boundary_tolerance_m,
        "training_rows_per_fold": int(all_well_metrics["train_rows"].max()),
        "predicted_row_records": int(all_well_metrics["n_rows"].sum()),
        "n_test_wells": int(all_well_metrics["well_id"].nunique()),
        "evaluated_folds": n_folds,
    }
    (results_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
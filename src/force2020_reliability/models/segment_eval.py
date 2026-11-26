from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from force2020_reliability.config import ExperimentConfig, load_config
from force2020_reliability.evaluation.segments import evaluate_prediction_segments
from force2020_reliability.models.baselines import make_model
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

SEGMENT_MODELS = ("hist_gradient_boosting", "lightgbm")
SEGMENT_PROTOCOLS = ("well_grouped", "spatial_block")
DATA = INTERIM_DIR / "force2020_labeled_strat.parquet"


def _test_wells_for_fold(protocol: str, fold: int, split_dir) -> set[str]:
    role_file = (
        "well_roles.csv" if protocol == "well_grouped" else "spatial_well_roles.csv"
    )
    roles = pd.read_csv(split_dir / role_file)
    selected = roles.loc[
        (roles["evaluation_fold"] == fold) & (roles["role"] == "test")
    ]
    return set(selected["well_id"])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def run_segment_evaluation(
    sample: pd.DataFrame,
    full: pd.DataFrame,
    config: ExperimentConfig,
    model_names=SEGMENT_MODELS,
    protocols=SEGMENT_PROTOCOLS,
    output_name: str = "segments_multiseed",
    gap_threshold_m: float = 0.5,
    boundary_tolerance_m: float = 0.5,
    max_folds: int | None = None,
    hgb_early_stopping: bool = False,
) -> pd.DataFrame:
    """Evaluate point-, boundary-, thickness- and stability-level metrics.

    ``sample`` supplies the capped training/calibration rows used by the fold
    resolver; ``full`` supplies every depth point of the held-out test wells so
    that continuous well sections are reconstructed. Metrics are computed per
    well and aggregated both as equal-well means and point/pair-weighted pools.
    """
    results_root = RESULTS_DIR / output_name
    results_root.mkdir(parents=True, exist_ok=True)
    split_dir = (
        INTERIM_DIR
        / "splits"
        / "sensitivity"
        / f"seed_{config.seed}"
        / "blocks5_buffer20"
    )
    features = config.numeric_features
    target = config.target_column
    metric_frames: list[pd.DataFrame] = []
    n_folds = config.n_folds if max_folds is None else min(max_folds, config.n_folds)

    for protocol in protocols:
        for model_name in model_names:
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
                test_wells = _test_wells_for_fold(protocol, fold, split_dir)
                test = full.loc[full["well_id"].isin(test_wells)].copy()
                if train.empty or test.empty:
                    raise ValueError(
                        f"{protocol}/{model_name} fold {fold} has empty rows"
                    )
                model = make_model(
                    model_name,
                    config.seed + fold,
                    hgb_early_stopping=hgb_early_stopping,
                )
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
                    seed=config.seed,
                    train_rows=len(train),
                    train_wells=int(train["well_id"].nunique()),
                )
                metric_frames.append(well_metrics)
                print(
                    f"segments/{protocol}/{model_name} fold={fold} "
                    f"boundary_f1={well_metrics['boundary_f1'].mean():.3f} "
                    f"adj_agree={well_metrics['pred_adjacent_agreement'].mean():.3f} "
                    f"thickness_rel={well_metrics['thickness_relative_error'].mean():.3f}",
                    flush=True,
                )

    all_well_metrics = pd.concat(metric_frames, ignore_index=True)
    all_well_metrics.to_csv(
        results_root / "well_segment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Per-seed summary: equal-well means plus point/pair-weighted pools.
    seed_rows = []
    for (protocol, model_name, seed), group in all_well_metrics.groupby(
        ["protocol", "model", "seed"], observed=True
    ):
        seed_rows.append(
            {
                "protocol": protocol,
                "model": model_name,
                "seed": seed,
                "n_well_rows": len(group),
                "n_unique_wells": group["well_id"].nunique(),
                "point_accuracy_equal_well": group["point_accuracy"].mean(),
                "point_accuracy_pooled": _weighted_mean(
                    group["point_accuracy"], group["n_rows"]
                ),
                "pred_adjacent_agreement_equal_well": group[
                    "pred_adjacent_agreement"
                ].mean(),
                "pred_adjacent_agreement_pooled": _weighted_mean(
                    group["pred_adjacent_agreement"], group["adjacent_pairs"]
                ),
                "true_adjacent_agreement_equal_well": group[
                    "true_adjacent_agreement"
                ].mean(),
                "pred_flip_rate_equal_well": group["pred_flip_rate"].mean(),
                "boundary_precision_mean": group["boundary_precision"].mean(),
                "boundary_recall_mean": group["boundary_recall"].mean(),
                "boundary_f1_mean": group["boundary_f1"].mean(),
                "boundary_f1_median": group["boundary_f1"].median(),
                "boundary_mae_m_mean": group["boundary_mae_m"].mean(),
                "thickness_mae_m_mean": group["thickness_mae_m"].mean(),
                "thickness_relative_error_mean": group[
                    "thickness_relative_error"
                ].mean(),
                "segment_count_abs_error_mean": group[
                    "segment_count_abs_error"
                ].mean(),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(
        results_root / "summary_by_seed.csv", index=False, encoding="utf-8-sig"
    )

    # Across-seed mean and between-seed SD (seed is the independent repeat;
    # wells/folds within a seed are not independent geographic replicates).
    metric_columns = [
        c
        for c in seed_summary.columns
        if c not in ("protocol", "model", "seed", "n_well_rows", "n_unique_wells")
    ]
    summary = (
        seed_summary.groupby(["protocol", "model"], observed=True)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = seed_summary.groupby(
        ["protocol", "model"], observed=True
    )["seed"].nunique().to_numpy()
    summary.to_csv(results_root / "summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "models": list(model_names),
        "protocols": list(protocols),
        "seeds": [int(s) for s in sorted(all_well_metrics["seed"].unique())],
        "gap_threshold_m": gap_threshold_m,
        "boundary_tolerance_m": boundary_tolerance_m,
        "hgb_early_stopping": hgb_early_stopping,
        "training_rows_per_fold_max": int(all_well_metrics["train_rows"].max()),
        "predicted_row_records": int(all_well_metrics["n_rows"].sum()),
        "n_unique_test_wells": int(all_well_metrics["well_id"].nunique()),
        "evaluated_folds_per_seed": n_folds,
        "aggregation_note": (
            "equal_well = unweighted mean over well-fold rows; pooled = "
            "point/depth-pair weighted; across-seed SD uses seed summaries."
        ),
    }
    (results_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rows-per-well", type=int, default=1000)
    parser.add_argument("--output", default="segments_multiseed")
    parser.add_argument("--models", nargs="+", default=list(SEGMENT_MODELS))
    parser.add_argument("--protocols", nargs="+", default=list(SEGMENT_PROTOCOLS))
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--hgb-early-stopping",
        action="store_true",
        help="Enable HGB internal early stopping (off by default to match the "
        "repeated well-level experiment).",
    )
    args = parser.parse_args()

    config = load_config()
    if args.seed is not None:
        config = replace(config, raw={**config.raw, "seed": args.seed})

    frame = pd.read_parquet(DATA)
    filtered, _ = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    sample = sample_pilot_rows(filtered, args.rows_per_well, config.seed)
    print(
        f"sampled rows: {len(sample)} wells: {sample['well_id'].nunique()} "
        f"seed={config.seed}"
    )
    summary = run_segment_evaluation(
        sample=sample,
        full=filtered,
        config=config,
        model_names=args.models,
        protocols=args.protocols,
        output_name=args.output,
        max_folds=args.max_folds,
        hgb_early_stopping=args.hgb_early_stopping,
    )
    print(summary.to_string(index=False))
    print("saved to", RESULTS_DIR / args.output)


if __name__ == "__main__":
    main()

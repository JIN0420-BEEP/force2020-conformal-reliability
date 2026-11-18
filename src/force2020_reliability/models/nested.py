from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from force2020_reliability.evaluation.metrics import evaluate_predictions, risk_coverage_curve
from force2020_reliability.models.baselines import make_model
from force2020_reliability.models.runner import (
    PROTOCOLS,
    _prediction_probabilities,
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members


def select_selective_threshold(y_true, y_pred, confidence, target_accuracy=0.90):
    curve = risk_coverage_curve(y_true, y_pred, confidence, coverage_points=1000)
    eligible = curve.loc[curve["selective_accuracy"] >= target_accuracy]
    if eligible.empty:
        return float(np.nextafter(confidence.max(), np.inf)), curve
    return float(eligible.sort_values("coverage").iloc[-1]["threshold"]), curve


def _align(probabilities, classes, global_classes):
    aligned = np.zeros((len(probabilities), len(global_classes)), dtype=float)
    aligned[:, np.searchsorted(global_classes, classes)] = probabilities
    return aligned


def run_nested_protocol(protocol, frame, config, splits_dir, output_dir, target_accuracy=0.90):
    model_dir = output_dir / protocol / "hist_gradient_boosting_nested"
    model_dir.mkdir(parents=True, exist_ok=True)
    features = config.numeric_features
    target = config.target_column
    global_classes = np.sort(frame[target].unique()).astype(int)
    fold_rows, curves, prediction_frames = [], [], []

    for fold in range(config.n_folds):
        train_index, test_index = resolve_fold_members(protocol, fold, frame, splits_dir)
        train, test = frame.iloc[train_index].copy(), frame.iloc[test_index].copy()
        wells = train["well_id"].unique()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed + fold)
        fit_index, calibration_index = next(splitter.split(wells, groups=wells))
        fit_wells, calibration_wells = set(wells[fit_index]), set(wells[calibration_index])
        fit_rows = train.loc[train["well_id"].isin(fit_wells)]
        calibration_rows = train.loc[train["well_id"].isin(calibration_wells)]
        if fit_rows.empty or calibration_rows.empty:
            raise ValueError(f"{protocol} fold {fold} nested split is empty")

        calibration_model = make_model("hist_gradient_boosting", config.seed + fold)
        calibration_model.fit(fit_rows[features], fit_rows[target].to_numpy(dtype=int))
        calibration_probabilities, calibration_classes = _prediction_probabilities(
            calibration_model, calibration_rows[features]
        )
        calibration_probabilities = _align(
            calibration_probabilities, calibration_classes, global_classes
        )
        calibration_predictions = global_classes[np.argmax(calibration_probabilities, axis=1)]
        threshold, calibration_curve = select_selective_threshold(
            calibration_rows[target].to_numpy(dtype=int),
            calibration_predictions,
            calibration_probabilities.max(axis=1),
            target_accuracy,
        )

        final_model = make_model("hist_gradient_boosting", config.seed + fold)
        final_model.fit(train[features], train[target].to_numpy(dtype=int))
        probabilities, classes = _prediction_probabilities(final_model, test[features])
        aligned_probabilities = _align(probabilities, classes, global_classes)
        predictions = global_classes[np.argmax(aligned_probabilities, axis=1)]
        metrics, _, _, prediction_frame = evaluate_predictions(
            test[target].to_numpy(dtype=int),
            predictions,
            aligned_probabilities,
            global_classes,
            test["well_id"].to_numpy(),
        )
        selected = prediction_frame["confidence"].to_numpy() >= threshold
        coverage = float(selected.mean())
        selective_accuracy = (
            float(prediction_frame.loc[selected, "correct"].mean())
            if selected.any()
            else float("nan")
        )
        metrics.update(
            {
                "protocol": protocol,
                "model": "hist_gradient_boosting_nested",
                "fold": fold,
                "target_accuracy": target_accuracy,
                "calibrated_threshold": threshold,
                "calibrated_coverage": coverage,
                "calibrated_selective_accuracy": selective_accuracy,
                "calibrated_risk": 1.0 - selective_accuracy,
                "calibration_fit_rows": len(fit_rows),
                "calibration_rows": len(calibration_rows),
                "calibration_fit_wells": len(fit_wells),
                "calibration_wells": len(calibration_wells),
            }
        )
        fold_rows.append(metrics)
        curves.append(calibration_curve.assign(protocol=protocol, fold=fold))
        prediction_frame = prediction_frame.assign(
            protocol=protocol,
            fold=fold,
            threshold=threshold,
            selected=selected,
            row_id=test["row_id"].to_numpy(),
            strat_group=test["strat_group"].to_numpy(),
            distance_to_nearest_train_m=0.0,
        )
        prediction_frames.append(prediction_frame)
        print(
            f"nested/{protocol} fold={fold} coverage={coverage:.3f} "
            f"selective_acc={selective_accuracy:.3f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    pooled = pd.concat(prediction_frames, ignore_index=True)
    pooled_row = {
        "protocol": protocol,
        "model": "hist_gradient_boosting_nested",
        "fold": -1,
        "target_accuracy": target_accuracy,
        "calibrated_threshold": float(fold_metrics["calibrated_threshold"].mean()),
        "calibrated_coverage": float(pooled["selected"].mean()),
        "calibrated_selective_accuracy": float(pooled.loc[pooled["selected"], "correct"].mean()),
    }
    pooled_row["calibrated_risk"] = 1.0 - pooled_row["calibrated_selective_accuracy"]
    all_metrics = pd.concat([fold_metrics, pd.DataFrame([pooled_row])], ignore_index=True)
    all_metrics.to_csv(model_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves, ignore_index=True).to_csv(
        model_dir / "calibration_curves.csv", index=False, encoding="utf-8-sig"
    )
    pooled.to_parquet(model_dir / "predictions.parquet", index=False)
    (model_dir / "summary.json").write_text(
        json.dumps(pooled_row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_metrics


def run_nested_calibration(frame, config, rows_per_well=1000, target_accuracy=0.90):
    results_root = RESULTS_DIR / "nested"
    results_root.mkdir(parents=True, exist_ok=True)
    filtered, _ = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    sampled = sample_pilot_rows(filtered, rows_per_well, config.seed)
    sampled.to_parquet(results_root / "nested_sample.parquet", index=False)
    primary = INTERIM_DIR / "splits/sensitivity" / f"seed_{config.seed}" / "blocks5_buffer20"
    secondary = INTERIM_DIR / "splits/sensitivity" / f"seed_{config.seed}" / "blocks6_buffer30"
    if not primary.exists() or not secondary.exists():
        raise FileNotFoundError("Run `force2020 sensitivity` first.")

    frames = []
    for protocol in PROTOCOLS:
        result = run_nested_protocol(
            protocol, sampled, config, primary, results_root / "blocks5_buffer20", target_accuracy
        )
        result["spatial_scheme"] = "blocks5_buffer20"
        frames.append(result)
    result = run_nested_protocol(
        "spatial_block",
        sampled,
        config,
        secondary,
        results_root / "blocks6_buffer30",
        target_accuracy,
    )
    result["spatial_scheme"] = "blocks6_buffer30"
    frames.append(result)
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(results_root / "all_metrics.csv", index=False, encoding="utf-8-sig")
    return all_metrics
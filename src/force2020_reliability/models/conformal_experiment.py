"""Conformal prediction experiments across the three validation protocols."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from force2020_reliability.evaluation import conformal as cp
from force2020_reliability.evaluation.calibration import TemperatureScaler
from force2020_reliability.evaluation.metrics import expected_calibration_error
from force2020_reliability.models.baselines import make_model
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")
ALPHA = 0.10
CALIBRATION_FRACTION = 0.25


def _align(probabilities, classes, global_classes):
    aligned = np.zeros((len(probabilities), len(global_classes)), dtype=float)
    aligned[:, np.searchsorted(global_classes, classes)] = probabilities
    return aligned


def _method_row(name, membership, y_test, classes, probabilities):
    diag = cp.prediction_set_diagnostics(membership, y_test, classes)
    point_pred = classes[np.argmax(probabilities, axis=1)]
    return {
        "method": name,
        "coverage": diag["coverage"],
        "mean_set_size": diag["mean_set_size"],
        "median_set_size": diag["median_set_size"],
        "max_set_size": diag["max_set_size"],
        "point_accuracy": float((point_pred == y_test).mean()),
        "ece": expected_calibration_error(
            y_test, point_pred, probabilities.max(axis=1)),
        "per_class_coverage": json.dumps(
            {str(k): round(v, 4) for k, v in diag["per_class_coverage"].items()},
            ensure_ascii=False),
    }


def run_conformal_fold(protocol, fold, frame, config, splits_dir, seed):
    rng = np.random.default_rng(seed + fold)
    features, target = config.numeric_features, config.target_column
    classes = np.sort(frame[target].unique()).astype(int)

    train_pos, test_pos = resolve_fold_members(protocol, fold, frame, splits_dir)
    train, test = frame.iloc[train_pos], frame.iloc[test_pos]

    wells = train["well_id"].unique()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=CALIBRATION_FRACTION, random_state=seed + fold)
    fit_sel, cal_sel = next(splitter.split(wells, groups=wells))
    fit_wells, cal_wells = set(wells[fit_sel]), set(wells[cal_sel])
    fit_rows = train.loc[train["well_id"].isin(fit_wells)]
    cal_rows = train.loc[train["well_id"].isin(cal_wells)]

    model = make_model("hist_gradient_boosting", seed + fold)
    model.fit(fit_rows[features], fit_rows[target].to_numpy(dtype=int))

    p_cal_raw, cal_model_classes = model.predict_proba(cal_rows[features]), np.asarray(model.classes_)
    p_test_raw, test_model_classes = model.predict_proba(test[features]), np.asarray(model.classes_)
    p_cal = _align(p_cal_raw, cal_model_classes, classes)
    p_test = _align(p_test_raw, test_model_classes, classes)
    y_cal = cal_rows[target].to_numpy(dtype=int)
    y_test = test[target].to_numpy(dtype=int)

    # Randomized uniform values shared across methods for fair comparison.
    u_cal, u_test = rng.random(len(y_cal)), rng.random(len(y_test))
    scores = cp.aps_calibration_scores(p_cal, y_cal, classes, u_cal)
    q_global = cp.conformal_quantile(scores, ALPHA)

    rows = []
    # 1. Marginal APS
    rows.append(_method_row(
        "aps", cp.aps_prediction_sets(p_test, classes, q_global, u_test),
        y_test, classes, p_test))
    # 2. Mondrian (class-conditional); classes absent from calibration wells
    # fall back to the marginal quantile.
    q_mondrian = cp.mondrian_quantiles(scores, y_cal, classes, ALPHA)
    q_arr = np.array([q_mondrian.get(int(v), q_global) for v in y_test])
    rows.append(_method_row(
        "mondrian", cp.aps_prediction_sets(p_test, classes, q_arr, u_test),
        y_test, classes, p_test))
    # 3. RAPS regularization
    rows.append(_method_row(
        "raps", cp.aps_prediction_sets(
            p_test, classes, q_global, u_test, raps_penalty=0.02,
            raps_target=max(1, int(1.0 / ALPHA))),
        y_test, classes, p_test))
    # 4. Temperature scaling then APS
    scaler = TemperatureScaler().fit(p_cal, y_cal, classes)
    p_cal_t, p_test_t = scaler.transform(p_cal), scaler.transform(p_test)
    scores_t = cp.aps_calibration_scores(p_cal_t, y_cal, classes, u_cal)
    q_t = cp.conformal_quantile(scores_t, ALPHA)
    rows.append(_method_row(
        "temp_aps", cp.aps_prediction_sets(p_test_t, classes, q_t, u_test),
        y_test, classes, p_test_t))

    for row in rows:
        row.update({"protocol": protocol, "fold": fold,
                    "alpha": ALPHA, "n_cal": len(y_cal), "n_test": len(y_test)})
    return rows


def run_conformal_experiment(
    frame,
    config,
    rows_per_well=1000,
    output_name="conformal",
    splits_subdir=("blocks5_buffer20",),
):
    output_dir = RESULTS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered, _ = filter_supported_classes(
        frame, target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells)
    sampled = sample_pilot_rows(filtered, rows_per_well, config.seed)

    all_rows = []
    for subdir in splits_subdir:
        splits_dir = INTERIM_DIR / "splits" / "sensitivity" / f"seed_{config.seed}" / subdir
        for protocol in PROTOCOLS:
            for fold in range(config.n_folds):
                rows = run_conformal_fold(
                    protocol, fold, sampled, config, splits_dir, config.seed)
                all_rows.extend(rows)
                for row in rows:
                    row["spatial_scheme"] = subdir
                print(f"conformal/{subdir}/{protocol} fold={fold} "
                      f"coverage={rows[0]['coverage']:.3f} "
                      f"size={rows[0]['mean_set_size']:.3f}", flush=True)

    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        results.groupby(["protocol", "method"], observed=True)
        .agg(coverage_mean=("coverage", "mean"),
             coverage_sd=("coverage", "std"),
             set_size_mean=("mean_set_size", "mean"),
             point_accuracy_mean=("point_accuracy", "mean"),
             ece_mean=("ece", "mean"))
        .reset_index())
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return summary

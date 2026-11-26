"""Well-level split-conformal experiment across models and validation protocols.

Unlike ``conformal_experiment``, which only records fold-level aggregates, this
driver retains one row per test well so that uncertainty is quantified at the
independent sampling unit (the well) rather than at depth points or folds that
share wells across seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from force2020_reliability.evaluation import conformal as cp
from force2020_reliability.models.baselines import make_model
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")
CALIBRATION_FRACTION = 0.25
ALPHA = 0.10

# Position/depth columns separated from the wireline-curve inputs for the
# feature-source ablation. Names follow the project configuration.
POSITION_COLUMNS = ("DEPTH_MD", "x_loc", "y_loc", "z_loc")
FEATURE_SETS = ("full", "logs_only", "logs_depth", "logs_coords")


def feature_columns_for_set(all_features, feature_set: str = "full"):
    """Return the columns for a named ablation group.

    logs_only   : wireline curves (14)
    logs_depth  : curves + measured depth DEPTH_MD (15)
    logs_coords : curves + x/y/z location, incl. vertical location (17)
    full        : all configured features (18)
    """
    all_features = list(all_features)
    if feature_set in (None, "full"):
        return all_features
    logs = [c for c in all_features if c not in POSITION_COLUMNS]
    if feature_set == "logs_only":
        return logs
    if feature_set == "logs_depth":
        return logs + ["DEPTH_MD"]
    if feature_set == "logs_coords":
        return logs + ["x_loc", "y_loc", "z_loc"]
    raise ValueError(
        f"Unknown feature_set={feature_set!r}; choose from {FEATURE_SETS}")


def _align(probabilities, classes, global_classes):
    aligned = np.zeros((len(probabilities), len(global_classes)), dtype=float)
    aligned[:, np.searchsorted(global_classes, classes)] = probabilities
    return aligned


def run_well_fold(model_name, protocol, fold, frame, config, splits_dir, seed,
                  feature_set="full"):
    rng = np.random.default_rng(seed + fold)
    features = feature_columns_for_set(config.numeric_features, feature_set)
    target = config.target_column
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

    # Fixed HGB budget (no internal early stopping) for robustness to
    # rare-class singleton validation splits; see baselines.make_model.
    model = make_model(model_name, seed + fold, hgb_early_stopping=False)
    model.fit(fit_rows[features], fit_rows[target].to_numpy(dtype=int))

    p_cal = _align(model.predict_proba(cal_rows[features]),
                   np.asarray(model.classes_), classes)
    p_test = _align(model.predict_proba(test[features]),
                    np.asarray(model.classes_), classes)
    y_cal = cal_rows[target].to_numpy(dtype=int)
    y_test = test[target].to_numpy(dtype=int)
    well_test = test["well_id"].to_numpy()

    u_cal, u_test = rng.random(len(y_cal)), rng.random(len(y_test))
    scores = cp.aps_calibration_scores(p_cal, y_cal, classes, u_cal)
    quantile = cp.conformal_quantile(scores, ALPHA)
    membership = cp.aps_prediction_sets(p_test, classes, quantile, u_test)

    class_idx = cp.true_class_index(y_test, classes)
    covered = membership[np.arange(len(y_test)), class_idx]
    sizes = membership.sum(axis=1)
    point_pred = classes[np.argmax(p_test, axis=1)]
    point_correct = point_pred == y_test
    macro_f1 = f1_score(
        y_test, point_pred, labels=classes, average="macro",
        zero_division=0)

    # ---- well-level aggregation (the independent unit) ----
    well_frame = pd.DataFrame({
        "well_id": well_test,
        "covered": covered.astype(int),
        "point_correct": point_correct.astype(int),
        "set_size": sizes,
    })
    grouped = well_frame.groupby("well_id", observed=True)
    well_metrics = grouped.agg(
        coverage=("covered", "mean"),
        point_accuracy=("point_correct", "mean"),
        mean_set_size=("set_size", "mean"),
        n_points=("covered", "size"),
    ).reset_index()
    well_metrics["model"] = model_name
    well_metrics["protocol"] = protocol
    well_metrics["fold"] = fold
    well_metrics["feature_set"] = feature_set

    fold_row = {
        "model": model_name,
        "protocol": protocol,
        "fold": fold,
        "feature_set": feature_set,
        "n_features": len(features),
        "coverage": float(covered.mean()),
        "point_accuracy": float(point_correct.mean()),
        "macro_f1": float(macro_f1),
        "mean_set_size": float(sizes.mean()),
        "n_cal": len(y_cal),
        "n_test": len(y_test),
        "n_test_wells": int(pd.unique(well_test).size),
    }
    return well_metrics, fold_row


def run_well_experiment(frame, config, model_names, schemes, output_name,
                        seed=None, protocols=PROTOCOLS, feature_set="full"):
    seed = config.seed if seed is None else seed
    output_dir = RESULTS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    well_frames, fold_rows = [], []
    for scheme in schemes:
        splits_dir = (
            INTERIM_DIR / "splits" / "sensitivity" / f"seed_{seed}" / scheme)
        for model_name in model_names:
            for protocol in protocols:
                for fold in range(config.n_folds):
                    well_metrics, fold_row = run_well_fold(
                        model_name=model_name, protocol=protocol, fold=fold,
                        frame=frame, config=config, splits_dir=splits_dir,
                        seed=seed, feature_set=feature_set)
                    well_metrics["scheme"] = scheme
                    well_metrics["seed"] = seed
                    fold_row["scheme"] = scheme
                    fold_row["seed"] = seed
                    well_frames.append(well_metrics)
                    fold_rows.append(fold_row)
                    print(
                        f"{scheme}/{model_name}/{protocol} fold={fold} "
                        f"well_cov={fold_row['coverage']:.3f} "
                        f"well_acc={fold_row['point_accuracy']:.3f}",
                        flush=True)

    well_table = pd.concat(well_frames, ignore_index=True)
    fold_table = pd.DataFrame(fold_rows)
    well_table.to_csv(
        output_dir / "well_metrics.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(
        output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    return output_dir

"""Boundary-localised uncertainty: prediction-set size vs distance to a true boundary."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from force2020_reliability.config import load_config
from force2020_reliability.evaluation import conformal as cp
from force2020_reliability.models.baselines import make_model
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import FIGURES_DIR, INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

ALPHA = 0.10
BINS_M = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)
BIN_LABELS = ("<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8")


def _distance_to_boundary(well: pd.DataFrame, target: str) -> np.ndarray:
    ordered = well.sort_values("DEPTH_MD")
    depths = ordered["DEPTH_MD"].to_numpy(float)
    labels = ordered[target].to_numpy()
    change = np.flatnonzero(labels[1:] != labels[:-1])
    boundary_depths = 0.5 * (depths[change] + depths[change + 1])
    if len(boundary_depths) == 0:
        return np.full(len(well), np.inf)
    distance = np.min(
        np.abs(depths[:, None] - boundary_depths[None, :]), axis=1)
    result = np.empty(len(well))
    result[ordered.index.get_indexer(well.index)] = distance
    return result


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
    features, target = config.numeric_features, config.target_column
    classes = np.sort(sampled[target].unique()).astype(int)

    point_rows = []
    for fold in range(config.n_folds):
        rng = np.random.default_rng(config.seed + fold)
        train_pos, test_pos = resolve_fold_members(
            "well_grouped", fold, sampled, splits_dir)
        train, test = sampled.iloc[train_pos], sampled.iloc[test_pos]
        wells = train["well_id"].unique()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=config.seed + fold)
        fit_sel, cal_sel = next(splitter.split(wells, groups=wells))
        fit_rows = train.loc[train["well_id"].isin(set(wells[fit_sel]))]
        cal_rows = train.loc[train["well_id"].isin(set(wells[cal_sel]))]

        model = make_model("hist_gradient_boosting", config.seed + fold)
        model.fit(fit_rows[features], fit_rows[target].to_numpy(int))
        p_cal = model.predict_proba(cal_rows[features])
        p_test = model.predict_proba(test[features])
        y_cal = cal_rows[target].to_numpy(int)
        y_test = test[target].to_numpy(int)
        scores = cp.aps_calibration_scores(
            p_cal, y_cal, np.asarray(model.classes_), rng.random(len(y_cal)))
        quantile = cp.conformal_quantile(scores, ALPHA)
        membership = cp.aps_prediction_sets(
            p_test, np.asarray(model.classes_), quantile, rng.random(len(y_test)))
        set_size = membership.sum(axis=1)
        point_pred = np.asarray(model.classes_)[np.argmax(p_test, axis=1)]

        boundary_distance = np.concatenate([
            _distance_to_boundary(well, target)
            for _, well in test.groupby("well_id", observed=True)])
        point_rows.append(pd.DataFrame({
            "set_size": set_size,
            "correct": point_pred == y_test,
            "boundary_distance_m": boundary_distance,
        }))

    points = pd.concat(point_rows, ignore_index=True)
    points["bin"] = pd.cut(
        points["boundary_distance_m"], BINS_M, labels=BIN_LABELS, right=False)
    binned = points.groupby("bin", observed=True).agg(
        mean_set_size=("set_size", "mean"),
        error_rate=("correct", lambda s: 1 - s.mean()),
        n=("set_size", "size")).reset_index()
    binned.to_csv(
        RESULTS_DIR / "conformal" / "boundary_uncertainty.csv",
        index=False, encoding="utf-8-sig")
    print(binned.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(binned))
    ax1.bar(x, binned["mean_set_size"], width=0.6, color="#0072B2", alpha=0.8,
            label="Mean prediction-set size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(binned["bin"])
    ax1.set_xlabel("Distance to true lithology boundary (m)")
    ax1.set_ylabel("Mean prediction-set size", color="#0072B2")
    ax2 = ax1.twinx()
    ax2.plot(x, binned["error_rate"], "o-", color="#D55E00", linewidth=1.6,
             label="Point error rate")
    ax2.set_ylabel("Point error rate", color="#D55E00")
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    output = FIGURES_DIR / "14_boundary_uncertainty.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print("saved", output)


if __name__ == "__main__":
    main()

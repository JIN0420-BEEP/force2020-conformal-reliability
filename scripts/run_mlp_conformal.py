"""Lightweight MLP conformal check: coverage degradation is model-agnostic."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from force2020_reliability.config import load_config
from force2020_reliability.evaluation import conformal as cp
from force2020_reliability.models.runner import (
    filter_supported_classes,
    sample_pilot_rows,
)
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")
ALPHA = 0.10


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

    rows = []
    for protocol in PROTOCOLS:
        for fold in range(config.n_folds):
            rng = np.random.default_rng(config.seed + fold)
            train_pos, test_pos = resolve_fold_members(
                protocol, fold, sampled, splits_dir)
            train, test = sampled.iloc[train_pos], sampled.iloc[test_pos]
            wells = train["well_id"].unique()
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.25, random_state=config.seed + fold)
            fit_sel, cal_sel = next(splitter.split(wells, groups=wells))
            fit_wells, cal_wells = set(wells[fit_sel]), set(wells[cal_sel])
            fit_rows = train.loc[train["well_id"].isin(fit_wells)]
            cal_rows = train.loc[train["well_id"].isin(cal_wells)]

            model = Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=200,
                    early_stopping=True, random_state=config.seed + fold)),
            ])
            model.fit(fit_rows[features], fit_rows[target].to_numpy(int))
            p_cal = model.predict_proba(cal_rows[features])
            p_test = model.predict_proba(test[features])
            model_classes = np.asarray(model.classes_)
            aligned_cal = np.zeros((len(p_cal), len(classes)))
            aligned_test = np.zeros((len(p_test), len(classes)))
            aligned_cal[:, np.searchsorted(classes, model_classes)] = p_cal
            aligned_test[:, np.searchsorted(classes, model_classes)] = p_test
            y_cal = cal_rows[target].to_numpy(int)
            y_test = test[target].to_numpy(int)

            scores = cp.aps_calibration_scores(
                aligned_cal, y_cal, classes, rng.random(len(y_cal)))
            quantile = cp.conformal_quantile(scores, ALPHA)
            membership = cp.aps_prediction_sets(
                aligned_test, classes, quantile, rng.random(len(y_test)))
            diag = cp.prediction_set_diagnostics(membership, y_test, classes)
            rows.append({"protocol": protocol, "fold": fold,
                         "coverage": diag["coverage"],
                         "set_size": diag["mean_set_size"],
                         "point_accuracy": float(
                             (model.predict(test[features]) == y_test).mean())})
            print(f"mlp/{protocol} fold={fold} coverage={diag['coverage']:.3f}",
                  flush=True)

    output = RESULTS_DIR / "conformal" / "mlp_coverage.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows).groupby("protocol").mean(numeric_only=True)
    print(summary.to_string())
    print("saved", output)


if __name__ == "__main__":
    main()

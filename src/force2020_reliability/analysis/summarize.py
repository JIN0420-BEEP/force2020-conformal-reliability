from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from force2020_reliability.labels import attach_lithology_names
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR

PROTOCOL_ORDER = ["random_point", "well_grouped", "spatial_block"]
MODEL_ORDER = ["hist_gradient_boosting", "random_forest"]
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "multiclass_brier",
    "ece_top_label",
    "aurc",
    "coverage_at_target",
]


def pilot_analysis_dir(result_dir: Path) -> Path:
    destination = result_dir / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def collect_metrics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(result_dir.glob("*/*/metrics.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv files under {result_dir}")
    metrics = pd.concat(frames, ignore_index=True)
    pooled = metrics.loc[metrics["fold"] == -1].copy()
    folds = metrics.loc[metrics["fold"] >= 0].copy()

    fold_summary = (
        folds.groupby(["model", "protocol"], observed=True)
        .agg(
            n_folds=("fold", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            ece_mean=("ece_top_label", "mean"),
            coverage90_mean=("coverage_at_target", "mean"),
            coverage90_std=("coverage_at_target", "std"),
        )
        .reset_index()
    )
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    protocol_order = {protocol: index for index, protocol in enumerate(PROTOCOL_ORDER)}
    fold_summary["_model"] = fold_summary["model"].map(order).fillna(99)
    fold_summary["_protocol"] = fold_summary["protocol"].map(protocol_order).fillna(99)
    fold_summary = fold_summary.sort_values(["_model", "_protocol"]).drop(
        columns=["_model", "_protocol"]
    )
    return folds, pooled, fold_summary


def _balanced_accuracy_from_predictions(group: pd.DataFrame) -> float:
    labels = np.sort(
        np.unique(
            np.concatenate(
                [
                    group["y_true"].to_numpy(dtype=int),
                    group["y_pred"].to_numpy(dtype=int),
                ]
            )
        )
    )
    matrix = confusion_matrix(
        group["y_true"].to_numpy(dtype=int),
        group["y_pred"].to_numpy(dtype=int),
        labels=labels,
    )
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros_like(support, dtype=float),
        where=support > 0,
    )
    return float(np.mean(recalls[support > 0]))


def bootstrap_by_well(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    labels = np.sort(
        np.unique(
            np.concatenate(
                [
                    predictions["y_true"].to_numpy(dtype=int),
                    predictions["y_pred"].to_numpy(dtype=int),
                ]
            )
        )
    )
    class_index = {label: index for index, label in enumerate(labels)}
    well_matrices: list[np.ndarray] = []
    well_ids = sorted(predictions["well_id"].unique())
    for well_id in well_ids:
        well = predictions.loc[predictions["well_id"] == well_id]
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for truth, prediction in zip(well["y_true"], well["y_pred"], strict=True):
            matrix[class_index[int(truth)], class_index[int(prediction)]] += 1
        well_matrices.append(matrix)
    matrices = np.stack(well_matrices)

    rng = np.random.default_rng(seed)
    samples = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(n_bootstrap):
        selection = rng.integers(0, len(well_ids), size=len(well_ids))
        aggregate = matrices[selection].sum(axis=0).astype(float)
        tp = np.diag(aggregate)
        support = aggregate.sum(axis=1)
        predicted = aggregate.sum(axis=0)
        total = aggregate.sum()
        accuracy = tp.sum() / total if total > 0 else np.nan
        present_recall = support > 0
        balanced = np.mean(tp[present_recall] / support[present_recall])
        f1_denominator = support + predicted
        present_f1 = f1_denominator > 0
        f1 = 2 * tp[present_f1] / f1_denominator[present_f1]
        samples["accuracy"].append(accuracy)
        samples["balanced_accuracy"].append(balanced)
        samples["macro_f1"].append(np.mean(f1))

    rows = []
    for metric, values in samples.items():
        values_array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": float(np.mean(values_array)),
                "ci_2_5": float(np.percentile(values_array, 2.5)),
                "ci_97_5": float(np.percentile(values_array, 97.5)),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def analyze_prediction_artifacts(
    result_dir: Path,
    n_bootstrap: int = 500,
    seed: int = 20261021,
) -> dict[str, pd.DataFrame]:
    analysis_dir = pilot_analysis_dir(result_dir)
    strat_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    distance_band_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, object]] = []

    for path in sorted(result_dir.glob("*/*/predictions.parquet")):
        predictions = pd.read_parquet(path)
        protocol = str(predictions["protocol"].iloc[0])
        model = str(predictions["model"].iloc[0])

        if "strat_group" in predictions.columns:
            strat = (
                predictions.groupby("strat_group", observed=True)
                .agg(
                    n_rows=("correct", "size"),
                    n_wells=("well_id", "nunique"),
                    accuracy=("correct", "mean"),
                    mean_confidence=("confidence", "mean"),
                )
                .reset_index()
            )
            balanced = (
                predictions.groupby("strat_group", observed=True)
                .apply(
                    _balanced_accuracy_from_predictions,
                    include_groups=False,
                )
                .rename("balanced_accuracy")
                .reset_index()
            )
            strat = strat.merge(balanced, on="strat_group", how="left")
            strat = strat.assign(protocol=protocol, model=model)
            strat_frames.append(strat)

        well = (
            predictions.groupby("well_id", observed=True)
            .agg(
                n_rows=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
                distance_to_nearest_train_m=("distance_to_nearest_train_m", "median"),
                modal_strat_group=("strat_group", lambda values: values.mode().iloc[0]),
            )
            .reset_index()
        )
        well["balanced_accuracy"] = (
            predictions.groupby("well_id", observed=True)
            .apply(
                _balanced_accuracy_from_predictions,
                include_groups=False,
            )
            .reindex(well["well_id"])
            .to_numpy()
        )
        well = well.assign(protocol=protocol, model=model)
        well_frames.append(well)

        distance_km = predictions["distance_to_nearest_train_m"] / 1000.0
        bins = [-0.001, 10, 20, 30, 50, 100, np.inf]
        bin_labels = ["0-10", "10-20", "20-30", "30-50", "50-100", "100+"]
        distance_bin = pd.cut(distance_km, bins=bins, labels=bin_labels, right=False)
        bands = (
            predictions.assign(distance_band=distance_bin)
            .groupby("distance_band", observed=True)
            .agg(
                n_rows=("correct", "size"),
                n_wells=("well_id", "nunique"),
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        bands = bands.assign(protocol=protocol, model=model)
        distance_band_frames.append(bands)

        if protocol != "random_point" and len(well) >= 5:
            rho, p_value = spearmanr(
                well["distance_to_nearest_train_m"],
                well["accuracy"],
                nan_policy="omit",
            )
            correlation_rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "spearman_rho_distance_accuracy": float(rho),
                    "p_value": float(p_value),
                    "n_wells": int(well["well_id"].nunique()),
                }
            )

        bootstrap = bootstrap_by_well(predictions, n_bootstrap=n_bootstrap, seed=seed)
        bootstrap = bootstrap.assign(protocol=protocol, model=model)
        bootstrap_frames.append(bootstrap)

    strat_metrics = pd.concat(strat_frames, ignore_index=True)
    well_metrics = pd.concat(well_frames, ignore_index=True)
    distance_bands = pd.concat(distance_band_frames, ignore_index=True)
    bootstrap_metrics = pd.concat(bootstrap_frames, ignore_index=True)
    distance_correlation = pd.DataFrame(correlation_rows)

    strat_metrics.to_csv(
        analysis_dir / "strat_group_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    well_metrics.to_csv(
        analysis_dir / "well_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distance_bands.to_csv(
        analysis_dir / "distance_band_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_metrics.to_csv(
        analysis_dir / "bootstrap_by_well.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distance_correlation.to_csv(
        analysis_dir / "distance_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "strat_group_metrics": strat_metrics,
        "well_metrics": well_metrics,
        "distance_band_metrics": distance_bands,
        "bootstrap_by_well": bootstrap_metrics,
        "distance_correlation": distance_correlation,
    }


def analyze_distribution_shift(
    result_dir: Path,
    analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = result_dir / "pilot_sample.parquet"
    sample = pd.read_parquet(
        sample_path,
        columns=["well_id", "strat_group", "FORCE_2020_LITHOFACIES_LITHOLOGY"],
    )
    sample = sample.rename(columns={"FORCE_2020_LITHOFACIES_LITHOLOGY": "lithology"})
    role_files = {
        "well_grouped": "well_roles.csv",
        "spatial_block": "spatial_well_roles.csv",
    }
    shift_rows: list[dict[str, object]] = []
    for protocol, role_file in role_files.items():
        roles = pd.read_csv(INTERIM_DIR / "splits" / role_file)
        for fold in sorted(roles["evaluation_fold"].unique()):
            fold_roles = roles.loc[roles["evaluation_fold"] == fold]
            train_wells = set(fold_roles.loc[fold_roles["role"] == "train", "well_id"])
            test_wells = set(fold_roles.loc[fold_roles["role"] == "test", "well_id"])
            train = sample.loc[sample["well_id"].isin(train_wells)]
            if train.empty:
                continue
            for descriptor in ("strat_group", "lithology"):
                train_distribution = train[descriptor].value_counts(normalize=True)
                groups = train_distribution.index.union(sample[descriptor].unique())
                p = train_distribution.reindex(groups, fill_value=0.0).to_numpy()
                for well_id in sorted(test_wells):
                    test_well = sample.loc[sample["well_id"] == well_id]
                    if test_well.empty:
                        continue
                    test_distribution = test_well[descriptor].value_counts(normalize=True)
                    q = test_distribution.reindex(groups, fill_value=0.0).to_numpy()
                    shift_rows.append(
                        {
                            "protocol": protocol,
                            "fold": int(fold),
                            "well_id": well_id,
                            "descriptor": descriptor,
                            "js_divergence": float(jensenshannon(p, q, base=2) ** 2),
                        }
                    )
    shift_frame = (
        pd.DataFrame(shift_rows)
        .pivot_table(
            index=["protocol", "fold", "well_id"],
            columns="descriptor",
            values="js_divergence",
        )
        .rename(
            columns={
                "strat_group": "strat_group_js",
                "lithology": "lithology_js",
            }
        )
        .reset_index()
    )

    correlation_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for protocol in role_files:
            prediction_path = result_dir / protocol / model / "predictions.parquet"
            predictions = pd.read_parquet(
                prediction_path,
                columns=["well_id", "fold", "correct"],
            )
            well_accuracy = (
                predictions.groupby(["fold", "well_id"], observed=True)["correct"]
                .mean()
                .rename("accuracy")
                .reset_index()
            )
            merged = well_accuracy.merge(
                shift_frame.loc[shift_frame["protocol"] == protocol],
                on=["fold", "well_id"],
                how="inner",
            )
            for descriptor in ("strat_group_js", "lithology_js"):
                rho, p_value = spearmanr(
                    merged[descriptor],
                    merged["accuracy"],
                    nan_policy="omit",
                )
                correlation_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "descriptor": descriptor,
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                        "n_wells": len(merged),
                    }
                )
    correlations = pd.DataFrame(correlation_rows)
    shift_frame.to_csv(
        analysis_dir / "well_distribution_shift.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        analysis_dir / "shift_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return shift_frame, correlations

def analyze_pilot(
    result_dir: Path = RESULTS_DIR / "pilot",
    n_bootstrap: int = 500,
    seed: int = 20261021,
) -> dict[str, Path]:
    analysis_dir = pilot_analysis_dir(result_dir)
    folds, pooled, fold_summary = collect_metrics(result_dir)
    folds.to_csv(analysis_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(analysis_dir / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(
        analysis_dir / "fold_metric_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_frames = []
    for path in sorted(result_dir.glob("*/*/per_class_recall_pooled.csv")):
        frame = pd.read_csv(path)
        protocol = path.parts[-3]
        model = path.parts[-2]
        frame = attach_lithology_names(frame, "class_code")
        frame = frame.assign(protocol=protocol, model=model)
        class_frames.append(frame)
    class_metrics = pd.concat(class_frames, ignore_index=True)
    class_metrics.to_csv(
        analysis_dir / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    artifacts = analyze_prediction_artifacts(
        result_dir,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    analyze_distribution_shift(result_dir, analysis_dir)
    payload = {
        "result_dir": str(result_dir),
        "n_folds": int(folds["fold"].nunique()),
        "models": sorted(pooled["model"].unique().tolist()),
        "protocols": sorted(pooled["protocol"].unique().tolist()),
        "bootstrap_iterations": n_bootstrap,
        "outputs": [
            "fold_metric_summary.csv",
            "pooled_metrics.csv",
            "per_class_recall.csv",
            "strat_group_metrics.csv",
            "well_metrics.csv",
            "distance_band_metrics.csv",
            "bootstrap_by_well.csv",
            "distance_correlation.csv",
            "well_distribution_shift.csv",
            "shift_correlation.csv",
        ],
        "unique_wells": int(artifacts["well_metrics"]["well_id"].nunique()),
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "analysis_dir": analysis_dir,
        "fold_summary": analysis_dir / "fold_metric_summary.csv",
        "bootstrap": analysis_dir / "bootstrap_by_well.csv",
    }
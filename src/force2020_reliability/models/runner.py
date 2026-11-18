from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from force2020_reliability.config import ExperimentConfig
from force2020_reliability.evaluation.metrics import (
    evaluate_predictions,
    risk_coverage_at_target,
)
from force2020_reliability.labels import attach_lithology_names
from force2020_reliability.models.baselines import make_model
from force2020_reliability.paths import INTERIM_DIR, RESULTS_DIR
from force2020_reliability.splits.generate import resolve_fold_members

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")


def sample_pilot_rows(
    frame: pd.DataFrame,
    max_rows_per_well: int,
    seed: int,
) -> pd.DataFrame:
    if max_rows_per_well <= 0:
        return frame
    selected_indices: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for _, well_rows in frame.groupby("well_id", sort=True, observed=True):
        indices = well_rows.index.to_numpy()
        if len(indices) > max_rows_per_well:
            indices = np.sort(rng.choice(indices, size=max_rows_per_well, replace=False))
        selected_indices.append(indices)
    result = frame.loc[np.concatenate(selected_indices)].sort_index().copy()
    return result


def filter_supported_classes(
    frame: pd.DataFrame,
    target_column: str,
    min_class_count: int,
    min_class_wells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        frame.groupby(target_column, observed=True)
        .agg(rows=(target_column, "size"), n_wells=("well_id", "nunique"))
        .reset_index()
        .rename(columns={target_column: "class_code"})
        .sort_values("class_code")
    )
    supported = counts.loc[
        (counts["rows"] >= min_class_count) & (counts["n_wells"] >= min_class_wells),
        "class_code",
    ].astype(int)
    filtered = frame.loc[frame[target_column].isin(set(supported))].copy()
    counts["retained"] = counts["class_code"].isin(set(supported))
    counts = attach_lithology_names(counts, "class_code")
    return filtered, counts


def _prediction_probabilities(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_, dtype=int)
    return probabilities, classes


def run_protocol(
    protocol: str,
    model_name: str,
    frame: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    splits_dir: Path = INTERIM_DIR / "splits",
    max_folds: int | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    protocol_dir = output_dir / protocol / model_name
    protocol_dir.mkdir(parents=True, exist_ok=True)
    fold_metric_rows: list[dict[str, float | int | str]] = []
    per_class_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    aligned_probability_frames: list[np.ndarray] = []
    n_folds = config.n_folds if max_folds is None else min(max_folds, config.n_folds)
    features = feature_columns or config.numeric_features
    target = config.target_column
    global_classes = np.sort(frame[target].unique()).astype(int)

    for fold in range(n_folds):
        train_index, test_index = resolve_fold_members(
            protocol=protocol,
            fold=fold,
            frame=frame,
            splits_dir=splits_dir,
        )
        if len(train_index) == 0 or len(test_index) == 0:
            raise ValueError(f"{protocol} fold {fold} has empty train or test partition")
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        model = make_model(model_name, config.seed + fold)
        model.fit(train.loc[:, features], train[target].to_numpy(dtype=int))
        probabilities, classes = _prediction_probabilities(model, test.loc[:, features])
        predictions = classes[np.argmax(probabilities, axis=1)]
        aligned_probabilities = np.zeros(
            (len(test), len(global_classes)), dtype=float
        )
        class_positions = np.searchsorted(global_classes, classes)
        aligned_probabilities[:, class_positions] = probabilities
        metrics, per_class, curve, prediction_frame = evaluate_predictions(
            y_true=test[target].to_numpy(dtype=int),
            y_pred=predictions,
            probabilities=aligned_probabilities,
            classes=global_classes,
            well_ids=test["well_id"].to_numpy(),
        )
        aligned_probability_frames.append(aligned_probabilities)
        metrics.update(
            {
                "protocol": protocol,
                "model": model_name,
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_wells": int(train["well_id"].nunique()),
                "test_wells": int(test["well_id"].nunique()),
            }
        )
        metrics.update(risk_coverage_at_target(curve, target_accuracy=0.90))
        fold_metric_rows.append(metrics)
        per_class = per_class.assign(
            protocol=protocol,
            model=model_name,
            fold=fold,
        )
        per_class_frames.append(per_class)
        curve_frames.append(curve.assign(protocol=protocol, model=model_name, fold=fold))
        prediction_frame["row_id"] = test["row_id"].to_numpy()
        if "strat_group" in test.columns:
            prediction_frame["strat_group"] = test["strat_group"].to_numpy()
        if protocol == "random_point":
            prediction_frame["distance_to_nearest_train_m"] = 0.0
        else:
            role_path = (
                splits_dir / "well_roles.csv"
                if protocol == "well_grouped"
                else splits_dir / "spatial_well_roles.csv"
            )
            fold_roles = pd.read_csv(role_path)
            fold_roles = fold_roles.loc[fold_roles["evaluation_fold"] == fold]
            distance_map = fold_roles.set_index("well_id")["distance_to_nearest_train_m"]
            prediction_frame["distance_to_nearest_train_m"] = (
                test["well_id"].map(distance_map).to_numpy(dtype=float)
            )
        prediction_frame = prediction_frame.assign(
            protocol=protocol,
            model=model_name,
            fold=fold,
        )
        prediction_frames.append(prediction_frame)
        print(
            f"{protocol}/{model_name} fold={fold} "
            f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    pooled_predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled_probabilities = np.vstack(aligned_probability_frames)
    np.savez_compressed(
        protocol_dir / "probabilities.npz",
        probabilities=pooled_probabilities,
        classes=global_classes,
    )
    pooled_metrics, pooled_per_class, pooled_curve, _ = evaluate_predictions(
        y_true=pooled_predictions["y_true"].to_numpy(dtype=int),
        y_pred=pooled_predictions["y_pred"].to_numpy(dtype=int),
        probabilities=pooled_probabilities,
        classes=global_classes,
        well_ids=pooled_predictions["well_id"].to_numpy(),
    )
    pooled_metrics.update(
        {
            "protocol": protocol,
            "model": model_name,
            "fold": -1,
            "train_rows": int(fold_metrics["train_rows"].sum()),
            "test_rows": int(fold_metrics["test_rows"].sum()),
            "train_wells": int(fold_metrics["train_wells"].sum()),
            "test_wells": int(pooled_predictions["well_id"].nunique()),
        }
    )
    pooled_metrics.update(risk_coverage_at_target(pooled_curve, target_accuracy=0.90))

    all_metrics = pd.concat([fold_metrics, pd.DataFrame([pooled_metrics])], ignore_index=True)
    all_metrics.to_csv(protocol_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_class_frames, ignore_index=True).to_csv(
        protocol_dir / "per_class_recall.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(curve_frames, ignore_index=True).to_csv(
        protocol_dir / "risk_coverage_folds.csv", index=False, encoding="utf-8-sig"
    )
    pooled_curve.to_csv(protocol_dir / "risk_coverage_pooled.csv", index=False, encoding="utf-8-sig")
    pooled_per_class.to_csv(
        protocol_dir / "per_class_recall_pooled.csv", index=False, encoding="utf-8-sig"
    )
    pooled_predictions.to_parquet(protocol_dir / "predictions.parquet", index=False)
    (protocol_dir / "summary.json").write_text(
        json.dumps(pooled_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pooled_metrics





def run_pilot(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    models: list[str] | None = None,
    max_folds: int | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    output = output_dir or RESULTS_DIR / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    filtered, class_counts = filter_supported_classes(
        frame,
        target_column=config.target_column,
        min_class_count=config.min_class_count,
        min_class_wells=config.min_class_wells,
    )
    sampled = sample_pilot_rows(
        filtered,
        max_rows_per_well=config.pilot_max_rows_per_well,
        seed=config.seed,
    )
    sampled.to_parquet(output / "pilot_sample.parquet", index=False)
    class_counts.to_csv(output / "class_counts.csv", index=False, encoding="utf-8-sig")

    selected_models = models or list(config.raw["models"])
    summaries: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for model_name in selected_models:
            summary = run_protocol(
                protocol=protocol,
                model_name=model_name,
                frame=sampled,
                config=config,
                output_dir=output,
                max_folds=max_folds,
            )
            summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "protocol_model_summary.csv", index=False, encoding="utf-8-sig")
    return summary_frame
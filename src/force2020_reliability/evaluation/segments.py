from __future__ import annotations

import numpy as np
import pandas as pd


def _with_runs(
    frame: pd.DataFrame,
    gap_threshold_m: float,
) -> pd.DataFrame:
    data = frame.sort_values(["well_id", "DEPTH_MD", "row_id"]).copy()
    data = data.dropna(subset=["DEPTH_MD"]).reset_index(drop=True)
    depth_gap = data.groupby("well_id", observed=True)["DEPTH_MD"].diff()
    new_run = depth_gap.isna() | (depth_gap > gap_threshold_m)
    data["run_id"] = new_run.cumsum().astype(int)
    return data


def _boundaries(
    frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    data = frame.sort_values("DEPTH_MD")
    changes = data[label_column].ne(data[label_column].shift())
    changes.iloc[0] = False
    return data.loc[changes, ["run_id", "DEPTH_MD"]].rename(columns={"DEPTH_MD": "boundary_m"})


def _match_boundaries(
    true_boundaries: np.ndarray,
    predicted_boundaries: np.ndarray,
    tolerance_m: float,
) -> tuple[int, float]:
    if len(true_boundaries) == 0 or len(predicted_boundaries) == 0:
        return 0, float("nan")
    true_values = np.sort(np.asarray(true_boundaries, dtype=float))
    predicted_values = np.sort(np.asarray(predicted_boundaries, dtype=float))
    true_index = 0
    predicted_index = 0
    matched = 0
    distances: list[float] = []
    while true_index < len(true_values) and predicted_index < len(predicted_values):
        difference = predicted_values[predicted_index] - true_values[true_index]
        if abs(difference) <= tolerance_m:
            matched += 1
            distances.append(abs(float(difference)))
            true_index += 1
            predicted_index += 1
        elif difference < -tolerance_m:
            predicted_index += 1
        else:
            true_index += 1
    mae = float(np.mean(distances)) if distances else float("nan")
    return matched, mae

def _depth_weights(frame: pd.DataFrame) -> np.ndarray:
    data = frame.sort_values("DEPTH_MD")
    depths = data["DEPTH_MD"].to_numpy(dtype=float)
    deltas = np.diff(depths, append=np.nan)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    fallback = float(np.median(positive)) if len(positive) else 0.152
    deltas[~np.isfinite(deltas) | (deltas <= 0)] = fallback
    return deltas


def evaluate_well_segments(
    well_predictions: pd.DataFrame,
    gap_threshold_m: float = 0.5,
    boundary_tolerance_m: float = 0.5,
) -> dict[str, float | str | int]:
    data = _with_runs(well_predictions, gap_threshold_m=gap_threshold_m)
    well_id = str(data["well_id"].iloc[0])
    total_matched = 0
    total_true_boundaries = 0
    total_predicted_boundaries = 0
    boundary_maes: list[float] = []
    true_thickness: dict[int, float] = {}
    predicted_thickness: dict[int, float] = {}
    true_segment_count = 0
    predicted_segment_count = 0

    for run_id, run in data.groupby("run_id", observed=True):
        run = run.copy()
        true_boundary_frame = _boundaries(run, "y_true")
        predicted_boundary_frame = _boundaries(run, "y_pred")
        true_boundaries = true_boundary_frame["boundary_m"].to_numpy(dtype=float)
        predicted_boundaries = predicted_boundary_frame["boundary_m"].to_numpy(dtype=float)
        matched, mae = _match_boundaries(
            true_boundaries,
            predicted_boundaries,
            tolerance_m=boundary_tolerance_m,
        )
        total_matched += matched
        total_true_boundaries += len(true_boundaries)
        total_predicted_boundaries += len(predicted_boundaries)
        if np.isfinite(mae):
            boundary_maes.append(mae)
        true_segment_count += int(len(true_boundaries) + 1)
        predicted_segment_count += int(len(predicted_boundaries) + 1)

        run = run.sort_values("DEPTH_MD")
        weights = _depth_weights(run)
        for label, weight in zip(run["y_true"], weights, strict=True):
            true_thickness[int(label)] = true_thickness.get(int(label), 0.0) + float(weight)
        for label, weight in zip(run["y_pred"], weights, strict=True):
            predicted_thickness[int(label)] = predicted_thickness.get(int(label), 0.0) + float(weight)

    precision = (
        total_matched / total_predicted_boundaries
        if total_predicted_boundaries > 0
        else float("nan")
    )
    recall = (
        total_matched / total_true_boundaries
        if total_true_boundaries > 0
        else float("nan")
    )
    if total_true_boundaries == 0 and total_predicted_boundaries == 0:
        f1 = 1.0
    elif np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    classes = set(true_thickness) | set(predicted_thickness)
    absolute_errors = [
        abs(predicted_thickness.get(label, 0.0) - true_thickness.get(label, 0.0))
        for label in classes
    ]
    total_true = sum(true_thickness.values())
    total_absolute_error = float(np.sum(absolute_errors)) if absolute_errors else 0.0
    return {
        "well_id": well_id,
        "n_rows": len(data),
        "point_accuracy": float((data["y_true"] == data["y_pred"]).mean()),
        "n_depth_runs": int(data["run_id"].nunique()),
        "true_boundaries": total_true_boundaries,
        "predicted_boundaries": total_predicted_boundaries,
        "matched_boundaries": total_matched,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_mae_m": float(np.mean(boundary_maes)) if boundary_maes else float("nan"),
        "true_segments": true_segment_count,
        "predicted_segments": predicted_segment_count,
        "segment_count_abs_error": abs(predicted_segment_count - true_segment_count),
        "thickness_total_abs_error_m": total_absolute_error,
        "thickness_mae_m": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
        "thickness_relative_error": (
            total_absolute_error / total_true if total_true > 0 else float("nan")
        ),
    }


def evaluate_prediction_segments(
    predictions: pd.DataFrame,
    gap_threshold_m: float = 0.5,
    boundary_tolerance_m: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for _, well in predictions.groupby("well_id", observed=True):
        rows.append(
            evaluate_well_segments(
                well,
                gap_threshold_m=gap_threshold_m,
                boundary_tolerance_m=boundary_tolerance_m,
            )
        )
    return pd.DataFrame(rows)

def smooth_label_sequence(labels: np.ndarray, min_run_points: int) -> np.ndarray:
    values = np.asarray(labels, dtype=int).copy()
    if min_run_points <= 1 or len(values) <= 1:
        return values
    while True:
        runs: list[dict[str, int]] = []
        start = 0
        for index in range(1, len(values) + 1):
            if index == len(values) or values[index] != values[start]:
                runs.append({"label": int(values[start]), "start": start, "end": index})
                start = index
        if len(runs) <= 1:
            break
        short = [
            (run["end"] - run["start"], index)
            for index, run in enumerate(runs)
            if run["end"] - run["start"] < min_run_points
        ]
        if not short:
            break
        _, short_index = min(short)
        left = runs[short_index - 1] if short_index > 0 else None
        right = runs[short_index + 1] if short_index + 1 < len(runs) else None
        left_length = left["end"] - left["start"] if left else -1
        right_length = right["end"] - right["start"] if right else -1
        target_label = left["label"] if left_length >= right_length else right["label"]
        values[runs[short_index]["start"] : runs[short_index]["end"]] = target_label
    return values


def smooth_predictions_min_run(
    predictions: pd.DataFrame,
    min_run_m: float,
    gap_threshold_m: float = 0.5,
) -> pd.DataFrame:
    data = _with_runs(predictions, gap_threshold_m=gap_threshold_m)
    data["y_pred_smooth"] = data["y_pred"].astype(int)
    for indices in data.groupby("run_id", observed=True).groups.values():
        run = data.loc[indices].sort_values("DEPTH_MD")
        depths = run["DEPTH_MD"].to_numpy(dtype=float)
        positive_steps = np.diff(depths)
        positive_steps = positive_steps[positive_steps > 0]
        step = float(np.median(positive_steps)) if len(positive_steps) else 0.152
        min_points = max(1, int(np.ceil(min_run_m / step)))
        smoothed = smooth_label_sequence(run["y_pred"].to_numpy(dtype=int), min_points)
        data.loc[run.index, "y_pred_smooth"] = smoothed
    return data
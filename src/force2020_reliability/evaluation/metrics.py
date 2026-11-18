from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def confidence_columns(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected a two-dimensional multiclass probability matrix")
    sorted_probabilities = np.sort(probabilities, axis=1)
    top1 = sorted_probabilities[:, -1]
    top2 = sorted_probabilities[:, -2]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log(clipped), axis=1)
    entropy_normalized = entropy / np.log(probabilities.shape[1])
    return {
        "confidence": top1,
        "margin": top1 - top2,
        "entropy_normalized": entropy_normalized,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 15,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = bins == index
        if not mask.any():
            continue
        accuracy = float(np.mean(y_true[mask] == y_pred[mask]))
        mean_confidence = float(np.mean(confidence[mask]))
        ece += float(mask.mean()) * abs(accuracy - mean_confidence)
    return ece


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    coverage_points: int = 100,
) -> pd.DataFrame:
    order = np.argsort(-confidence, kind="stable")
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    correct = (y_true_sorted == y_pred_sorted).astype(np.int8)
    cumulative_correct = np.cumsum(correct)
    n_samples = len(correct)
    coverages = np.linspace(1.0 / n_samples, 1.0, coverage_points)
    counts = np.maximum(1, np.ceil(coverages * n_samples).astype(int))
    selective_accuracy = cumulative_correct[counts - 1] / counts
    return pd.DataFrame(
        {
            "coverage": counts / n_samples,
            "selective_accuracy": selective_accuracy,
            "risk": 1.0 - selective_accuracy,
            "threshold": confidence[order][counts - 1],
            "n_selected": counts,
        }
    )


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    ordered = curve.sort_values("coverage")
    if len(ordered) < 2:
        return float("nan")
    return float(np.trapezoid(ordered["risk"], ordered["coverage"]))


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities, dtype=float)
    for row_index, label in enumerate(y_true):
        one_hot[row_index, class_to_index[int(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    well_ids: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred]))).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros_like(support, dtype=float),
        where=support > 0,
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(np.mean(recalls[support > 0]))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    nll = float(log_loss(y_true, probabilities, labels=classes))
    brier = multiclass_brier(y_true, probabilities, classes)

    confidences = confidence_columns(probabilities)
    top_confidence = confidences["confidence"]
    ece = expected_calibration_error(y_true, y_pred, top_confidence)
    curve = risk_coverage_curve(y_true, y_pred, top_confidence)
    metrics: dict[str, float | int] = {
        "n_rows": len(y_true),
        "n_wells": int(pd.Series(well_ids).nunique()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": nll,
        "multiclass_brier": brier,
        "ece_top_label": ece,
        "mean_confidence": float(np.mean(top_confidence)),
        "aurc": area_under_risk_coverage(curve),
    }

    per_class = pd.DataFrame(
        {
            "class_code": labels,
            "support": support,
            "recall": recalls,
        }
    )

    prediction_frame = pd.DataFrame(
        {
            "well_id": well_ids,
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "confidence": top_confidence,
            "margin": confidences["margin"],
            "entropy_normalized": confidences["entropy_normalized"],
            "correct": y_true == y_pred,
        }
    )
    return metrics, per_class, curve, prediction_frame


def risk_coverage_at_target(curve: pd.DataFrame, target_accuracy: float = 0.90) -> dict[str, float]:
    eligible = curve.loc[curve["selective_accuracy"] >= target_accuracy].sort_values(
        "coverage", ascending=False
    )
    if eligible.empty:
        return {
            "target_accuracy": target_accuracy,
            "coverage_at_target": 0.0,
            "threshold_at_target": float("nan"),
        }
    row = eligible.iloc[0]
    return {
        "target_accuracy": target_accuracy,
        "coverage_at_target": float(row["coverage"]),
        "threshold_at_target": float(row["threshold"]),
    }
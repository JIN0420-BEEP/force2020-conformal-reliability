from __future__ import annotations

import numpy as np

from force2020_reliability.evaluation.metrics import (
    evaluate_predictions,
    risk_coverage_at_target,
    risk_coverage_curve,
)
from force2020_reliability.models.runner import filter_supported_classes


def test_risk_coverage_reaches_full_coverage():
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 1, 2])
    confidence = np.array([0.95, 0.90, 0.70, 0.85, 0.60, 0.55])
    curve = risk_coverage_curve(y_true, y_pred, confidence, coverage_points=20)
    assert curve["coverage"].is_monotonic_increasing
    assert curve.iloc[-1]["coverage"] == 1.0
    assert curve.iloc[-1]["selective_accuracy"] == np.mean(y_true == y_pred)
    target = risk_coverage_at_target(curve, target_accuracy=0.8)
    assert 0.0 <= target["coverage_at_target"] <= 1.0


def test_prediction_metrics_include_calibration_and_reliability():
    classes = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.5, 0.3, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    y_true = np.array([0, 1, 2, 2])
    y_pred = probabilities.argmax(axis=1)
    metrics, per_class, curve, predictions = evaluate_predictions(
        y_true, y_pred, probabilities, classes, np.array(["A", "A", "B", "B"])
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["n_wells"] == 2
    assert "ece_top_label" in metrics
    assert {"class_code", "support", "recall"}.issubset(per_class.columns)
    assert {"coverage", "selective_accuracy", "risk"}.issubset(curve.columns)
    assert {"confidence", "margin", "correct"}.issubset(predictions.columns)


def test_class_filter_requires_enough_rows_and_wells(synthetic_frame):
    filtered, counts = filter_supported_classes(
        synthetic_frame,
        target_column="lithology",
        min_class_count=50,
        min_class_wells=21,
    )
    assert filtered.empty
    assert not counts["retained"].any()
    assert "lithology_name" in counts.columns
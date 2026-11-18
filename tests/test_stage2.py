from __future__ import annotations

import numpy as np

from force2020_reliability.models.feature_ablation import feature_sets
from force2020_reliability.models.nested import select_selective_threshold


def test_selective_threshold_uses_calibration_curve():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    confidence = np.array([0.90, 0.80, 0.70, 0.60])
    threshold, curve = select_selective_threshold(
        y_true,
        y_pred,
        confidence,
        target_accuracy=0.80,
    )
    assert threshold == 0.70
    assert curve.iloc[-1]["coverage"] == 1.0
    assert curve.iloc[-1]["selective_accuracy"] == 0.75


def test_feature_sets_remove_location_and_depth(config):
    sets = feature_sets(config)
    assert set(sets) == {"full", "logs_only", "logs_depth", "logs_location"}
    assert "x_loc" not in sets["logs_only"]
    assert "y_loc" not in sets["logs_only"]
    assert "x_loc" in sets["logs_location"]
    assert "DEPTH_MD" in sets["logs_depth"]
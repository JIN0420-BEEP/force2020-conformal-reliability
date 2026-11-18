from __future__ import annotations

import numpy as np
import pandas as pd

from force2020_reliability.evaluation.segments import evaluate_well_segments, smooth_label_sequence
from force2020_reliability.models.baselines import make_model


def test_segment_metrics_match_shifted_boundary():
    frame = pd.DataFrame(
        {
            "well_id": "W1",
            "row_id": np.arange(6),
            "DEPTH_MD": np.arange(6) * 0.5,
            "y_true": [30000, 30000, 65000, 65000, 70000, 70000],
            "y_pred": [30000, 30000, 30000, 65000, 65000, 70000],
        }
    )
    metrics = evaluate_well_segments(
        frame,
        gap_threshold_m=0.5,
        boundary_tolerance_m=0.5,
    )
    assert metrics["point_accuracy"] == 0.6666666666666666
    assert metrics["boundary_f1"] == 1.0
    assert metrics["boundary_mae_m"] == 0.5
    assert metrics["thickness_total_abs_error_m"] == 1.0


def test_segment_metrics_break_runs_on_large_gap():
    frame = pd.DataFrame(
        {
            "well_id": "W1",
            "row_id": np.arange(4),
            "DEPTH_MD": [0.0, 0.5, 5.0, 5.5],
            "y_true": [30000, 30000, 65000, 65000],
            "y_pred": [30000, 30000, 65000, 65000],
        }
    )
    metrics = evaluate_well_segments(frame, gap_threshold_m=0.5)
    assert metrics["n_depth_runs"] == 2
    assert metrics["boundary_f1"] == 1.0


def test_lightgbm_maps_noncontiguous_force_labels():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(45, 3))
    labels = np.array([30000, 65000, 70000] * 15)
    model = make_model("lightgbm", seed=42).fit(features, labels)
    predictions = model.predict(features[:9])
    assert model.classes_.tolist() == [30000, 65000, 70000]
    assert set(predictions).issubset(set(labels))
    assert model.predict_proba(features[:9]).shape == (9, 3)

def test_minimum_run_smoothing_removes_short_fragment():
    labels = np.array([30000, 30000, 65000, 30000, 30000])
    smoothed = smooth_label_sequence(labels, min_run_points=2)
    assert smoothed.tolist() == [30000] * 5
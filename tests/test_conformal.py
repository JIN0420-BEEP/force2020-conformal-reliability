from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from force2020_reliability.evaluation import conformal as cp
from force2020_reliability.evaluation.distribution_shift import js_divergence
from force2020_reliability.evaluation.calibration import TemperatureScaler


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(100, dtype=float)
    # ceil((100+1)*0.9) = 91 -> index 90 -> value 90
    assert cp.conformal_quantile(scores, alpha=0.1) == 90.0
    with pytest.raises(ValueError):
        cp.conformal_quantile(np.array([]), alpha=0.1)


def test_mondrian_quantiles_one_per_class():
    classes = np.array([0, 1])
    labels = np.array([0] * 20 + [1] * 20)
    scores = np.linspace(0, 1, 40)
    quantiles = cp.mondrian_quantiles(scores, labels, classes, alpha=0.2)
    assert set(quantiles) == {0, 1}


def test_js_divergence_bounded_and_zero_for_equal_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    q = np.array([0.5, 0.2, 0.3])
    assert 0.0 < js_divergence(p, q) < np.log(2)


def test_prediction_sets_cover_and_are_nonempty():
    X, y = make_classification(
        n_samples=6000, n_features=15, n_informative=5, n_redundant=2,
        n_classes=5, n_clusters_per_class=2, random_state=1)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=1)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=1)

    model = HistGradientBoostingClassifier(max_iter=80, random_state=1).fit(X_tr, y_tr)
    classes = np.asarray(model.classes_)
    p_cal, p_test = model.predict_proba(X_cal), model.predict_proba(X_test)

    alpha = 0.1
    scores = cp.aps_calibration_scores(
        p_cal, y_cal, classes, random_values=np.random.default_rng(2).random(len(y_cal)))
    quantile = cp.conformal_quantile(scores, alpha)
    membership = cp.aps_prediction_sets(
        p_test, classes, quantile,
        random_values=np.random.default_rng(3).random(len(y_test)))
    diag = cp.prediction_set_diagnostics(membership, y_test, classes)

    assert diag["empty_sets"] == 0
    assert diag["coverage"] >= 1 - alpha - 0.03
    assert diag["mean_set_size"] >= 1.0


def test_temperature_scaler_returns_positive_temperature():
    rng = np.random.default_rng(0)
    classes = np.array([0, 1, 2])
    probabilities = rng.dirichlet(np.ones(3), size=300)
    labels = rng.integers(0, 3, size=300)
    scaler = TemperatureScaler().fit(probabilities, labels, classes)
    assert scaler.temperature_ > 0
    calibrated = scaler.transform(probabilities)
    assert np.allclose(calibrated.sum(axis=1), 1.0)

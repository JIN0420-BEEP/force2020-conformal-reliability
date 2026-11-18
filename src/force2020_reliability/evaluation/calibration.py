"""Post-hoc probability calibration: temperature scaling and isotonic mapping."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-12


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, _EPS, 1.0)
    return np.log(clipped)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _negative_log_likelihood(
    temperature: float,
    logits: np.ndarray,
    class_index: np.ndarray,
) -> float:
    probabilities = _softmax(logits / max(temperature, _EPS))
    selected = np.clip(probabilities[np.arange(len(class_index)), class_index], _EPS, 1.0)
    return float(-np.mean(np.log(selected)))


class TemperatureScaler:
    """Single-parameter temperature scaling fitted on a held-out calibration set."""

    def __init__(self) -> None:
        self.temperature_: float | None = None

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        classes: np.ndarray,
    ) -> "TemperatureScaler":
        class_index = np.searchsorted(classes, labels)
        logits = _logits(probabilities)
        result = minimize_scalar(
            _negative_log_likelihood,
            bounds=(0.05, 20.0),
            args=(logits, class_index),
            method="bounded",
            options={"xatol": 1e-4},
        )
        self.temperature_ = float(result.x)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        return _softmax(_logits(probabilities) / self.temperature_)


class IsotonicTopLabel:
    """Per-score isotonic regression applied to the top-class probability."""

    def __init__(self) -> None:
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probabilities, labels, classes) -> "IsotonicTopLabel":
        class_index = np.searchsorted(classes, labels)
        top_index = np.argmax(probabilities, axis=1)
        scores = probabilities[np.arange(len(labels)), top_index]
        correct = (top_index == class_index).astype(float)
        self.model_.fit(scores, correct)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = probabilities.copy()
        top_index = np.argmax(probabilities, axis=1)
        rows = np.arange(len(probabilities))
        top_scores = probabilities[rows, top_index]
        mapped = self.model_.predict(top_scores)
        calibrated[rows, top_index] = mapped
        # Renormalize the remaining mass onto the other classes proportionally.
        remaining = 1.0 - mapped
        non_top = np.ones_like(probabilities, dtype=bool)
        non_top[rows, top_index] = False
        non_top_sum = np.where(probabilities * non_top > 0, probabilities, 0.0).sum(axis=1)
        factor = np.divide(remaining, non_top_sum, out=np.zeros_like(remaining),
                           where=non_top_sum > 0)
        calibrated = np.where(non_top, probabilities * factor[:, None], calibrated)
        return calibrated

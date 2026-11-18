"""Quantify covariate and label distribution shift between train and test folds."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Label distribution shift
# ---------------------------------------------------------------------------


def _probability_vector(counts: np.ndarray, support: np.ndarray) -> np.ndarray:
    totals = np.array([counts[support == value].sum() for value in support], dtype=float)
    return totals / max(totals.sum(), 1.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence with the natural logarithm, bounded by ln 2."""
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log(a / b))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _class_distribution(labels: np.ndarray, support: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(support), dtype=float)
    if len(labels):
        values, frequencies = np.unique(labels, return_counts=True)
        counts[np.searchsorted(support, values)] = frequencies
    return counts / max(counts.sum(), 1.0)


def label_shift(y_train: np.ndarray, y_test: np.ndarray) -> float:
    support = np.unique(np.concatenate([y_train, y_test]))
    p = _class_distribution(y_train, support)
    q = _class_distribution(y_test, support)
    return js_divergence(p, q)


# ---------------------------------------------------------------------------
# Covariate shift
# ---------------------------------------------------------------------------


def _standardize(X: np.ndarray, reference_mean, reference_std) -> np.ndarray:
    return (X - reference_mean) / np.where(reference_std > 0, reference_std, 1.0)


def _subsample(X: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) > max_points:
        index = rng.choice(len(X), size=max_points, replace=False)
        return X[index]
    return X


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Unbiased squared MMD with an RBF kernel on standardized features."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = _standardize(X, mean, std)
    Y = _standardize(Y, mean, std)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    xx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    yy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    xy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    n, m = len(X), len(Y)
    np.fill_diagonal(xx, 0.0)
    np.fill_diagonal(yy, 0.0)
    return float(xx.sum() / (n * (n - 1)) + yy.sum() / (m * (m - 1))
                 - 2.0 * xy.sum() / (n * m))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Sample energy distance on standardized features."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = _standardize(X, mean, std)
    Y = _standardize(Y, mean, std)
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return float(2.0 * d_xy - d_xx - d_yy)


def fold_shift(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_points: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Return all shift metrics for one fold, with bounded kernel cost."""
    rng = np.random.default_rng(seed)
    # Well-log curves contain missing values; impute with training column means
    # before kernel-based shift metrics (whole-NaN columns become zero).
    column_mean = np.nanmean(X_train, axis=0)
    column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
    X_train = np.where(np.isnan(X_train), column_mean, X_train)
    X_test = np.where(np.isnan(X_test), column_mean, X_test)
    X_train = _subsample(X_train, max_points, rng)
    X_test = _subsample(X_test, max_points, rng)
    return {
        "mmd": mmd_rbf(X_train, X_test),
        "energy": energy_distance(X_train, X_test),
        "label_js": label_shift(y_train, y_test),
    }

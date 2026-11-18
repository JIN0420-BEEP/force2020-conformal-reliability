"""Distribution-free uncertainty quantification via split conformal prediction.

This module implements multiclass split conformal prediction with the adaptive
prediction-set (APS) nonconformity score of Romano et al. (2020), its regularized
variant (RAPS; Angelopoulos et al., 2022), and Mondrian (class-conditional)
calibration.  All routines are vectorized and operate on probability matrices
aligned to a fixed, sorted array of class labels.

Under exchangeability, the resulting prediction sets satisfy the finite-sample
marginal guarantee ``P(Y in C(X)) >= 1 - alpha``.  When the calibration and test
populations differ (for example across a spatial hold-out), exchangeability is
violated and the empirical coverage is allowed to drift; quantifying that drift is
the purpose of the accompanying experiments.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------


def _rank_matrix(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return classes ranked by descending probability and the ranked matrix."""
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked = np.take_along_axis(probabilities, order, axis=1)
    return order, ranked


def true_class_index(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Position of each integer label inside the sorted ``classes`` array."""
    return np.searchsorted(classes, labels)


def aps_calibration_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    random_values: np.ndarray | None = None,
) -> np.ndarray:
    """Randomized APS score for every calibration example.

    The score is the cumulative probability, over classes ranked by descending
    probability, up to and including the observed class; a uniform randomization
    subtracts ``U * p_true`` so the method attains exact rather than conservative
    coverage.
    """
    order, ranked = _rank_matrix(probabilities)
    cumulative = np.cumsum(ranked, axis=1)
    class_idx = true_class_index(labels, classes)
    rank_pos = np.argmax(order == class_idx[:, None], axis=1)
    rows = np.arange(len(labels))
    p_true = probabilities[rows, class_idx]
    score = cumulative[rows, rank_pos]
    if random_values is None:
        random_values = np.random.default_rng().random(len(labels))
    return score - random_values * p_true


def tps_calibration_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Naive nonconformity score ``1 - p(Y|X)`` (higher means less conforming)."""
    class_idx = true_class_index(labels, classes)
    return 1.0 - probabilities[np.arange(len(labels)), class_idx]


# ---------------------------------------------------------------------------
# Quantile calibration
# ---------------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile with the ``(n+1)`` correction."""
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate on an empty calibration set")
    k = np.ceil((n + 1) * (1.0 - alpha)).astype(int)
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def mondrian_quantiles(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    alpha: float,
) -> dict[int, float]:
    """Class-conditional quantiles, one per observed label value."""
    quantiles: dict[int, float] = {}
    for class_label in classes:
        mask = labels == class_label
        if mask.any():
            quantiles[int(class_label)] = conformal_quantile(scores[mask], alpha)
    return quantiles


# ---------------------------------------------------------------------------
# Prediction sets
# ---------------------------------------------------------------------------


def aps_prediction_sets(
    probabilities: np.ndarray,
    classes: np.ndarray,
    quantile: float,
    random_values: np.ndarray | None = None,
    raps_penalty: float = 0.0,
    raps_target: int | None = None,
) -> np.ndarray:
    """Construct a boolean membership matrix of APS/RAPS prediction sets.

    Parameters
    ----------
    raps_penalty:
        Additional per-class penalty ``lambda_reg`` applied once the ranked
        cumulative probability exceeds ``raps_target`` (the RAPS regularization).
    raps_target:
        Rank index ``k_reg`` at which the RAPS penalty starts; defaults to the
        nominal ``1 - alpha`` quantile of a uniform ranking.
    """
    n, n_classes = probabilities.shape
    order, ranked = _rank_matrix(probabilities)
    if random_values is None:
        random_values = np.random.default_rng().random(n)

    q_values = np.asarray(quantile, dtype=float)
    if q_values.ndim == 0:
        q_values = np.full(n, float(q_values))
    if q_values.shape != (n,):
        raise ValueError("quantile must be a scalar or a length-n array")

    adjusted = ranked.copy()
    if raps_penalty > 0.0:
        target = raps_target if raps_target is not None else max(1, int(1.0 / max(alpha, 1e-9)))
        rank_index = np.arange(n_classes)[None, :]
        adjusted = ranked + raps_penalty * np.maximum(0, rank_index + 1 - target)

    cumulative = np.cumsum(adjusted, axis=1)
    # First rank whose randomized cumulative value reaches the quantile.
    stop = np.sum(cumulative - random_values[:, None] * adjusted <= q_values[:, None], axis=1)
    stop = np.clip(stop, 1, n_classes)

    membership = np.zeros((n, n_classes), dtype=bool)
    rows = np.arange(n)
    chosen_ranks = np.repeat(rows, stop)
    chosen_positions = np.concatenate([np.arange(s) for s in stop])
    flat_class_index = order[chosen_ranks, chosen_positions]
    membership[chosen_ranks, flat_class_index] = True
    return membership


def _select_quantile(
    quantile: float | dict[int, float],
    labels: np.ndarray | None,
) -> np.ndarray:
    if isinstance(quantile, dict):
        if labels is None:
            raise ValueError("Mondrian quantiles require labels for each row")
        return np.array([quantile[int(label)] for label in labels], dtype=float)
    return np.full(len(labels) if labels is not None else 0, float(quantile))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def prediction_set_diagnostics(
    membership: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Marginal/class-conditional coverage and prediction-set size statistics."""
    class_idx = true_class_index(labels, classes)
    covered = membership[np.arange(len(labels)), class_idx]
    sizes = membership.sum(axis=1)

    per_class_coverage: dict[int, float] = {}
    per_class_size: dict[int, float] = {}
    for class_label in classes:
        mask = labels == class_label
        if mask.any():
            per_class_coverage[int(class_label)] = float(covered[mask].mean())
            per_class_size[int(class_label)] = float(sizes[mask].mean())

    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "max_set_size": int(sizes.max()),
        "empty_sets": int((sizes == 0).sum()),
        "per_class_coverage": per_class_coverage,
        "per_class_size": per_class_size,
        "covered": covered,
        "sizes": sizes,
    }

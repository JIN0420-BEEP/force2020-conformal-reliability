"""Sanity check: split conformal coverage on exchangeable synthetic data."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from force2020_reliability.evaluation import conformal as cp

ALPHA = 0.1
rng = np.random.default_rng(0)

X, y = make_classification(
    n_samples=12000, n_features=20, n_informative=8, n_classes=10,
    n_clusters_per_class=2, random_state=0,
)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.6, random_state=0)

model = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(X_tr, y_tr)
classes = np.asarray(model.classes_)
p_cal = model.predict_proba(X_cal)
p_test = model.predict_proba(X_test)

# APS with global (marginal) calibration
scores = cp.aps_calibration_scores(p_cal, y_cal, classes,
                                   random_values=rng.random(len(y_cal)))
q = cp.conformal_quantile(scores, ALPHA)
membership = cp.aps_prediction_sets(
    p_test, classes, q, random_values=rng.random(len(y_test)))
diag = cp.prediction_set_diagnostics(membership, y_test, classes)
print(f"APS marginal coverage={diag['coverage']:.3f} (target {1 - ALPHA})")
print(f"APS mean set size={diag['mean_set_size']:.3f}, empty={diag['empty_sets']}")

# Mondrian (class-conditional)
mq = cp.mondrian_quantiles(scores, y_cal, classes, ALPHA)
m_membership = cp.aps_prediction_sets(
    p_test, classes, np.array([mq[int(v)] for v in y_test]),
    random_values=rng.random(len(y_test)))
m_diag = cp.prediction_set_diagnostics(m_membership, y_test, classes)
print(f"Mondrian coverage={m_diag['coverage']:.3f}, "
      f"mean size={m_diag['mean_set_size']:.3f}")
print("class-conditional coverage:",
      {k: round(v, 3) for k, v in m_diag["per_class_coverage"].items()})

# RAPS regularization should shrink large sets without losing coverage
r_membership = cp.aps_prediction_sets(
    p_test, classes, q, random_values=rng.random(len(y_test)),
    raps_penalty=0.01, raps_target=5)
r_diag = cp.prediction_set_diagnostics(r_membership, y_test, classes)
print(f"RAPS coverage={r_diag['coverage']:.3f}, mean size={r_diag['mean_set_size']:.3f}, "
      f"max size={r_diag['max_set_size']}")

assert diag["coverage"] >= 1 - ALPHA - 0.02, "marginal coverage below target"
assert r_diag["coverage"] >= 1 - ALPHA - 0.02, "RAPS coverage below target"
print("OK")

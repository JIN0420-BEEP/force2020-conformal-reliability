from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EncodedClassifier:
    """Encode arbitrary integer labels to the contiguous range LightGBM expects."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, features, labels):
        self.classes_ = np.unique(np.asarray(labels, dtype=int))
        encoded = np.searchsorted(self.classes_, np.asarray(labels, dtype=int))
        self.estimator_ = clone(self.estimator).fit(features, encoded)
        return self

    def predict_proba(self, features):
        return self.estimator_.predict_proba(features)

    def predict(self, features):
        encoded = np.asarray(self.estimator_.predict(features), dtype=int)
        return self.classes_[encoded]


def make_model(name: str, seed: int, hgb_early_stopping: bool = True):
    if name == "hist_gradient_boosting":
        # The full-data and point-CV benchmarks use internal early stopping
        # (10% validation, patience 15, 180-iteration cap). The repeated
        # well-level experiment disables it: its smaller grouped/spatial
        # fitting subsets can contain a rare class only once, which makes the
        # stratified internal validation split fail; a fixed budget is also
        # more reproducible across folds.
        hgb_kwargs = dict(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=hgb_early_stopping,
            random_state=seed,
        )
        if hgb_early_stopping:
            hgb_kwargs.update(validation_fraction=0.1, n_iter_no_change=15)
        return HistGradientBoostingClassifier(**hgb_kwargs)
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        return EncodedClassifier(
            LGBMClassifier(
                n_estimators=350,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
                verbosity=-1,
            )
        )
    if name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    if name == "knn":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
            ]
        )
    raise ValueError(f"Unknown model: {name}")
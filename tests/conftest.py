from __future__ import annotations

import pandas as pd
import pytest

from force2020_reliability.config import ExperimentConfig


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        raw={
            "seed": 42,
            "n_folds": 5,
            "spatial_clusters": 5,
            "spatial_buffer_m": 20_000,
            "min_class_count": 2,
            "min_class_wells": 2,
            "pilot_max_rows_per_well": 10,
            "target_column": "lithology",
            "confidence_column": "confidence",
            "numeric_features": ["x_loc", "y_loc", "value"],
            "curve_columns": ["x_loc", "y_loc", "value", "confidence", "lithology"],
            "models": ["hist_gradient_boosting"],
        }
    )


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rows = []
    row_id = 0
    for well in range(20):
        for depth in range(30):
            label_index = 0 if depth < 12 else (1 if depth < 24 else 2)
            label = (30000, 65000, 65030)[label_index]
            rows.append(
                {
                    "well_id": f"W{well:02d}",
                    "row_id": row_id,
                    "x_loc": 100000.0 + 1000.0 * well,
                    "y_loc": 1000000.0 + 50000.0 * (well // 4),
                    "value": float(depth),
                    "confidence": 1,
                    "lithology": label,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from force2020_reliability.paths import CONFIG_DIR


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def n_folds(self) -> int:
        return int(self.raw["n_folds"])

    @property
    def spatial_clusters(self) -> int:
        return int(self.raw["spatial_clusters"])

    @property
    def spatial_buffer_m(self) -> float:
        return float(self.raw["spatial_buffer_m"])

    @property
    def target_column(self) -> str:
        return str(self.raw["target_column"])

    @property
    def confidence_column(self) -> str:
        return str(self.raw["confidence_column"])

    @property
    def numeric_features(self) -> list[str]:
        return list(self.raw["numeric_features"])

    @property
    def curve_columns(self) -> list[str]:
        return list(self.raw["curve_columns"])

    @property
    def min_class_count(self) -> int:
        return int(self.raw["min_class_count"])

    @property
    def min_class_wells(self) -> int:
        return int(self.raw["min_class_wells"])

    @property
    def pilot_max_rows_per_well(self) -> int:
        return int(self.raw["pilot_max_rows_per_well"])


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else CONFIG_DIR / "pilot.json"
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return ExperimentConfig(raw=raw)
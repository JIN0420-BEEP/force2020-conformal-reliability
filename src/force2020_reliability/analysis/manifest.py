from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from force2020_reliability.config import load_config
from force2020_reliability.paths import FIGURES_DIR, INTERIM_DIR, PROJECT_ROOT, RESULTS_DIR


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _hash_file(path),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_pilot_manifest(
    output_path: Path = RESULTS_DIR / "pilot" / "manifest.json",
) -> Path:
    raw_zip = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"
    )
    key_files = [
        raw_zip,
        PROJECT_ROOT / "data" / "raw" / "NPD_Lithostratigraphy_groups_all_wells.xlsx",
        INTERIM_DIR / "parse_summary.json",
        INTERIM_DIR / "class_counts.csv",
        INTERIM_DIR / "force2020_labeled_strat.parquet",
        INTERIM_DIR / "splits" / "split_summary.json",
        INTERIM_DIR / "splits" / "well_roles.csv",
        INTERIM_DIR / "splits" / "spatial_well_roles.csv",
        RESULTS_DIR / "pilot" / "class_counts.csv",
        RESULTS_DIR / "pilot" / "protocol_model_summary.csv",
        RESULTS_DIR / "pilot" / "analysis" / "fold_metric_summary.csv",
        RESULTS_DIR / "pilot" / "analysis" / "bootstrap_by_well.csv",
        RESULTS_DIR / "pilot" / "analysis" / "distance_correlation.csv",
        RESULTS_DIR / "pilot" / "analysis" / "shift_correlation.csv",
        RESULTS_DIR / "sensitivity" / "summary.csv",
        RESULTS_DIR / "sensitivity" / "fold_metrics.csv",
        RESULTS_DIR / "nested" / "all_metrics.csv",
        RESULTS_DIR / "stage2_analysis" / "sensitivity_summary.csv",
        RESULTS_DIR / "stage2_analysis" / "nested_summary.csv",
        RESULTS_DIR / "stage2_analysis" / "summary.json",
        RESULTS_DIR / "feature_ablation" / "summary.csv",
        RESULTS_DIR / "feature_ablation" / "metadata.json",
        PROJECT_ROOT / "REPORT.md",
        PROJECT_ROOT / "README.md",
    ]
    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    source_files = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    config = load_config()
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study": "FORCE 2020 lithology reliability pilot",
        "data_source": {
            "doi": "10.5281/zenodo.4351156",
            "record_id": 4351156,
        },
        "config": config.raw,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "pyarrow",
                    "lasio",
                    "matplotlib",
                )
            },
        },
        "commands": [
            "uv sync --project research/force2020-reliability",
            "uv run --project research/force2020-reliability force2020 parse --overwrite",
            "uv run --project research/force2020-reliability force2020 splits",
            "uv run --project research/force2020-reliability force2020 pilot",
            "uv run --project research/force2020-reliability force2020 analyze --bootstrap 500",
            "uv run --project research/force2020-reliability force2020 sensitivity --rows-per-well 1000",
            "uv run --project research/force2020-reliability force2020 nested --rows-per-well 1000",
            "uv run --project research/force2020-reliability force2020 stage2",
            "uv run --project research/force2020-reliability force2020 ablation --rows-per-well 1000",
            "uv run pytest  # 8 passed in research/force2020-reliability",
        ],
        "outputs": {
            "key_files": [_file_record(path) for path in key_files if path.exists()],
            "figures": [_file_record(path) for path in figure_files],
            "source_files": [_file_record(path) for path in source_files],
            "test_files": [_file_record(path) for path in test_files],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
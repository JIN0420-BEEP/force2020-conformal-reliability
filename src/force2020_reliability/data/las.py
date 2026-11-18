from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

from force2020_reliability.config import ExperimentConfig
from force2020_reliability.labels import attach_lithology_names
from force2020_reliability.paths import INTERIM_DIR, RAW_DIR

LAS_GLOB = "las/Force_2020_all_wells_train_test_blind_hidden_final/*.las"
WELLS_DIR = INTERIM_DIR / "las_wells"
COMBINED_PATH = INTERIM_DIR / "force2020_labeled.parquet"
AUDIT_PATH = INTERIM_DIR / "data_audit.csv"
METADATA_PATH = INTERIM_DIR / "well_metadata.csv"


@dataclass(frozen=True)
class ParsedWell:
    frame: pd.DataFrame
    audit: dict[str, object]


def discover_las_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    files = sorted((raw_dir / LAS_GLOB).parent.glob("*.las"))
    if not files:
        raise FileNotFoundError(
            f"No LAS files found under {raw_dir}. Extract the official ZIP first."
        )
    return files


def _clean_column_name(value: object) -> str:
    return re.sub(r"\s+", "_", str(value).strip())


def _read_one(path: Path, config: ExperimentConfig, well_id: str) -> ParsedWell:
    las = lasio.read(path, ignore_header_errors=True, mnemonic_case="preserve")
    frame = las.df().reset_index()
    frame.columns = [_clean_column_name(column) for column in frame.columns]

    if "DEPTH_MD" not in frame and "DEPT" in frame:
        frame = frame.rename(columns={"DEPT": "DEPTH_MD"})
    if "x_loc" not in frame and "X_LOC" in frame:
        frame = frame.rename(columns={"X_LOC": "x_loc"})
    if "y_loc" not in frame and "Y_LOC" in frame:
        frame = frame.rename(columns={"Y_LOC": "y_loc"})
    if "z_loc" not in frame and "Z_LOC" in frame:
        frame = frame.rename(columns={"Z_LOC": "z_loc"})

    total_rows = len(frame)
    missing = [column for column in config.curve_columns if column not in frame]
    for column in missing:
        frame[column] = np.nan

    selected = frame.loc[:, config.curve_columns].copy()
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    target = config.target_column
    confidence = config.confidence_column
    label_nonmissing = int(selected[target].notna().sum())
    confidence_nonmissing = int(selected[confidence].notna().sum())
    selected[target] = selected[target].round().astype("Int64")
    selected[confidence] = selected[confidence].round().astype("Int16")
    selected = selected.loc[selected[target].notna()].copy()
    selected[target] = selected[target].astype("int32")
    selected.insert(0, "well_id", well_id)
    selected["row_id"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.reset_index(drop=True)

    x = selected["x_loc"].replace(0, np.nan).dropna()
    y = selected["y_loc"].replace(0, np.nan).dropna()
    x_median = float(x.median()) if not x.empty else np.nan
    y_median = float(y.median()) if not y.empty else np.nan
    common = selected[["x_loc", "y_loc"]].replace(0, np.nan).dropna()
    if len(common) > 1:
        coordinate_spread = float(
            np.sqrt(np.var(common["x_loc"].to_numpy()) + np.var(common["y_loc"].to_numpy()))
        )
    else:
        coordinate_spread = np.nan

    counts = selected[target].value_counts().sort_index()
    audit = {
        "well_id": well_id,
        "source_file": path.name,
        "rows_total": total_rows,
        "rows_labeled": len(selected),
        "label_nonmissing": label_nonmissing,
        "label_missing_rate": 1.0 - label_nonmissing / max(total_rows, 1),
        "confidence_nonmissing": confidence_nonmissing,
        "x_median": x_median,
        "y_median": y_median,
        "coordinate_spread_m": coordinate_spread,
        "n_classes": int(counts.size),
        "min_class_count": int(counts.min()) if not counts.empty else 0,
        "max_class_count": int(counts.max()) if not counts.empty else 0,
    }
    return ParsedWell(frame=selected, audit=audit)


def parse_las_corpus(
    config: ExperimentConfig,
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse LAS files into per-well and combined parquet datasets."""
    WELLS_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_las_files(raw_dir)
    audit_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for index, path in enumerate(files, start=1):
        well_id = path.stem
        well_path = WELLS_DIR / f"{well_id}.parquet"
        audit_path = WELLS_DIR / f"{well_id}.audit.json"
        if well_path.exists() and audit_path.exists() and not overwrite:
            frame = pd.read_parquet(well_path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            parsed = _read_one(path, config, well_id)
            frame = parsed.frame
            audit = parsed.audit
            frame.to_parquet(well_path, index=False)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        audit_rows.append(audit)
        metadata_rows.append(
            {
                "well_id": well_id,
                "x": audit["x_median"],
                "y": audit["y_median"],
                "rows_labeled": audit["rows_labeled"],
            }
        )
        frames.append(frame)
        if index % 10 == 0 or index == len(files):
            print(f"Parsed {index}/{len(files)} wells", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(COMBINED_PATH, index=False)
    audit_frame = pd.DataFrame(audit_rows).sort_values("well_id")
    metadata = pd.DataFrame(metadata_rows).sort_values("well_id")
    audit_frame.to_csv(AUDIT_PATH, index=False, encoding="utf-8-sig")
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    return combined, audit_frame


def summarize_classes(
    frame: pd.DataFrame,
    target_column: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    summary = (
        frame.groupby([target_column, "well_id"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = (
        summary.groupby(target_column, observed=True)["rows"]
        .sum()
        .rename("total_rows")
        .reset_index()
        .sort_values("total_rows", ascending=False)
    )
    wells = (
        summary.groupby(target_column, observed=True)["well_id"]
        .nunique()
        .rename("n_wells")
        .reset_index()
    )
    totals = totals.merge(wells, on=target_column, how="left")
    totals = attach_lithology_names(totals, target_column)
    if output_path is not None:
        totals.to_csv(output_path, index=False, encoding="utf-8-sig")
    return totals


def load_processed(path: Path = COMBINED_PATH, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    return pd.read_parquet(path, columns=columns)


def write_parse_summary(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    config: ExperimentConfig,
    output_path: Path | None = None,
) -> Path:
    destination = output_path or INTERIM_DIR / "parse_summary.json"
    payload = {
        "n_wells": int(frame["well_id"].nunique()),
        "n_labeled_rows": len(frame),
        "n_classes": int(frame[config.target_column].nunique()),
        "wells_with_coordinates": int(audit[["x_median", "y_median"]].notna().all(axis=1).sum()),
        "label_nonmissing_rate": float(
            audit["label_nonmissing"].sum() / max(audit["rows_total"].sum(), 1)
        ),
        "coordinate_spread_p95_m": float(audit["coordinate_spread_m"].quantile(0.95)),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
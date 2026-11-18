from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from force2020_reliability.paths import INTERIM_DIR, RAW_DIR

GROUPS_FILE = "NPD_Lithostratigraphy_groups_all_wells.xlsx"
DEFAULT_OUTPUT = INTERIM_DIR / "force2020_labeled_strat.parquet"


def normalize_well_id(value: object) -> str:
    text = str(value).strip().upper().replace("_", "/")
    return re.sub(r"\s+", " ", text)


def _clean_surface(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+Top\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def load_group_horizons(path: Path = RAW_DIR / GROUPS_FILE) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0, usecols="A:H")
    frame = frame.rename(
        columns={
            "Well identifier": "well_id_raw",
            "Surface": "surface",
            "MD": "md",
        }
    )
    frame = frame.loc[:, ["well_id_raw", "surface", "md"]].copy()
    frame["well_id"] = frame["well_id_raw"].map(normalize_well_id)
    frame["strat_group"] = frame["surface"].map(_clean_surface)
    frame["md"] = pd.to_numeric(frame["md"], errors="coerce")
    frame = frame.dropna(subset=["well_id", "strat_group", "md"])
    frame = frame.sort_values(["well_id", "md"]).drop_duplicates(["well_id", "md"])
    return frame.loc[:, ["well_id", "md", "strat_group"]].reset_index(drop=True)


def assign_group_by_depth(
    frame: pd.DataFrame,
    horizons: pd.DataFrame | None = None,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = load_group_horizons() if horizons is None else horizons.copy()
    left = frame.copy()
    left["well_id_norm"] = left["well_id"].map(normalize_well_id)
    horizons["well_id_norm"] = horizons["well_id"].map(normalize_well_id)
    right = horizons.rename(columns={"md": "horizon_md"})

    assigned_groups: list[pd.DataFrame] = []
    for well_id, well_frame in left.groupby("well_id_norm", sort=False, observed=True):
        well_horizons = right.loc[right["well_id_norm"] == well_id].sort_values("horizon_md")
        missing_depth = well_frame["DEPTH_MD"].isna()
        valid_depth = well_frame.loc[~missing_depth].sort_values("DEPTH_MD")
        if well_horizons.empty or valid_depth.empty:
            valid_assigned = valid_depth.copy()
            valid_assigned["strat_group"] = "UNASSIGNED"
        else:
            valid_assigned = pd.merge_asof(
                valid_depth,
                well_horizons[["horizon_md", "strat_group"]],
                left_on="DEPTH_MD",
                right_on="horizon_md",
                direction="backward",
                allow_exact_matches=True,
            )
        missing_assigned = well_frame.loc[missing_depth].copy()
        missing_assigned["strat_group"] = "UNASSIGNED"
        assigned_groups.extend([valid_assigned, missing_assigned])

    assigned = pd.concat(assigned_groups, ignore_index=False)
    assigned["strat_group"] = assigned["strat_group"].fillna("UNASSIGNED")
    assigned = assigned.drop(columns=["well_id_norm"]).sort_index()
    assigned.to_parquet(output_path, index=False)

    coverage = (
        assigned.groupby(["well_id", "strat_group"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    coverage.to_csv(
        INTERIM_DIR / "well_strat_group_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return assigned, coverage

def load_stratified(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stratigraphy-enriched data not found: {path}")
    return pd.read_parquet(path)
from __future__ import annotations

import pandas as pd

LITHOLOGY_NAMES = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}


def lithology_name(code: int) -> str:
    return LITHOLOGY_NAMES.get(int(code), f"Unknown ({int(code)})")


def attach_lithology_names(frame: pd.DataFrame, code_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["lithology_name"] = result[code_column].astype(int).map(LITHOLOGY_NAMES)
    missing = result["lithology_name"].isna()
    if missing.any():
        codes = sorted(result.loc[missing, code_column].unique().tolist())
        raise ValueError(f"Unknown FORCE lithology codes: {codes}")
    return result
"""Combine multi-seed continuous-section (segment) evaluations.

Globs ``results/segments_seed*/well_segment_metrics.csv`` and reports, per
protocol/model, equal-well means alongside point/depth-pair pooled values.
Between-seed SD uses the three seed summaries; wells and folds within a seed
are not independent geographic replicates.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from force2020_reliability.paths import RESULTS_DIR


def _weighted(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def main() -> None:
    frames = []
    for directory in sorted(RESULTS_DIR.glob("segments_seed*")):
        if not re.fullmatch(r"segments_seed\d+", directory.name):
            continue
        path = directory / "well_segment_metrics.csv"
        if path.exists():
            table = pd.read_csv(path)
            table["source_dir"] = directory.name
            frames.append(table)
    if not frames:
        raise FileNotFoundError("No segments_seed*/well_segment_metrics.csv found")
    wells = pd.concat(frames, ignore_index=True)

    seed_rows = []
    for (protocol, model, seed), group in wells.groupby(
        ["protocol", "model", "seed"], observed=True
    ):
        seed_rows.append({
            "protocol": protocol,
            "model": model,
            "seed": seed,
            "n_well_rows": len(group),
            "point_acc_equal": group["point_accuracy"].mean(),
            "point_acc_pooled": _weighted(group["point_accuracy"], group["n_rows"]),
            "pred_agree_equal": group["pred_adjacent_agreement"].mean(),
            "pred_agree_pooled": _weighted(
                group["pred_adjacent_agreement"], group["adjacent_pairs"]),
            "true_agree_equal": group["true_adjacent_agreement"].mean(),
            "boundary_f1": group["boundary_f1"].mean(),
            "boundary_precision": group["boundary_precision"].mean(),
            "boundary_recall": group["boundary_recall"].mean(),
            "boundary_mae_m": group["boundary_mae_m"].mean(),
            "thickness_rel_error": group["thickness_relative_error"].mean(),
            "segment_count_abs_error": group["segment_count_abs_error"].mean(),
        })
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(
        RESULTS_DIR / "segments_multiseed_seed_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    metric_cols = [c for c in seed_summary.columns
                   if c not in ("protocol", "model", "seed", "n_well_rows")]
    overall = (
        seed_summary.groupby(["protocol", "model"], observed=True)[metric_cols]
        .agg(["mean", "std"])
    )
    overall.columns = [f"{c}_{stat}" for c, stat in overall.columns]
    overall = overall.reset_index()
    overall["n_seeds"] = seed_summary.groupby(
        ["protocol", "model"], observed=True)["seed"].nunique().to_numpy()
    overall.to_csv(
        RESULTS_DIR / "segments_multiseed_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    show_cols = [
        "protocol", "model", "n_seeds",
        "point_acc_equal_mean", "point_acc_pooled_mean",
        "pred_agree_equal_mean", "boundary_f1_mean",
        "boundary_mae_m_mean", "thickness_rel_error_mean",
        "segment_count_abs_error_mean",
    ]
    print(overall[show_cols].round(4).to_string(index=False))
    print("\nsaved:", RESULTS_DIR / "segments_multiseed_summary.csv")


if __name__ == "__main__":
    main()

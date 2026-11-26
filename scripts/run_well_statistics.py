"""Cluster-robust inference at the well/block level.

Replaces the pseudo-independent t-tests in run_statistics.py. Resampling units
are wells (for marginal inference) and spatial blocks (for the spatial
protocol), not depth points or folds that share wells across seeds.

The MMD-coverage association is evaluated *within* a protocol at the
seed x fold level (n = 15 for three seeds x five folds). These 15 rows are NOT
15 independent geographic samples: the three seeds reuse the same 118 wells
and the same five/six spatial blocks, so their Spearman p-values are nominal
and descriptive only and must not be cited as significance evidence. Pooling
the six model families repeats the same fold-level MMD six times, so that
pooling is reported only as a secondary per-model consistency check. The
closest analysis to the independent sampling unit is the per-seed block-level
rank correlation (n = 5 blocks for the primary design), reported separately in
``block_level_correlation.csv`` and likewise treated as descriptive and
low-powered.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from force2020_reliability.paths import RESULTS_DIR

N_BOOT = 2999
NOMINAL = 0.90
BOOT_RNG = np.random.default_rng(20261119)


def bootstrap_ci(values, n_boot=N_BOOT, rng=BOOT_RNG):
    values = np.asarray(values, dtype=float)
    n = len(values)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def well_unit_inference(well_table):
    """Equal-weight one value per well, averaged over folds and seeds."""
    rows = []
    for (model, protocol), group in well_table.groupby(["model", "protocol"]):
        per_well = group.groupby("well_id").agg(
            coverage=("coverage", "mean"),
            point_accuracy=("point_accuracy", "mean"),
            mean_set_size=("mean_set_size", "mean"),
        )
        cov = per_well["coverage"].to_numpy()
        ci_lo, ci_hi = bootstrap_ci(cov)
        t_stat, p_value = stats.ttest_1samp(cov, NOMINAL)
        rows.append({
            "model": model,
            "protocol": protocol,
            "n_wells": len(cov),
            "coverage_mean": round(float(cov.mean()), 4),
            "coverage_ci_lo": round(ci_lo, 4),
            "coverage_ci_hi": round(ci_hi, 4),
            "p_vs_nominal": round(float(p_value), 5),
            "t_stat": round(float(t_stat), 3),
            "point_accuracy_mean": round(
                float(per_well["point_accuracy"].mean()), 4),
            "set_size_mean": round(
                float(per_well["mean_set_size"].mean()), 4),
        })
    return pd.DataFrame(rows)


def fold_level_performance(fold_table):
    """Mean over seed x fold observations; macro-F1 lives at fold level."""
    agg = {
        "coverage": ["mean", "std"],
        "point_accuracy": ["mean", "std"],
        "mean_set_size": ["mean", "std"],
    }
    if "macro_f1" in fold_table.columns:
        agg["macro_f1"] = ["mean", "std"]
    grouped = fold_table.groupby(["model", "protocol"]).agg(agg)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    for col in grouped.columns:
        if col.endswith(("_mean", "_std")):
            grouped[col] = grouped[col].round(4)
    return grouped


def block_level_summary(fold_table):
    """Spatial held-out blocks: cross-model mean and HGB-specific values."""
    spatial = fold_table[fold_table["protocol"] == "spatial_block"]

    cross = spatial.groupby(["seed", "fold"]).agg(
        block_coverage=("coverage", "mean"),
        block_accuracy=("point_accuracy", "mean"),
    ).reset_index()

    hgb = spatial[spatial["model"] == "hist_gradient_boosting"]
    hgb_fold = hgb.groupby(["seed", "fold"]).agg(
        hgb_coverage=("coverage", "mean"),
        hgb_accuracy=("point_accuracy", "mean"),
    ).reset_index()

    blocks = cross.merge(hgb_fold, on=["seed", "fold"], how="left")

    summary = blocks.groupby("seed").agg(
        n_blocks=("fold", "nunique"),
        block_cov_mean=("block_coverage", "mean"),
        block_cov_min=("block_coverage", "min"),
        block_cov_max=("block_coverage", "max"),
        block_cov_sd=("block_coverage", "std"),
        hgb_cov_mean=("hgb_coverage", "mean"),
        hgb_cov_min=("hgb_coverage", "min"),
    ).reset_index().round(4)
    return blocks.round(4), summary


def _corr_row(protocol, aggregation, sub):
    rho, p = stats.spearmanr(sub["mmd"], sub["coverage"])
    return {
        "protocol": protocol,
        "aggregation": aggregation,
        "n_obs": len(sub),
        "spearman_mmd_coverage": round(float(rho), 3),
        "p_value_nominal": round(float(p), 4),
        "note": ("descriptive seed x fold rank association; the 15 rows share "
                 "the same wells and blocks across seeds, so p_value_nominal "
                 "is not an independent-sample significance test; observational"),
    }


def block_level_correlation(block_table, shift_table, scheme="blocks5_buffer20"):
    """Rank association at the independent geographic unit (held-out block).

    There are only five blocks per seed for the primary design (six in the
    robustness design), and the same geography is reused across seeds. We
    therefore report per-seed correlations rather than pooling seeds into a
    larger, pseudo-replicated sample, and treat them as descriptive.
    """
    spatial_shift = shift_table[
        (shift_table["protocol"] == "spatial_block")
        & (shift_table["scheme"] == scheme)
    ]
    merged = block_table.merge(
        spatial_shift[["seed", "fold", "mmd"]], on=["seed", "fold"], how="inner")
    rows = []
    for seed, group in merged.groupby("seed"):
        for column, aggregation in (
            ("block_coverage", "cross_model_mean"),
            ("hgb_coverage", "hist_gradient_boosting"),
        ):
            if len(group) >= 3:
                rho, p = stats.spearmanr(group["mmd"], group[column])
                rows.append({
                    "seed": seed,
                    "aggregation": aggregation,
                    "n_blocks": len(group),
                    "spearman_mmd_coverage": round(float(rho), 3),
                    "p_value_nominal": round(float(p), 4),
                    "note": ("one value per held-out spatial block; n=5 per "
                             "seed; descriptive and low-powered; blocks reused "
                             "across seeds; do not pool into n=15"),
                })
    return pd.DataFrame(rows)


def within_protocol_correlation(fold_table, shift_table):
    keys = ["protocol", "fold", "scheme", "seed"]
    rows = []

    # Primary: average coverage across model families per seed x fold.
    fold_mean = (
        fold_table.groupby(keys, as_index=False)
        .agg(coverage=("coverage", "mean"))
    )
    merged_mean = fold_mean.merge(shift_table, on=keys, how="inner")
    for protocol in ("well_grouped", "spatial_block"):
        sub = merged_mean[merged_mean["protocol"] == protocol]
        if len(sub) >= 3:
            rows.append(_corr_row(protocol, "cross_model_mean", sub))

    # Primary single family: HGB.
    hgb = fold_table[fold_table["model"] == "hist_gradient_boosting"]
    merged_hgb = hgb.merge(shift_table, on=keys, how="inner")
    for protocol in ("well_grouped", "spatial_block"):
        sub = merged_hgb[merged_hgb["protocol"] == protocol]
        if len(sub) >= 3:
            rows.append(_corr_row(protocol, "hist_gradient_boosting", sub))

    # Secondary: per-model consistency (each n = 15).
    for model in sorted(fold_table["model"].unique()):
        model_fold = fold_table[fold_table["model"] == model]
        merged = model_fold.merge(shift_table, on=keys, how="inner")
        for protocol in ("well_grouped", "spatial_block"):
            sub = merged[merged["protocol"] == protocol]
            if len(sub) >= 3:
                rows.append(_corr_row(protocol, f"model:{model}", sub))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+",
                        default=["well_level_seed1", "well_level_seed2",
                                 "well_level_seed3"])
    parser.add_argument("--output", default="well_level")
    args = parser.parse_args()

    well_tables, fold_tables = [], []
    for directory in args.result_dirs:
        base = RESULTS_DIR / directory
        well_tables.append(pd.read_csv(base / "well_metrics.csv"))
        fold_tables.append(pd.read_csv(base / "fold_metrics.csv"))
    well_table = pd.concat(well_tables, ignore_index=True)
    fold_table = pd.concat(fold_tables, ignore_index=True)

    inference = well_unit_inference(well_table)
    performance = fold_level_performance(fold_table)
    blocks, block_summary = block_level_summary(fold_table)

    out = RESULTS_DIR / args.output
    out.mkdir(parents=True, exist_ok=True)

    inference.to_csv(out / "well_unit_inference.csv", index=False,
                     encoding="utf-8-sig")
    performance.to_csv(out / "fold_level_performance.csv", index=False,
                       encoding="utf-8-sig")
    blocks.to_csv(out / "block_level_observations.csv", index=False,
                  encoding="utf-8-sig")
    block_summary.to_csv(out / "block_level_summary.csv", index=False,
                         encoding="utf-8-sig")

    shift_path = out / "shift_long.csv"
    if shift_path.exists():
        shift_table = pd.read_csv(shift_path)
        correlations = within_protocol_correlation(fold_table, shift_table)
        correlations.to_csv(out / "within_protocol_correlation.csv",
                            index=False, encoding="utf-8-sig")
        block_corr = block_level_correlation(blocks, shift_table)
        block_corr.to_csv(out / "block_level_correlation.csv",
                          index=False, encoding="utf-8-sig")
    else:
        correlations = pd.DataFrame()
        block_corr = pd.DataFrame()

    print("=== well-unit inference ===")
    print(inference.to_string(index=False))
    print("\n=== fold-level performance ===")
    print(performance.to_string(index=False))
    print("\n=== block-level summary ===")
    print(block_summary.to_string(index=False))
    if not correlations.empty:
        print("\n=== within-protocol correlations (descriptive; n=15 "
              "seed x fold shares wells/blocks, nominal p not inferential) ===")
        print(correlations[correlations["aggregation"].isin(
            ["cross_model_mean", "hist_gradient_boosting"])].to_string(index=False))
    if not block_corr.empty:
        print("\n=== block-level correlations (independent geographic units, "
              "n=5 per seed; descriptive) ===")
        print(block_corr.to_string(index=False))


if __name__ == "__main__":
    main()

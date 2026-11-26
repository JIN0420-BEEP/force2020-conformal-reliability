"""Aggregate the feature-source ablation (4 sets x 3 seeds x HGB).

Reads ``results/ablation_<set>_seed<k>/{fold,well}_metrics.csv`` produced by
``run_well_level.py --feature-set`` and writes:

* ``results/ablation/feature_ablation_summary.csv``  - fold-pooled means/SDs
* ``results/ablation/feature_ablation_seed_summary.csv`` - per-seed means
* ``results/ablation/feature_ablation_equal_well.csv`` - equal-well coverage
* ``figures/17_feature_ablation.{png,pdf}``

Coverage and set size are empirical: calibration units are depth points and
within-well dependence is retained; these summaries carry no cluster-level
finite-sample guarantee.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force2020_reliability.analysis.plots import PROTOCOL_COLORS, PROTOCOL_LABELS
from force2020_reliability.paths import FIGURES_DIR, RESULTS_DIR

FEATURE_ORDER = ["logs_only", "logs_depth", "logs_coords", "full"]
FEATURE_LABELS = {
    "logs_only": "Wireline curves (14)",
    "logs_depth": "Curves + depth (15)",
    "logs_coords": "Curves + x/y/z (17)",
    "full": "All features (18)",
}
PROTOCOL_ORDER = ["random_point", "well_grouped", "spatial_block"]


def _load(pattern_name: str) -> pd.DataFrame:
    frames = []
    for directory in sorted(RESULTS_DIR.glob("ablation_*_seed*")):
        match = re.fullmatch(r"ablation_([a-z_]+)_seed(\d+)", directory.name)
        if not match:
            continue
        path = directory / pattern_name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "feature_set" not in table.columns:
            table["feature_set"] = match.group(1)
        if "seed_index" not in table.columns:
            table["seed_index"] = int(match.group(2))
        frames.append(table)
    if not frames:
        raise FileNotFoundError(
            "No ablation_*_seed*/fold_metrics.csv found under results/"
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_dir = RESULTS_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = _load("fold_metrics.csv")
    wells = _load("well_metrics.csv")

    folds = folds[folds["model"] == "hist_gradient_boosting"].copy()
    folds["feature_set"] = pd.Categorical(
        folds["feature_set"], FEATURE_ORDER, ordered=True
    )
    folds["protocol"] = pd.Categorical(
        folds["protocol"], PROTOCOL_ORDER, ordered=True
    )

    # (fold-level column, seed-summary column, output short name)
    metric_specs = [
        ("macro_f1", "macro_f1", "macro_f1"),
        ("coverage", "coverage", "coverage"),
        ("mean_set_size", "set_size", "set_size"),
        ("point_accuracy", "point_accuracy", "point_acc"),
    ]

    # Per-seed means over the five folds (the independent repeat is the seed
    # partition; folds share wells/blocks, so do not treat 15 rows as 15
    # independent geographic samples).
    seed_summary = (
        folds.groupby(["feature_set", "protocol", "seed", "seed_index"],
                      observed=True)
        .agg(
            macro_f1=("macro_f1", "mean"),
            coverage=("coverage", "mean"),
            set_size=("mean_set_size", "mean"),
            point_accuracy=("point_accuracy", "mean"),
        )
        .reset_index()
    )
    seed_summary.to_csv(
        out_dir / "feature_ablation_seed_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    rows = []
    for (feature_set, protocol), group in folds.groupby(
        ["feature_set", "protocol"], observed=True
    ):
        seed_group = seed_summary[
            (seed_summary["feature_set"] == feature_set)
            & (seed_summary["protocol"] == protocol)
        ]
        row = {
            "feature_set": str(feature_set),
            "n_features": int(group["n_features"].iloc[0]),
            "protocol": str(protocol),
            "n_folds": len(group),
            "n_seeds": seed_group["seed"].nunique(),
        }
        for fold_col, seed_col, short in metric_specs:
            row[f"{short}_mean"] = float(group[fold_col].mean())
            row[f"{short}_fold_sd"] = float(group[fold_col].std(ddof=1))
            row[f"{short}_seed_sd"] = float(seed_group[seed_col].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        out_dir / "feature_ablation_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    # Equal-well aggregation: average each well's coverage/set size within a
    # seed, then across seeds (wells recur across seeds -> descriptive only).
    equal_rows = []
    for (feature_set, protocol, seed), group in wells.groupby(
        ["feature_set", "protocol", "seed"], observed=True
    ):
        equal_rows.append({
            "feature_set": feature_set,
            "protocol": protocol,
            "seed": seed,
            "equal_well_coverage": group["coverage"].mean(),
            "equal_well_set_size": group["mean_set_size"].mean(),
            "equal_well_accuracy": group["point_accuracy"].mean(),
        })
    equal_seed = pd.DataFrame(equal_rows)
    equal_summary = (
        equal_seed.groupby(["feature_set", "protocol"], observed=True)
        .agg(
            equal_well_coverage_mean=("equal_well_coverage", "mean"),
            equal_well_coverage_seed_sd=("equal_well_coverage", "std"),
            equal_well_set_size_mean=("equal_well_set_size", "mean"),
            equal_well_accuracy_mean=("equal_well_accuracy", "mean"),
        )
        .reset_index()
    )
    equal_summary.to_csv(
        out_dir / "feature_ablation_equal_well.csv",
        index=False, encoding="utf-8-sig",
    )

    pivot = summary.pivot_table(
        index="feature_set", columns="protocol",
        values=["macro_f1_mean", "coverage_mean", "set_size_mean"],
        observed=True,
    )
    print(pivot.round(4).to_string())
    print("\nsaved tables to", out_dir)

    # ---------------- figure ----------------
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })
    panels = [
        ("macro_f1_mean", "macro_f1_seed_sd", "Macro-F1", (0.0, None)),
        ("coverage_mean", "coverage_seed_sd", "Empirical coverage (90% target)",
         (0.5, 1.0)),
        ("set_size_mean", "set_size_seed_sd", "Mean set size", (0.0, None)),
    ]
    x = np.arange(len(PROTOCOL_ORDER))
    width = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    for ax, (value_col, sd_col, title, ylim) in zip(axes, panels):
        for j, feature_set in enumerate(FEATURE_ORDER):
            sub = summary[summary["feature_set"] == feature_set].set_index(
                "protocol").reindex(PROTOCOL_ORDER)
            offset = (j - 1.5) * width
            ax.bar(
                x + offset, sub[value_col], width,
                yerr=sub[sd_col].fillna(0.0), capsize=2.5,
                label=FEATURE_LABELS[feature_set],
                color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"][j],
                edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
            )
        if value_col == "coverage_mean":
            ax.axhline(0.90, color="#222222", linestyle="--", linewidth=1.0)
            ax.text(2.45, 0.905, "0.90", fontsize=8, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([
            PROTOCOL_LABELS.get(p, p).replace(" ", "\n", 1)
            for p in PROTOCOL_ORDER
        ])
        ax.set_title(title, fontsize=10)
        if ylim[0] is not None:
            ax.set_ylim(bottom=ylim[0])
        if ylim[1] is not None:
            ax.set_ylim(top=ylim[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig_dir = FIGURES_DIR
    fig_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"17_feature_ablation.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print("saved figure to", fig_dir / "17_feature_ablation.png")


if __name__ == "__main__":
    main()

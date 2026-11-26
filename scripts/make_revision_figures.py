"""Review-response figures (revision).

1. ``11_coverage_vs_mmd.png``: descriptive within-protocol seed x fold
   association between MMD shift and coverage (n=15 rows per protocol, but
   wells/blocks are reused across seeds), replacing the old pooled plot that
   mixed protocols and treated repeated seeds/models as independent. Nominal
   p-values are deliberately not shown; independent geographic inference has
   only 5-6 blocks (see block_level_correlation.csv).
2. ``15_multimodel_well_ci.png``: equal-well bootstrap coverage CIs for six
   learners under three protocols.
3. ``16_robustness_macro_f1.png``: blocks5/20 km vs blocks6/30 km robustness
   and class-balanced macro-F1.

All numbers are read from the corrected CSVs in results/well_level.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force2020_reliability.analysis.plots import PROTOCOL_COLORS, PROTOCOL_LABELS
from force2020_reliability.paths import FIGURES_DIR, RESULTS_DIR

WL = RESULTS_DIR / "well_level"
SEED_DIRS = {
    20261021: "well_level_seed1",
    20261022: "well_level_seed2",
    20261023: "well_level_seed3",
}
TARGET = 0.90

MODEL_LABELS = {
    "hist_gradient_boosting": "HGB",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "extra_trees": "Extra trees",
    "logistic": "Logistic",
    "knn": "k-NN",
}
MODEL_ORDER = [
    "hist_gradient_boosting", "lightgbm", "random_forest",
    "extra_trees", "logistic", "knn",
]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    })


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _load_b5_folds() -> pd.DataFrame:
    frames = []
    for seed, directory in SEED_DIRS.items():
        table = pd.read_csv(RESULTS_DIR / directory / "fold_metrics.csv")
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def plot_within_protocol_shift(folds: pd.DataFrame, corr: pd.DataFrame) -> None:
    shift = pd.read_csv(WL / "shift_long.csv")
    hgb = folds[folds["model"] == "hist_gradient_boosting"][
        ["seed", "protocol", "fold", "coverage"]]
    mean_cov = (folds.groupby(["seed", "protocol", "fold"])["coverage"]
                .mean().reset_index().rename(columns={"coverage": "mean_coverage"}))
    merged = shift.merge(hgb, on=["seed", "protocol", "fold"]).merge(
        mean_cov, on=["seed", "protocol", "fold"])

    panels = ["well_grouped", "spatial_block"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, protocol in zip(axes, panels):
        sub = merged[merged["protocol"] == protocol]
        color = PROTOCOL_COLORS[protocol]
        ax.scatter(sub["mmd"], sub["coverage"], s=30, color=color,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label="HGB, seed$\\times$fold")
        ax.scatter(sub["mmd"], sub["mean_coverage"], s=34, marker="x",
                   color="0.25", linewidth=1.1, zorder=3,
                   label="6-model mean")
        # HGB linear trend for visual direction only.
        coef = np.polyfit(sub["mmd"], sub["coverage"], 1)
        xs = np.linspace(sub["mmd"].min(), sub["mmd"].max(), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color=color, lw=1.2,
                alpha=0.7, zorder=2)
        ax.axhline(TARGET, color="0.15", ls="--", lw=1.0, zorder=1)

        def rho(agg):
            row = corr[(corr["protocol"] == protocol)
                       & (corr["aggregation"] == agg)].iloc[0]
            return row["spearman_mmd_coverage"]

        r_h = rho("hist_gradient_boosting")
        r_m = rho("cross_model_mean")
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set_xlabel("Covariate shift (MMD$^2$, RBF)")
        ax.text(0.04, 0.06,
                f"HGB $\\rho$={r_h:.2f}; 6-model mean $\\rho$={r_m:.2f}\n"
                "n=15 seed$\\times$fold; descriptive only\n"
                "(wells/blocks reused across seeds)",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="0.8", alpha=0.9))
    axes[0].set_ylabel("APS coverage")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "11_coverage_vs_mmd.png")


def plot_model_coverage_ci() -> None:
    inf = pd.read_csv(WL / "well_unit_inference.csv")
    protocols = ["random_point", "well_grouped", "spatial_block"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.2), sharey=True)
    y = np.arange(len(MODEL_ORDER))[::-1]
    for ax, protocol in zip(axes, protocols):
        sub = inf[inf["protocol"] == protocol].set_index("model")
        for yi, model in zip(y, MODEL_ORDER):
            row = sub.loc[model]
            lo, hi = row["coverage_ci_lo"], row["coverage_ci_hi"]
            mean = row["coverage_mean"]
            if hi < TARGET:
                color = "#D55E00"       # significantly below target
            elif lo > TARGET:
                color = "#0072B2"       # conservative / above target
            else:
                color = "0.45"          # CI includes target
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, zorder=2)
            ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=color, lw=1.6)
            ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=color, lw=1.6)
            ax.scatter([mean], [yi], s=34, color=color, zorder=3,
                       edgecolor="white", linewidth=0.5)
            ax.text(hi + 0.004, yi, f"{mean:.3f}", va="center",
                    fontsize=7.5, color=color)
        ax.axvline(TARGET, color="0.15", ls="--", lw=1.0)
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set_xlim(0.82, 1.0)
        ax.set_xlabel("Well-equal coverage (95% bootstrap CI)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    fig.tight_layout()
    _save(fig, "15_multimodel_well_ci.png")


def plot_robustness_f1() -> None:
    rob = pd.read_csv(WL / "spatial_scheme_robustness.csv")
    perf = pd.read_csv(WL / "fold_level_performance.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.1))

    # Panel A: spatial coverage under two block schemes.
    x = np.arange(len(MODEL_ORDER))
    width = 0.38
    b5 = rob[rob["scheme_label"] == "blocks5_buffer20"].set_index("model")
    b6 = rob[rob["scheme_label"] == "blocks6_buffer30"].set_index("model")
    ax1.bar(x - width / 2, [b5.loc[m, "coverage_mean"] for m in MODEL_ORDER],
            width, yerr=[b5.loc[m, "coverage_std"] for m in MODEL_ORDER],
            capsize=2.5, color="#009E73", alpha=0.85, label="5 blocks / 20 km",
            error_kw={"lw": 0.8, "ecolor": "0.3"})
    ax1.bar(x + width / 2, [b6.loc[m, "coverage_mean"] for m in MODEL_ORDER],
            width, yerr=[b6.loc[m, "coverage_std"] for m in MODEL_ORDER],
            capsize=2.5, color="#CC79A7", alpha=0.85, label="6 blocks / 30 km",
            error_kw={"lw": 0.8, "ecolor": "0.3"})
    ax1.axhline(TARGET, color="0.15", ls="--", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],
                        rotation=25, ha="right")
    ax1.set_ylabel("Spatial-block APS coverage")
    ax1.set_ylim(0.80, 1.01)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("(a) Partition robustness")

    # Panel B: HGB macro-F1 across protocols (b5) + b6 spatial.
    hgb = perf[(perf["model"] == "hist_gradient_boosting")].set_index("protocol")
    labels = ["Random\npoint", "Grouped\nwell", "Spatial\n5-block", "Spatial\n6-block"]
    means = [hgb.loc["random_point", "macro_f1_mean"],
             hgb.loc["well_grouped", "macro_f1_mean"],
             hgb.loc["spatial_block", "macro_f1_mean"],
             b6.loc["hist_gradient_boosting", "macro_f1_mean"]]
    sds = [hgb.loc["random_point", "macro_f1_std"],
           hgb.loc["well_grouped", "macro_f1_std"],
           hgb.loc["spatial_block", "macro_f1_std"],
           b6.loc["hist_gradient_boosting", "macro_f1_std"]]
    colors = [PROTOCOL_COLORS["random_point"], PROTOCOL_COLORS["well_grouped"],
              PROTOCOL_COLORS["spatial_block"], "#CC79A7"]
    ax2.bar(np.arange(4), means, yerr=sds, capsize=3, color=colors, alpha=0.88,
            error_kw={"lw": 0.9, "ecolor": "0.3"})
    for i, v in enumerate(means):
        ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Macro-F1 (HGB, nested test folds)")
    ax2.set_ylim(0, 0.82)
    ax2.set_title("(b) Class-balanced performance collapse")
    fig.tight_layout()
    _save(fig, "16_robustness_macro_f1.png")


def main() -> None:
    _style()
    folds = _load_b5_folds()
    corr = pd.read_csv(WL / "within_protocol_correlation.csv")
    plot_within_protocol_shift(folds, corr)
    plot_model_coverage_ci()
    plot_robustness_f1()


if __name__ == "__main__":
    main()

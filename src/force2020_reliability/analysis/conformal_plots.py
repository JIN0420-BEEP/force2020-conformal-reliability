"""Publication-quality figures for the conformal-prediction reliability study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force2020_reliability.analysis.plots import PROTOCOL_COLORS, PROTOCOL_LABELS
from force2020_reliability.paths import FIGURES_DIR, RESULTS_DIR

PROTOCOLS = ("random_point", "well_grouped", "spatial_block")
METHODS = ("aps", "mondrian", "raps", "temp_aps")
METHOD_LABELS = {
    "aps": "Marginal APS",
    "mondrian": "Mondrian APS",
    "raps": "RAPS",
    "temp_aps": "Temperature + APS",
}
TARGET = 0.90


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage_degradation(
    fold_metrics: pd.DataFrame,
    output: Path = FIGURES_DIR / "10_coverage_degradation.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.18
    x = np.arange(len(PROTOCOLS))
    for i, method in enumerate(METHODS):
        means, sds = [], []
        for protocol in PROTOCOLS:
            subset = fold_metrics.loc[
                (fold_metrics["protocol"] == protocol)
                & (fold_metrics["method"] == method), "coverage"]
            means.append(subset.mean())
            sds.append(subset.std(ddof=1))
        offset = (i - 1.5) * width
        ax.bar(
            x + offset, means, width=width, yerr=sds, capsize=2.5,
            label=METHOD_LABELS[method], alpha=0.9,
            error_kw={"linewidth": 0.8, "ecolor": "0.25"},
        )
    ax.axhline(TARGET, color="0.15", linestyle="--", linewidth=1.1,
               label="Nominal 90% coverage")
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOLS])
    ax.set_ylabel("Empirical coverage of prediction sets")
    ax.set_ylim(0.75, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, output)


def plot_coverage_vs_shift(
    fold_metrics: pd.DataFrame,
    shift: pd.DataFrame,
    output: Path = FIGURES_DIR / "11_coverage_vs_mmd.png",
) -> Path:
    aps = fold_metrics.loc[fold_metrics["method"] == "aps"].copy()
    merged = aps.merge(shift, on=["protocol", "fold"])
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for protocol in PROTOCOLS:
        subset = merged.loc[merged["protocol"] == protocol]
        ax.scatter(
            subset["mmd"], subset["coverage"], s=34,
            color=PROTOCOL_COLORS[protocol], edgecolor="white", linewidth=0.5,
            label=PROTOCOL_LABELS[protocol], zorder=3,
        )
    ax.axhline(TARGET, color="0.15", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Covariate shift (squared MMD, RBF kernel)")
    ax.set_ylabel("Empirical coverage (marginal APS)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, output)


def _per_class_frame(fold_metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    records = []
    for protocol in PROTOCOLS:
        for _, row in fold_metrics.loc[
            (fold_metrics["protocol"] == protocol)
            & (fold_metrics["method"] == method)].iterrows():
            coverage = json.loads(row["per_class_coverage"])
            for code, value in coverage.items():
                records.append(
                    {"protocol": protocol, "code": int(code), "coverage": value})
    frame = pd.DataFrame(records)
    return frame.groupby(["protocol", "code"], observed=True)["coverage"].mean().unstack(0)


def plot_class_conditional_coverage(
    fold_metrics: pd.DataFrame,
    output: Path = FIGURES_DIR / "12_class_conditional.png",
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, method in zip(axes, ("aps", "mondrian"), strict=True):
        pivot = _per_class_frame(fold_metrics, method).loc[:, list(PROTOCOLS)]
        image = ax.imshow(
            pivot.to_numpy(), vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(PROTOCOLS)))
        ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOLS],
                           rotation=20, ha="right")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(False)
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                value = pivot.iloc[r, c]
                if pd.notna(value):
                    ax.text(c, r, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < 0.62 else "black", fontsize=7)
    axes[0].set_yticks(np.arange(pivot.shape[0]))
    axes[0].set_yticklabels([str(code) for code in pivot.index])
    axes[0].set_ylabel("Lithology class code")
    fig.subplots_adjust(left=0.10, right=0.90, wspace=0.08)
    cbar_axis = fig.add_axes([0.92, 0.22, 0.02, 0.66])
    fig.colorbar(image, cax=cbar_axis, label="Class-conditional coverage")
    return _save(fig, output)


def plot_set_size_and_calibration(
    fold_metrics: pd.DataFrame,
    output: Path = FIGURES_DIR / "13_setsize_ece.png",
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    x = np.arange(len(PROTOCOLS))
    width = 0.35
    for ax, metric, ylabel in zip(
        axes, ("mean_set_size", "ece"),
        ("Mean prediction-set size", "Expected calibration error"), strict=True):
        for i, method in enumerate(("aps", "temp_aps")):
            values = [
                fold_metrics.loc[
                    (fold_metrics["protocol"] == p)
                    & (fold_metrics["method"] == method), metric].mean()
                for p in PROTOCOLS]
            ax.bar(x + (i - 0.5) * width, values, width=width,
                   label=METHOD_LABELS[method], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOLS],
                           rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, output)


def make_conformal_figures(
    result_dir: Path = RESULTS_DIR / "conformal_full",
    shift_path: Path = RESULTS_DIR / "conformal" / "shift_metrics.csv",
) -> list[Path]:
    _style()
    fold_metrics = pd.read_csv(result_dir / "fold_metrics.csv")
    shift = pd.read_csv(shift_path)
    return [
        plot_coverage_degradation(fold_metrics),
        plot_coverage_vs_shift(fold_metrics, shift),
        plot_class_conditional_coverage(fold_metrics),
        plot_set_size_and_calibration(fold_metrics),
    ]

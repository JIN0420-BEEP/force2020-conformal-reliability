from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force2020_reliability.paths import FIGURES_DIR, RESULTS_DIR

CONDITIONS = [
    ("random_point", "blocks5_buffer20", "Random point"),
    ("well_grouped", "blocks5_buffer20", "Grouped by well"),
    ("spatial_block", "blocks5_buffer20", "Spatial 5 blocks / 20 km"),
    ("spatial_block", "blocks6_buffer30", "Spatial 6 blocks / 30 km"),
]
COLORS = ["#D55E00", "#0072B2", "#009E73", "#7F3C8D"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def _condition_values(
    frame: pd.DataFrame,
    metric: str,
) -> list[np.ndarray]:
    return [
        frame.loc[
            (frame["protocol"] == protocol)
            & (frame["spatial_scheme"] == scheme),
            metric,
        ].to_numpy()
        for protocol, scheme, _ in CONDITIONS
    ]


def plot_sensitivity(
    results_root: Path = RESULTS_DIR,
    output_path: Path = FIGURES_DIR / "07_split_sensitivity.png",
) -> Path:
    frame = pd.read_csv(results_root / "sensitivity" / "fold_metrics.csv")
    frame = frame.loc[frame["fold"] >= 0]
    rng = np.random.default_rng(20261021)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    values = _condition_values(frame, "macro_f1")
    boxes = ax.boxplot(
        values,
        positions=np.arange(len(values)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for index, (patch, color, data) in enumerate(zip(boxes["boxes"], COLORS, values, strict=True)):
        patch.set_facecolor(color)
        patch.set_alpha(0.32)
        jitter = rng.uniform(-0.08, 0.08, len(data))
        ax.scatter(index + jitter, data, s=18, color=color, edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([label for _, _, label in CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("Macro-F1 across 5 folds × 3 seeds")
    ax.set_ylim(0.20, 0.85)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_nested_selective(
    results_root: Path = RESULTS_DIR,
    output_path: Path = FIGURES_DIR / "08_nested_selective.png",
) -> Path:
    frame = pd.read_csv(results_root / "nested" / "all_metrics.csv")
    frame = frame.loc[frame["fold"] >= 0]
    rng = np.random.default_rng(20261026)
    metrics = [
        ("calibrated_coverage", "Coverage", (0.0, 0.80), None),
        (
            "calibrated_selective_accuracy",
            "Selective accuracy",
            (0.65, 1.01),
            0.90,
        ),
        ("calibrated_threshold", "Calibrated threshold", (0.70, 1.00), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    for ax, (metric, label, ylim, target_line) in zip(axes, metrics, strict=True):
        values = _condition_values(frame, metric)
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(values)),
            widths=0.50,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for index, (patch, color, data) in enumerate(zip(boxes["boxes"], COLORS, values, strict=True)):
            patch.set_facecolor(color)
            patch.set_alpha(0.30)
            jitter = rng.uniform(-0.08, 0.08, len(data))
            ax.scatter(index + jitter, data, s=18, color=color, edgecolor="white", linewidth=0.4)
        if target_line is not None:
            ax.axhline(target_line, color="0.30", linestyle="--", linewidth=0.9)
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels([label for _, _, label in CONDITIONS], rotation=32, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_ablation(
    results_root: Path = RESULTS_DIR,
    output_path: Path = FIGURES_DIR / "09_feature_ablation.png",
) -> Path:
    frame = pd.read_csv(results_root / "feature_ablation" / "summary.csv")
    protocols = ["random_point", "well_grouped", "spatial_block"]
    protocol_labels = ["Random point", "Grouped by well", "Spatial block"]
    feature_sets = ["full", "logs_location", "logs_depth", "logs_only"]
    feature_labels = ["Full", "Logs + location", "Logs + depth", "Logs only"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    positions = np.arange(len(protocols))
    for index, (feature_set, feature_label, color) in enumerate(
        zip(feature_sets, feature_labels, colors, strict=True)
    ):
        subset = (
            frame.loc[frame["feature_set"] == feature_set]
            .set_index("protocol")
            .reindex(protocols)
        )
        offsets = positions + (index - 1.5) * width
        ax.bar(
            offsets,
            subset["macro_f1_mean"],
            width=width,
            color=color,
            alpha=0.82,
            label=feature_label,
            yerr=subset["macro_f1_std"],
            capsize=2,
            error_kw={"linewidth": 0.7},
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(protocol_labels, rotation=15, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 0.88)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

def make_stage2_figures(results_root: Path = RESULTS_DIR) -> list[Path]:
    _style()
    return [
        plot_sensitivity(results_root),
        plot_nested_selective(results_root),
        plot_feature_ablation(results_root),
    ]
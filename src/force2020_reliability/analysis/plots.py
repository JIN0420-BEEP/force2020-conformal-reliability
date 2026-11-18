from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from force2020_reliability.paths import FIGURES_DIR, INTERIM_DIR, RESULTS_DIR

PROTOCOL_LABELS = {
    "random_point": "Random point",
    "well_grouped": "Grouped by well",
    "spatial_block": "Spatial block",
}
PROTOCOL_COLORS = {
    "random_point": "#D55E00",
    "well_grouped": "#0072B2",
    "spatial_block": "#009E73",
}
MODEL_LABELS = {
    "hist_gradient_boosting": "Histogram gradient boosting",
    "random_forest": "Random forest",
}


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


def plot_spatial_blocks(
    output_path: Path = FIGURES_DIR / "01_spatial_blocks.png",
) -> Path:
    metadata = pd.read_csv(INTERIM_DIR / "well_metadata.csv")
    folds = pd.read_csv(INTERIM_DIR / "splits" / "spatial_well_folds.csv")
    frame = metadata.merge(folds[["well_id", "fold"]], on="well_id", how="inner")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    colors = plt.get_cmap("tab10")
    for fold, group in frame.groupby("fold", observed=True):
        ax.scatter(
            group["x"] / 1000.0,
            group["y"] / 1000.0,
            s=np.sqrt(group["rows_labeled"]) * 1.5,
            color=colors(int(fold) % 10),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label=f"Block {int(fold) + 1}",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("Spatial blocks defined from well locations")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_protocol_performance(
    analysis_dir: Path = RESULTS_DIR / "pilot" / "analysis",
    output_path: Path = FIGURES_DIR / "02_protocol_macro_f1.png",
) -> Path:
    folds = pd.read_csv(analysis_dir / "fold_metrics.csv")
    models = ["hist_gradient_boosting", "random_forest"]
    protocols = ["random_point", "well_grouped", "spatial_block"]
    rng = np.random.default_rng(20261021)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        data = [
            folds.loc[
                (folds["model"] == model) & (folds["protocol"] == protocol),
                "macro_f1",
            ].to_numpy()
            for protocol in protocols
        ]
        box = ax.boxplot(
            data,
            positions=np.arange(len(protocols)),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, protocol in zip(box["boxes"], protocols, strict=True):
            patch.set_facecolor(PROTOCOL_COLORS[protocol])
            patch.set_alpha(0.35)
        for index, values in enumerate(data):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                index + jitter,
                values,
                s=18,
                color=PROTOCOL_COLORS[protocols[index]],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_xticks(np.arange(len(protocols)))
        ax.set_xticklabels([PROTOCOL_LABELS[item] for item in protocols], rotation=18, ha="right")
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylim(0.15, 0.80)
    axes[0].set_ylabel("Macro-F1 across five folds")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_risk_coverage(
    result_dir: Path = RESULTS_DIR / "pilot",
    output_path: Path = FIGURES_DIR / "03_risk_coverage.png",
) -> Path:
    models = ["hist_gradient_boosting", "random_forest"]
    protocols = ["random_point", "well_grouped", "spatial_block"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        for protocol in protocols:
            path = result_dir / protocol / model / "risk_coverage_pooled.csv"
            curve = pd.read_csv(path)
            ax.plot(
                curve["coverage"],
                curve["selective_accuracy"],
                color=PROTOCOL_COLORS[protocol],
                linewidth=1.6,
                label=PROTOCOL_LABELS[protocol],
            )
        ax.axhline(0.90, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.25, 1.01)
        ax.set_xlabel("Coverage of retained depth points")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Selective accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_recall(
    analysis_dir: Path = RESULTS_DIR / "pilot" / "analysis",
    output_path: Path = FIGURES_DIR / "04_class_recall.png",
) -> Path:
    frame = pd.read_csv(analysis_dir / "per_class_recall.csv")
    protocols = ["random_point", "well_grouped", "spatial_block"]
    models = ["hist_gradient_boosting", "random_forest"]
    support_order = (
        frame.loc[frame["protocol"] == "random_point"]
        .groupby("lithology_name", observed=True)["support"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), sharey=True)
    image = None
    for ax, model in zip(axes, models, strict=True):
        subset = frame.loc[frame["model"] == model]
        pivot = subset.pivot(
            index="lithology_name", columns="protocol", values="recall"
        ).reindex(support_order)
        pivot = pivot.loc[:, protocols]
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(protocols)))
        ax.set_xticklabels([PROTOCOL_LABELS[item] for item in protocols], rotation=22, ha="right")
        ax.set_yticks(np.arange(len(support_order)))
        ax.set_title(MODEL_LABELS[model])
        ax.grid(False)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.iloc[row, column]
                if pd.notna(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.55 else "black",
                        fontsize=7,
                    )
    axes[0].set_yticklabels(support_order)
    if image is not None:
        fig.subplots_adjust(left=0.24, right=0.88, bottom=0.20, top=0.90, wspace=0.10)
        colorbar_axis = fig.add_axes([0.90, 0.20, 0.022, 0.70])
        fig.colorbar(image, cax=colorbar_axis, label="Recall")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_strat_group_accuracy(
    analysis_dir: Path = RESULTS_DIR / "pilot" / "analysis",
    output_path: Path = FIGURES_DIR / "05_strat_group_accuracy.png",
) -> Path:
    frame = pd.read_csv(analysis_dir / "strat_group_metrics.csv")
    protocols = ["random_point", "well_grouped", "spatial_block"]
    models = ["hist_gradient_boosting", "random_forest"]
    order = (
        frame.groupby("strat_group", observed=True)["n_rows"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.4), sharey=True)
    image = None
    for ax, model in zip(axes, models, strict=True):
        subset = frame.loc[frame["model"] == model]
        pivot = (
            subset.pivot(index="strat_group", columns="protocol", values="accuracy")
            .reindex(order)
            .loc[:, protocols]
        )
        image = ax.imshow(pivot.to_numpy(), vmin=0.2, vmax=1.0, cmap="magma", aspect="auto")
        ax.set_xticks(np.arange(len(protocols)))
        ax.set_xticklabels([PROTOCOL_LABELS[item] for item in protocols], rotation=22, ha="right")
        ax.set_yticks(np.arange(len(order)))
        ax.set_title(MODEL_LABELS[model])
        ax.grid(False)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.iloc[row, column]
                if pd.notna(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.7 else "black",
                        fontsize=6.5,
                    )
    axes[0].set_yticklabels(order, fontsize=7)
    if image is not None:
        fig.subplots_adjust(left=0.28, right=0.88, bottom=0.20, top=0.90, wspace=0.10)
        colorbar_axis = fig.add_axes([0.90, 0.20, 0.022, 0.70])
        fig.colorbar(image, cax=colorbar_axis, label="Accuracy")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_distance_accuracy(
    analysis_dir: Path = RESULTS_DIR / "pilot" / "analysis",
    output_path: Path = FIGURES_DIR / "06_distance_accuracy.png",
) -> Path:
    frame = pd.read_csv(analysis_dir / "well_metrics.csv")
    frame = frame.loc[frame["protocol"] != "random_point"].copy()
    frame["distance_km"] = frame["distance_to_nearest_train_m"] / 1000.0
    models = ["hist_gradient_boosting", "random_forest"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        subset = frame.loc[frame["model"] == model]
        for protocol in ["well_grouped", "spatial_block"]:
            protocol_data = subset.loc[subset["protocol"] == protocol]
            ax.scatter(
                protocol_data["distance_km"],
                protocol_data["accuracy"],
                s=22,
                alpha=0.68,
                color=PROTOCOL_COLORS[protocol],
                edgecolor="white",
                linewidth=0.3,
                label=PROTOCOL_LABELS[protocol],
            )
        ax.set_xlabel("Distance to nearest training well (km)")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Well-level accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_all_figures(
    result_dir: Path = RESULTS_DIR / "pilot",
    figure_dir: Path = FIGURES_DIR,
) -> list[Path]:
    _style()
    analysis_dir = result_dir / "analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_spatial_blocks(figure_dir / "01_spatial_blocks.png"),
        plot_protocol_performance(analysis_dir, figure_dir / "02_protocol_macro_f1.png"),
        plot_risk_coverage(result_dir, figure_dir / "03_risk_coverage.png"),
        plot_class_recall(analysis_dir, figure_dir / "04_class_recall.png"),
        plot_strat_group_accuracy(analysis_dir, figure_dir / "05_strat_group_accuracy.png"),
        plot_distance_accuracy(analysis_dir, figure_dir / "06_distance_accuracy.png"),
    ]
    return outputs
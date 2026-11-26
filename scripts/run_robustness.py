"""Spatial-partition robustness: blocks5/20 km vs blocks6/30 km.

Compares spatial-block coverage, accuracy, macro-F1 and set size across the
two block-count/buffer configurations over three seeds. Only the spatial
protocol depends on the blocking scheme; random-point and grouped-well folds
are identical across schemes and are therefore not rerun for blocks6.
"""

import argparse

import pandas as pd

from force2020_reliability.paths import RESULTS_DIR

SEED_IDS = [20261021, 20261022, 20261023]


def _load(directories, scheme):
    frames = []
    for directory in directories:
        table = pd.read_csv(RESULTS_DIR / directory / "fold_metrics.csv")
        table["scheme_label"] = scheme
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b5-dirs", nargs="+",
                        default=["well_level_seed1", "well_level_seed2",
                                 "well_level_seed3"])
    parser.add_argument("--b6-dirs", nargs="+",
                        default=["well_level_blocks6_seed1",
                                 "well_level_blocks6_seed2",
                                 "well_level_blocks6_seed3"])
    parser.add_argument("--output", default="well_level")
    args = parser.parse_args()

    b5 = _load(args.b5_dirs, "blocks5_buffer20")
    b6 = _load(args.b6_dirs, "blocks6_buffer30")
    both = pd.concat([b5, b6], ignore_index=True)
    spatial = both[both["protocol"] == "spatial_block"].copy()

    agg_dict = {
        "coverage": ["mean", "std"],
        "point_accuracy": ["mean", "std"],
        "mean_set_size": ["mean"],
    }
    if "macro_f1" in spatial.columns:
        agg_dict["macro_f1"] = ["mean", "std"]

    summary = spatial.groupby(["scheme_label", "model"]).agg(agg_dict)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().round(4)

    # Paired HGB per-seed comparison (same seed, spatial folds only).
    hgb = spatial[spatial["model"] == "hist_gradient_boosting"]
    per_seed = hgb.groupby(["scheme_label", "seed"])["coverage"].mean().unstack(0)
    per_seed["delta_b6_minus_b5"] = (
        per_seed.get("blocks6_buffer30") - per_seed.get("blocks5_buffer20"))
    per_seed = per_seed.round(4).reset_index()

    out = RESULTS_DIR / args.output
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "spatial_scheme_robustness.csv", index=False,
                   encoding="utf-8-sig")
    per_seed.to_csv(out / "spatial_scheme_hgb_by_seed.csv", index=False,
                    encoding="utf-8-sig")

    print("=== spatial robustness by scheme/model ===")
    print(summary.to_string(index=False))
    print("\n=== HGB spatial coverage by seed ===")
    print(per_seed.to_string(index=False))


if __name__ == "__main__":
    main()

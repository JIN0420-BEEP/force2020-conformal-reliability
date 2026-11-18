"""Entry point: run conformal experiments (default 1000 rows/well)."""

import argparse
from dataclasses import replace

import pandas as pd

from force2020_reliability.config import load_config
from force2020_reliability.models.conformal_experiment import (
    run_conformal_experiment,
)
from force2020_reliability.paths import INTERIM_DIR

DATA = INTERIM_DIR / "force2020_labeled_strat.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-well", type=int, default=1000)
    parser.add_argument("--output", default="conformal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--schemes", nargs="+", default=["blocks5_buffer20"])
    args = parser.parse_args()

    frame = pd.read_parquet(DATA)
    config = load_config()
    if args.seed is not None:
        config = replace(config, raw={**config.raw, "seed": args.seed})
    summary = run_conformal_experiment(
        frame, config,
        rows_per_well=args.rows_per_well,
        output_name=args.output,
        splits_subdir=tuple(args.schemes),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

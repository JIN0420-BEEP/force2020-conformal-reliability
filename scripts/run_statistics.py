"""Cross-seed statistical inference: confidence intervals, tests, effect sizes."""

import pandas as pd
from scipy import stats

from force2020_reliability.analysis.conformal_plots import METHODS, PROTOCOLS
from force2020_reliability.paths import RESULTS_DIR

SEED_DIRS = {
    20261021: "conformal_full",
    20261022: "conformal_seed2",
    20261023: "conformal_seed3",
}
NOMINAL = 0.90


def load_pooled() -> pd.DataFrame:
    frames = []
    for seed, directory in SEED_DIRS.items():
        path = RESULTS_DIR / directory / "fold_metrics.csv"
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    pooled = load_pooled()
    rows = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            values = pooled.loc[
                (pooled["protocol"] == protocol)
                & (pooled["method"] == method), "coverage"].to_numpy()
            mean, sd = values.mean(), values.std(ddof=1)
            t, p_value = stats.ttest_1samp(values, NOMINAL)
            ci_low, ci_high = stats.t.interval(
                0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
            rows.append({
                "protocol": protocol,
                "method": method,
                "coverage_mean": mean,
                "coverage_sd": sd,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "gap_from_nominal": mean - NOMINAL,
                "t_vs_nominal": t,
                "p_value": p_value,
            })
    table = pd.DataFrame(rows)
    output = RESULTS_DIR / "conformal" / "coverage_statistics.csv"
    table.to_csv(output, index=False, float_format="%.5f", encoding="utf-8-sig")

    # Spearman correlation between coverage and covariate shift.
    shift = pd.read_csv(RESULTS_DIR / "conformal" / "shift_metrics.csv")
    aps = pooled.loc[
        (pooled["method"] == "aps") & (pooled["seed"] == 20261021)].merge(
        shift, on=["protocol", "fold"])
    rho, p_rho = stats.spearmanr(aps["mmd"], aps["coverage"])

    print(table.to_string(index=False))
    print(f"\nSpearman rho(coverage, MMD) = {rho:.3f}, p = {p_rho:.4f}")
    print("saved", output)


if __name__ == "__main__":
    main()

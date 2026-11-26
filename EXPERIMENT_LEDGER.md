# Experiment ledger — data scale, seeds, schemes, and provenance

Every numerical claim in the manuscript is traceable to one row below. The
independent sampling unit for inference is the **well** (and, for spatial
extrapolation, the **spatial block**); depth points and folds are not treated
as independent because the same well recurs across folds and seeds.

| # | Experiment / artifact | Data scale | Model(s) | Protocols × methods | Seeds | Spatial scheme | Result location |
|---|---|---|---|---|---|---|---|
| 1 | Full-scale primary result | Full 1,431,383 points, 118 wells | HGB | 3 protocols × 4 conformal methods | 20261021 | blocks5_buffer20 | `results/conformal_full` |
| 2 | Multi-seed conformal (legacy statistics) | ≤1000 points/well = 118,000 points | HGB | 3 protocols × 4 methods | 3 seeds | blocks5_buffer20 | `results/conformal`, `conformal_seed2`, `conformal_seed3` |
| 3 | **Well-level multi-model (new)** | ≤1000 points/well = 118,000 points | 6 families | 3 protocols, marginal APS | 3 seeds | blocks5_buffer20 | `results/well_level_seed1/2/3` |
| 4 | **Spatial-scheme robustness (new)** | ≤1000 points/well = 118,000 points | 6 families | spatial_block only, marginal APS | 3 seeds | blocks6_buffer30 | `results/well_level_blocks6_seed1/2/3` |
| 5 | **Distribution shift (new)** | ≤1000 points/well | — | 3 protocols | 3 seeds | blocks5_buffer20 | `results/well_level/shift_long.csv` |
| 5b | **Cluster-robust inference (new)** | ≤1000 points/well | 6 families | well-unit bootstrap, fold performance (HGB macro-F1), block summary, within-protocol Spearman (n=15) | 3 seeds | blocks5_buffer20 | `results/well_level/well_unit_inference.csv`, `fold_level_performance.csv`, `block_level_summary.csv`, `within_protocol_correlation.csv` |
| 5c | **Block-design robustness (new)** | ≤1000 points/well | 6 families | spatial only: coverage/accuracy/macro-F1/set size, 5-block vs 6-block | 3 seeds | blocks5_buffer20 + blocks6_buffer30 | `results/well_level/spatial_scheme_robustness.csv`, `spatial_scheme_hgb_by_seed.csv` (`scripts/run_robustness.py`) |
| 6 | Boundary uncertainty | ≤1000 points/well | HGB | set size vs. boundary distance | 20261021 | blocks5_buffer20 | `results/conformal/boundary_uncertainty.csv` |
| 7 | Split sensitivity (point CV; no calibration-well holdout) | 118,000 points | HGB | accuracy / balanced accuracy / macro-F1, 15 runs | 3 seeds × 5 folds | blocks5 + blocks6 | `results/sensitivity/summary.csv` |

> `results/sensitivity/summary.csv` has a `coverage90_*` column from a different
> pointwise interval diagnostic; it is **not** split-conformal set coverage and
> is not cited in the manuscript.

## Model families (experiment 3 and 4)

1. `hist_gradient_boosting` — boosting, native NaN handling (HGB)
2. `random_forest` — bagged trees, median imputation
3. `lightgbm` — leaf-wise boosting, balanced weights
4. `extra_trees` — random-split bagged trees (different split rule)
5. `logistic` — multinomial linear model, standardised features
6. `knn` — instance-based, k = 25, standardised features

## Reconciling the spatial-coverage numbers (do not mix scales/estimands)

- **0.860** — single full-scale run (1,431,383 points; experiment 1; seed
  20261021), point-weighted fold mean. Method-variant/class-conditional
  analyses only.
- **0.8647** — HGB point-weighted fold mean over 3 seeds × 5 spatial folds on
  the 118,000-point sample (`fold_level_performance.csv`).
- **0.8662, 95% well-bootstrap CI 0.8457–0.8840, p = 0.00075** — equal-well
  estimate (each well averaged over folds/seeds; n = 118) from
  `well_unit_inference.csv`. Primary equal-well descriptive interval: it
  quantifies across-well sampling variability of **empirical** coverage, not a
  conformal prediction interval or a cluster-level finite-sample guarantee.
- **0.86389, legacy CI 0.84480–0.88297, p = 0.00117** — **retired**. It treated
  15 seed×fold values as independent even though the same wells recur across
  seeds; do not cite.

Model heterogeneity (equal-well spatial coverage): HGB 0.866 [0.846,0.884] and
LightGBM 0.869 [0.848,0.887] are below 0.90; random forest 0.888, extra trees
0.888, and logistic 0.890 have CIs including 0.90; k-NN 0.921 [0.910,0.933] is
above target via ~6-class sets. Do not state that all models under-cover.

**Guarantee boundary.** All grouped-well and spatial numbers are empirical
coverage. The conformal quantile is calibrated over depth points; grouping
validation folds and well-bootstrapping do not make it cluster conformal
prediction, and the bootstrap intervals are not prediction intervals. A
well/block-level calibration (cluster score/quantile or weighted conformal)
would be required for a cluster-level guarantee and was not run.

## Correlation correction (second pass: no independent-fold significance)

The earlier pooled Spearman rho = -0.807 (p = 0.0003) merged the three protocols
on the MMD axis and repeated each fold MMD across six models; it is **retired**.

The replacement within-protocol seed×fold table (`within_protocol_correlation.csv`)
has 15 rows (3 seeds × 5 folds), but the seeds reuse the same 118 wells and the
same five geographic blocks. These are **not 15 independent geographic samples**,
so its `p_value_nominal` column is descriptive only and must not be cited as
significance evidence (this specifically removes the earlier "HGB spatial
p = 0.045" and "LightGBM p = 0.001" significance wording). Effect sizes:
six-model mean well rho = -0.450, spatial -0.404; HGB well -0.425, spatial
-0.525; per-model spatial values vary (LightGBM -0.761; k-NN +0.197;
logistic +0.046).

The independent geographic unit is the held-out block: only five blocks per
seed (blocks5/20 km; six for blocks6/30 km), reused across seeds. Per-seed
block rank correlations are in `block_level_correlation.csv`: HGB
rho = -0.50 / -0.70 / -1.00 and six-model mean 0.00 / -0.50 / -0.60 for seeds
20261021/22/23. Treat these as low-powered descriptive summaries, not pooled
inference. Overall wording: moderate, model-dependent, observational
associations—not a universal shift–coverage law and not causal.

## Class-balanced numbers

- Nested conformal folds, HGB macro-F1 (`fold_level_performance.csv`):
  random 0.712 (SD 0.053), well 0.401 (0.029), spatial 0.265 (0.036).
- Point-CV sensitivity without calibration holdout (`sensitivity/summary.csv`,
  15 runs): accuracy 0.891/0.739/0.688; macro-F1 0.788/0.424/0.301; balanced
  accuracy 0.747/0.407/0.303 (random/well/spatial). blocks6 spatial macro-F1
  0.292. The two macro-F1 sources differ by design and are labelled separately.
- blocks6/30 km spatial coverage (`spatial_scheme_robustness.csv`) is close to
  blocks5/20 km (HGB 0.874 vs 0.865); six-model blocks6 macro-F1 0.186–0.287.

## Model configuration note

HGB hyperparameters are learning rate 0.08, max_iter 180, 31 leaves, L2 1.0.
The early-stopping rule is parameterised via
`make_model(..., hgb_early_stopping=...)`:

- **Full-data conformal variants (1), point-CV sensitivity (7), boundary /
  nested analyses**: `hgb_early_stopping=True` (10% validation fraction,
  patience 15), matching the archived `results/conformal_full` (2026-11-18);
  a copy is kept in `results/_archive_earlystop/conformal_full`.
- **Repeated well-level multi-model and block-robustness experiments (3, 4,
  5b, 5c)**: `hgb_early_stopping=False` (fixed 180 iterations), set explicitly
  in `well_level_experiment.py`, because the smaller grouped/spatial fitting
  subsets can contain a rare class only once and the stratified internal
  validation split then fails; the fixed budget is also more reproducible.

The other five learners are identical across scales. The two HGB rules differ
only in the effective number of trees actually used (early stopping is capped
at 180); the scales already differ in point count and calibration design and
are never pooled into one estimate.

## Reporting discipline

- Point accuracy and macro-F1 are always reported together; accuracy alone
  overstates class-balanced performance under spatial shift.
- All imputation, feature scaling, class weighting, calibration quantiles, and
  decision thresholds are estimated strictly inside the training fold.
- The random-point protocol is an optimistic upper bound only, never evidence
  of spatial generalisation.
- Split-conformal coverage below the nominal level under spatial
  extrapolation is reported as an empirical observation; the method is not
  claimed to detect or quantify distribution shift, nor to act as an
  out-of-distribution detector.

## Third revision: feature-source ablation and continuous sections

- **Feature ablation (experiment 8a)**: HGB only, same 118,000-point sample,
  three seeds (`20261021/22/23`), blocks5/20 km, three protocols, five folds,
  25% calibration-well holdout, APS, `hgb_early_stopping=False` (fixed 180).
  Feature groups via `--feature-set`: `logs_only` (14 curves), `logs_depth`
  (+`DEPTH_MD`, 15), `logs_coords` (+`x_loc/y_loc/z_loc`, 17), `full` (18).
  Per-run tables: `results/ablation_<set>_seed{1,2,3}/{fold,well}_metrics.csv`
  (`feature_set`, `n_features`, `macro_f1`, empirical `coverage`,
  `mean_set_size`). Aggregation:
  `results/ablation/feature_ablation_summary.csv` (fold means/SD and
  between-seed SD), `..._seed_summary.csv`, `..._equal_well.csv`; figure
  `figures/17_feature_ablation.{png,pdf}` via
  `scripts/run_ablation_summary.py`. Coverage is point-level empirical; the
  feature change does not alter the calibration unit.
- **Continuous sections (experiment 8b)**: HGB, same sample/seeds/splits and
  fixed HGB config; every depth point of each held-out test well is predicted.
  Outputs `results/segments_seed{1,2,3}/`; combined
  `results/segments_multiseed_summary.csv` and
  `segments_multiseed_seed_summary.csv` via
  `scripts/run_segment_summary.py`. Metrics: boundary P/R/F1 and matched
  depth MAE (0.5 m run gap and matching tolerance), thickness-weighted error,
  segment-count absolute error, adjacent-depth agreement/flip rate; equal-well
  and point/pair-weighted pools. The archived single-seed
  `results/segments/` used early stopping and the fixed sensitivity sample and
  is not pooled with the three-seed run.
- **Cluster-calibration positioning**: two assurance targets are distinguished
  in the discussion (coverage of a random depth point in a new well vs the
  probability that a whole new well reaches a target coverage fraction).
  Hierarchical/group/equal-cluster calibration is cited as future work
  (Dunn et al., 2023; Bhattacharyya and Barber, 2026; Allohibi, 2026), not run.


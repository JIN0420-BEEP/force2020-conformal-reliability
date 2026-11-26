# Reproducibility manifest

This file is the command-level entry point. `EXPERIMENT_LEDGER.md` maps every
reported number to its data scale, seed, split, and result file. The two
documents use the same conventions; if they ever disagree, the CSV files in
`results/` and `EXPERIMENT_LEDGER.md` are authoritative.

## Environment

- Python 3.13 virtual environment (`.venv`), for example managed with `uv`.
- Key packages: scikit-learn, pandas, numpy, scipy, lightgbm, pyarrow,
  matplotlib.
- Set `PYTHONPATH` to the project `src` directory for every experiment run.
- Building the `.docx` uses the **system** Python with `python-docx`; the PDF
  is exported through Microsoft Word (COM) or an equivalent renderer.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$py = ".\.venv\Scripts\python.exe"
```

## Data and splits

- Source: FORCE 2020, 118 wells, 1,431,383 labelled depth points
  (https://doi.org/10.5281/zenodo.4351156).
- Local labelled table: `data/interim/force2020_labeled_strat.parquet`.
- Ten lithologies are retained; Halite (4 wells) and Basement (1 well) are
  excluded and logged.
- Precomputed splits for seeds `20261021`, `20261022`, `20261023` live under
  `data/interim/splits/sensitivity/`:
  - `blocks5_buffer20/` for `random_point`, `well_grouped`, `spatial_block`;
  - `blocks6_buffer30/` for the spatial-only robustness run.

## Two experimental scales (do not pool)

1. **Full-data single seed** (`1,431,383` points, seed 20261021): conformal
   variants, class-conditional coverage, calibration, boundary analysis.
   HGB uses internal early stopping here.
2. **Capped multi-seed experiment** (`118,000` points = at most 1,000 per well,
   all 118 wells; three seeds; six models): equal-well bootstrap, fold
   macro-F1, block robustness, shift and correlations. In this entry HGB uses
   a fixed 180 iterations with early stopping disabled, because small
   grouped/spatial fitting subsets can contain a rare class only once and the
   internal validation split then fails. The configuration difference is
   recorded in `EXPERIMENT_LEDGER.md`.

## Command sequence (run from project root)

```powershell
# 1. Full-data conformal benchmark (1.43M points; APS/Mondrian/RAPS/temperature)
& $py .\scripts\run_conformal.py --rows-per-well 0 --output conformal_full

# 2. Secondary single-seed checks (neural baseline, boundary uncertainty)
& $py .\scripts\run_mlp_conformal.py
& $py .\scripts\run_boundary_uncertainty.py

# 3. Per-fold MMD / Jensen-Shannon shift for three seeds (blocks5, all protocols)
& $py .\scripts\run_shift_multi.py

# 4. Six-model well-level experiment, one output directory per seed.
#    Repeat for seeds 20261022/20261023 with matching output names.
& $py .\scripts\run_well_level.py --seed 20261021 --output well_level_seed1 `
    --schemes blocks5_buffer20
& $py .\scripts\run_well_level.py --seed 20261022 --output well_level_seed2 `
    --schemes blocks5_buffer20
& $py .\scripts\run_well_level.py --seed 20261023 --output well_level_seed3 `
    --schemes blocks5_buffer20

# 5. Spatial robustness: six blocks / 30 km (spatial protocol only).
#    Repeat per seed with the matching output name.
& $py .\scripts\run_well_level.py --seed 20261021 `
    --schemes blocks6_buffer30 --protocols spatial_block `
    --output well_level_blocks6_seed1
& $py .\scripts\run_well_level.py --seed 20261022 `
    --schemes blocks6_buffer30 --protocols spatial_block `
    --output well_level_blocks6_seed2
& $py .\scripts\run_well_level.py --seed 20261023 `
    --schemes blocks6_buffer30 --protocols spatial_block `
    --output well_level_blocks6_seed3

# 5b. Feature-source ablation (HGB only): four input groups x three seeds.
#     Feature sets: logs_only (14 curves), logs_depth (+DEPTH_MD),
#     logs_coords (+x/y/z), full (18). Repeat each for seeds 20261021/22/23
#     with matching seed index 1/2/3, then aggregate and draw Figure 9.
foreach ($set in @("logs_only","logs_depth","logs_coords","full")) {
  & $py .\scripts\run_well_level.py --seed 20261021 `
      --output "ablation_${set}_seed1" --schemes blocks5_buffer20 `
      --models hist_gradient_boosting --feature-set $set
  & $py .\scripts\run_well_level.py --seed 20261022 `
      --output "ablation_${set}_seed2" --schemes blocks5_buffer20 `
      --models hist_gradient_boosting --feature-set $set
  & $py .\scripts\run_well_level.py --seed 20261023 `
      --output "ablation_${set}_seed3" --schemes blocks5_buffer20 `
      --models hist_gradient_boosting --feature-set $set
}
& $py .\scripts\run_ablation_summary.py

# 6. Well bootstrap, fold performance, block summary, correlations.
#    Reads results/well_level_seed1-3 and results/well_level/shift_long.csv.
& $py .\scripts\run_well_statistics.py

# 7. blocks5/20 km vs blocks6/30 km robustness table.
& $py .\scripts\run_robustness.py

# 8. Revision figures 11, 15, 16 (PNG + PDF) into figures/.
& $py .\scripts\make_revision_figures.py

# 8b. Continuous-section metrics (boundary/thickness/stability), three seeds.
#     HGB fixed 180 iterations, early stopping disabled, matching the main
#     well-level scale; every depth point of held-out wells is predicted.
& $py -m force2020_reliability.models.segment_eval `
    --seed 20261021 --output segments_seed1 --models hist_gradient_boosting
& $py -m force2020_reliability.models.segment_eval `
    --seed 20261022 --output segments_seed2 --models hist_gradient_boosting
& $py -m force2020_reliability.models.segment_eval `
    --seed 20261023 --output segments_seed3 --models hist_gradient_boosting
& $py .\scripts\run_segment_summary.py

# 9. Unit tests.
& $py -m pytest .\tests -q

# 10. Manuscript and submission materials with system Python (python-docx).
python .\manuscript\build_english_manuscript.py
python .\manuscript\build_submission_materials.py
```

The separate sensitivity cross-validation (same 118,000-point sample and
seeds, but **no calibration-well holdout**) produces
`results/sensitivity/summary.csv`; its accuracy/macro-F1 values are not mixed
with the nested conformal folds. Its generation details are logged in
`EXPERIMENT_LEDGER.md`.

## Key outputs

- `results/conformal_full/`: full-data fold/summary metrics, per-class
  coverage JSON, ECE, set size.
- `results/well_level_seed1..3/`: per-fold and per-well metrics for six
  models (HGB rows include `macro_f1`).
- `results/well_level_blocks6_seed1..3/`: six-block spatial runs.
- `results/well_level/`:
  - `well_unit_inference.csv` — equal-well means and 2,999-replicate
    bootstrap intervals (n = 118 wells);
  - `fold_level_performance.csv` — point-weighted seed-fold means/SD,
    including HGB macro-F1;
  - `block_level_summary.csv`, `block_level_observations.csv`;
  - `shift_long.csv` — per-fold MMD/energy/JS;
  - `within_protocol_correlation.csv` — descriptive seed-fold rank
    correlations (n = 15 rows; wells/blocks reused across seeds);
  - `block_level_correlation.csv` — per-seed block rank correlations
    (n = 5 independent blocks per seed);
  - `spatial_scheme_robustness.csv`, `spatial_scheme_hgb_by_seed.csv`.
- `results/ablation_<set>_seed1..3/`: feature-source ablation fold/well
  metrics (`feature_set`, `n_features`, macro-F1, empirical coverage, set size).
- `results/ablation/feature_ablation_summary.csv` (fold means/SD and
  between-seed SD), `..._seed_summary.csv`, `..._equal_well.csv`; Figure 9 is
  `figures/17_feature_ablation.{png,pdf}`.
- `results/segments_seed1..3/`: per-well boundary precision/recall/F1,
  boundary depth error, thickness and segment-count error, adjacent-depth
  agreement/flip rate, plus per-fold prediction parquet.
- `results/segments_multiseed_summary.csv` and
  `segments_multiseed_seed_summary.csv`: equal-well and point/pair-weighted
  section summaries across three seeds. The older single-seed
  `results/segments/` used internal HGB early stopping and the fixed
  sensitivity sample; it is archived and not pooled with the three-seed run.
- `figures/`: manuscript figures (PNG and vector PDF).
- `manuscript/`: `.docx`, exported `.pdf`, Highlights, Cover Letter, and the
  point-by-point response document.

## Reporting conventions

- **All grouped-well and spatial coverage values are empirical coverage.**
  Point-level split-conformal finite-sample coverage requires exchangeable
  calibration/test depths; within-well dependence and spatial shift weaken
  that premise, and the point-level quantile is not a well- or region-level
  guarantee. The well-bootstrap intervals describe across-well sampling
  variability, not conformal prediction intervals. A cluster-level guarantee
  would require well/block-level calibration (future work).
- The seed-fold correlation rows (n = 15) share the same wells and blocks
  across seeds; their nominal p-values are **not** independent-sample
  significance evidence. Independent geographic inference has only five to
  six blocks per seed, so block correlations are descriptive and low-powered.
- Feature-ablation coverage and all section metrics are likewise descriptive
  empirical results on depth points/wells; changing the input feature group
  or aggregating by well section does not change the point-level calibration
  unit and creates no cluster-level conformal guarantee. Section metrics are
  equal-well weighted unless a point/pair-weighted pool is explicitly labelled.

## Headline numbers (alpha = 0.10, marginal APS)

Full data, single seed, point-weighted fold means (HGB with early stopping):

| Protocol | Empirical coverage | Set size | Point acc. |
|---|---|---|---|
| Random point | 0.965 | 1.71 | 0.859 |
| Grouped well | 0.895 | 1.90 | 0.735 |
| Spatial block | 0.860 | 2.09 | 0.692 |

Three seeds, equal-well HGB empirical coverage (118 wells; bootstrap interval):

| Protocol | Coverage | 95% well interval |
|---|---|---|
| Random point | 0.963 | 0.957–0.968 |
| Grouped well | 0.906 | 0.894–0.917 |
| Spatial block | 0.866 | 0.846–0.884 |

HGB fold macro-F1 (point-weighted): 0.712 random, 0.401 grouped well,
0.265 spatial block.

**Retracted:** the earlier pooled correlation of coverage with MMD,
rho = -0.807 (p = 0.0003), merged the three protocols on the MMD axis and
treated repeated seeds, folds, and models as independent. It must not be
cited. The replacement analyses are the within-protocol descriptive
seed-fold correlations (e.g. HGB rho = -0.43 grouped, -0.53 spatial; nominal
p-values not inferential) and the per-seed five-block correlations (HGB
rho = -0.50, -0.70, -1.00), all in `results/well_level/`.

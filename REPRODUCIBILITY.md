# Reproducibility manifest

## Environment
- Python 3.13 virtual environment (`.venv`), managed with `uv`.
- Key packages: scikit-learn, pandas, numpy, scipy, lightgbm, pyarrow, pytest, matplotlib.
- Set `PYTHONPATH` to the project `src` directory for every run.
- Manuscript build uses the system Python with `python-docx`; PDF is exported via Word.

## Data
- Source: FORCE 2020, 118 wells, 1,431,383 labelled depth points.
- Local table: `data/interim/force2020_labeled_strat.parquet`.
- Ten lithologies retained; Halite (4 wells) and Basement (1 well) excluded and logged.
- Precomputed splits for three seeds under `data/interim/splits/sensitivity/`.

## Command sequence (run from project root)
```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

# 1. Core conformal experiment, 1000 points/well (fast)
python .\scripts\run_conformal.py --output conformal

# 2. Full-data conformal run (1.43M points)
python .\scripts\run_conformal.py --rows-per-well 0 --output conformal_full

# 3. Distribution-shift metrics (MMD, energy, Jensen-Shannon)
python .\scripts\run_shift.py

# 4. Cross-seed statistical inference (CI, t-tests, Spearman)
python .\scripts\run_statistics.py

# 5. Model-agnostic neural baseline
python .\scripts\run_mlp_conformal.py

# 6. Boundary-localised uncertainty (null finding)
python .\scripts\run_boundary_uncertainty.py

# 7. Unit tests (17 tests)
python -m pytest .\tests -q

# 8. Manuscript (system Python with python-docx)
python .\manuscript\build_english_manuscript.py
```

## Outputs
- Per-fold and summary metrics: `results/conformal*/`, `results/conformal_full/`.
- Shift and statistics tables: `results/conformal/shift_metrics.csv`,
  `results/conformal/coverage_statistics.csv`,
  `results/conformal/mlp_coverage.csv`,
  `results/conformal/boundary_uncertainty.csv`.
- Figures (PNG + vector PDF): `figures/10_` through `figures/14_`.
- Manuscript: `manuscript/FORCE2020_conformal_manuscript.docx` and `.pdf`.

## Headline numbers (full-data, alpha = 0.10, marginal APS)
| Protocol | Coverage | 95% CI | Set size | Point acc. |
|---|---|---|---|---|
| Random point | 0.965 | 0.963–0.968 | 1.71 | 0.859 |
| Grouped well | 0.895 | 0.895–0.917 | 1.90 | 0.735 |
| Spatial block | 0.860 | 0.845–0.883 | 2.09 | 0.692 |

Spearman correlation of coverage with covariate MMD: -0.807 (p = 0.0003).

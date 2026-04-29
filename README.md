# GMR-Covariate-Shift

This repository contains the public code for reproducing the simulation study
and empirical air-quality case study in the paper on Statistical Inference for General
Moment Restrictions under Covariate Shift Transfer Learning.

The code is organized so that reusable methodology is kept in `main_code/`,
simulation-specific code is kept in `simulation/`, and empirical-study code and
outputs are kept in `case_study/`.

## Requirements

The code was developed in Python and requires:

```bash
python3 -m pip install numpy pandas scipy matplotlib torch
```

PyTorch uses `cuda` when available, then `mps` on Apple Silicon, and otherwise
falls back to `cpu`.



## Repository Structure

### Top-level files

- `README.md`: this file.
- `.gitignore`: excludes generated caches, Python bytecode, temporary files,
  and operating-system metadata from version control.

### `main_code/`

Reusable implementation of the proposed methods.

- `main_code/__init__.py`: package marker.
- `main_code/config.py`: dataclass configurations for neural networks,
  mutual-information density fitting, and empirical likelihood.
- `main_code/preprocess.py`: standardization helpers for covariates and outcomes.
- `main_code/moments.py`: moment functions used in the simulation study.
- `main_code/nuisance_dr.py`: neural least-squares density-ratio estimation.
- `main_code/nuisance_mi.py`: neural conditional-density and conditional-moment
  nuisance estimation.
- `main_code/crossfit.py`: cross-fitting logic for source nuisance functions
  and target conditional moments.
- `main_code/empirical_likelihood.py`: empirical-likelihood objective,
  profiling, and confidence-interval routines.
- `main_code/inference.py`: normal and sandwich confidence-interval utilities.

### `simulation/`

Code and outputs for the main-text simulation figures.

- `simulation/__init__.py`: package marker.
- `simulation/design.py`: simulation settings, output paths, data-generating
  mechanisms, and oracle quantities.
- `simulation/experiment.py`: method wrappers and result-table generation for
  OM-DW, DRW, target-labelled, source-unadjusted, and oracle benchmarks.
- `simulation/plot_style.py`: shared plotting style and functions for all
  simulation panels.
- `simulation/Figure1.py`: reads `Result_S1.csv` and `Result_S2.csv`, then
  generates `s1_error_boxplot.png` and `s2_error_boxplot.png`.
- `simulation/Figure2.py`: reads `Result_S1.csv` and generates `wilks_qq.png`.
- `simulation/Figure3.py`: reads `Result_S1.csv` and generates
  `coverage_rate.png` and `ci_length.png`.
- `simulation/output/Result_S1.csv`: saved simulation results for Scenario S1.
- `simulation/output/Result_S2.csv`: saved simulation results for Scenario S2.
- `simulation/output/figures/`: generated simulation figures.
  The tracked figure files are `s1_error_boxplot.png`, `s2_error_boxplot.png`,
  `wilks_qq.png`, `coverage_rate.png`, and `ci_length.png`.

### `case_study/`

Code, data, and outputs for the empirical air-quality study.

- `case_study/__init__.py`: package marker.
- `case_study/preprocess.py`: loads raw city files, keeps complete 7:00-18:00
  observations from 2018-03-01 to 2018-05-31, applies a moving average with
  window size 20, rebuilds lagged covariates, and saves cleaned tables.
- `case_study/estimators.py`: empirical-study nuisance fitting, density-ratio
  fitting, O3 imputation, OM-DW/DRW/benchmark estimating equations, confidence
  intervals, and diagnostics.
- `case_study/Figure4.py`: produces `impute-O3.png` and the corresponding
  prediction tables.
- `case_study/Figure5.py`: produces `O3-drw.png` and the density-ratio table.
- `case_study/Figure6.py`: produces `O3-inference.png` and the inference and
  diagnostic tables.
- `case_study/data/Beijing.csv`, `case_study/data/Xian.csv`,
  `case_study/data/Jinan.csv`: source-domain city data.
- `case_study/data/Taiyuan.csv`: target-domain city data.
- `case_study/data/Tianjin.csv`: additional raw city file retained with the
  released data but not used in the main empirical analysis.
- `case_study/data.zip`: archived copy of the raw city data.
- `case_study/output/processed/`: cleaned source and target samples generated
  by `case_study/preprocess.py`.
  The files are `beijing_spring2018.csv`, `xian_spring2018.csv`,
  `jinan_spring2018.csv`, `taiyuan_spring2018.csv`, `source_sample.csv`,
  `target_sample.csv`, and `preprocessing_summary.csv`.
- `case_study/output/tables/`: CSV tables used to generate the empirical
  figures.
  `Figure4_predictions.csv` contains hourly target O3 predictions;
  `Figure4_daily_predictions.csv` contains daily averages for `impute-O3.png`;
  `Figure5_density_ratio.csv` contains source density-ratio estimates for
  `O3-drw.png`; `Figure6_estimates.csv` contains estimates and confidence
  intervals for `O3-inference.png`; `Figure6_diagnostics.csv` contains tuning
  and diagnostic summaries.
- `case_study/output/figures/`: generated empirical-study figures.
  The tracked figure files are `impute-O3.png`, `O3-drw.png`, and
  `O3-inference.png`.





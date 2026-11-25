# Reproducing the simulations and GAMA case study

This directory contains the analyses for *Semiparametric inference with parameter-indexed
conditional moments under covariate shift*. Run the commands below from `Code/`.

## Files

- In `simulation/`, `simulate.py` and `supplement.py` run the main and supplementary experiments.
  `nuisance.py` fits nuisance models, `inference.py` constructs confidence sets, and
  `drf_interface.R` fits distributional random forests for both analyses.
- In `case_study/`, `run.py` prepares the GAMA sample and runs the analyses.
  `pointwise_inference.py` estimates the two Table 4 parameters, `quantile_band.py` constructs
  the simultaneous band, and `plot.py` draws Figure 1.
- `requirements.txt` lists the Python package versions used for both analyses.

## Setup

The analyses used Python 3.9.13 and R 4.4.2 on 64-bit Linux. Install the Python
dependencies with `python -m pip install -r requirements.txt`. Put `Rscript` and GNU
`timeout` on your `PATH`. In R, install the packages used for the forest fits.

```r
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("remotes")
versions <- c(Rcpp = "1.1.2", RcppEigen = "0.3.4.0.2", Matrix = "1.6-5",
              fastDummies = "1.7.6", transport = "0.15-4", drf = "1.3.0")
for (p in names(versions)) {
  remotes::install_version(p, version = versions[[p]], upgrade = "never")
}
```

## Simulations

The six methods use the same simulated data in each replication. Tables 1–2 compare
absolute error, coverage and interval length for the median and Max10 at four sample sizes.
Table 3 reports RMSE, joint coverage and running time for seven quantiles.

```sh
python -B -m simulation.simulate run --experiment all --replications 500 --workers 24 > main-sim.jsonl
python -B -m simulation.simulate summarize main-sim.jsonl --latex

# Supplementary tables on source correction, sample size and band calibration
python -B -m simulation.supplement run --replications 500 --workers 1 > extra-sim.jsonl
python -B -m simulation.supplement tables extra-sim.jsonl
```

The main run used 24 workers on a machine with 64 virtual CPUs and 128 GB of RAM.
Use `--workers 1` if memory is limited. To check the installation against archived
results, run `python -B -m simulation.simulate check --workers 1`.

## GAMA case study

The fitting commands download and verify two public GAMA DR4 catalogues,
[TilingCatv46.fits](https://www.gama-survey.org/dr4/data/cat/EqInputCat/v46/TilingCatv46.fits) and
[ExternalSpecAllv07.fits](https://www.gama-survey.org/dr4/data/cat/ExternalSpec/v07/ExternalSpecAllv07.fits).
To use local copies, pass their directory with `--raw /path/to/catalogues`.

```sh
# Table 4 and the two supplementary case-study tables
python -B -m case_study.run pointwise > gama-pointwise.json
python -B -m case_study.run tables gama-pointwise.json

# Figure 1
python -B -m case_study.run curve > gama-curve.json

# Supplementary full-source sensitivity analysis
python -B -m case_study.run full-source > gama-full-source.json
```

Figure 1 compares OEL-DW and its simultaneous 95% band with the target and source-only
curves at 17 quantiles of normalized absolute redshift error. Draw it from the saved fit
in a separate Python environment with `matplotlib==3.9.2` and `numpy==1.26.4`.

```sh
python -B -m case_study.plot gama-curve.json gama_error_quantiles.pdf
```

The Table 4 Plug-DRF fit fixes the R seed for bandwidth selection. The Figure 1 code
fits the conditional redshift distribution in all five folds.

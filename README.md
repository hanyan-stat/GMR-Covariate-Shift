# Transfer Learning Under Covariate Shift

This repository contains the code used to reproduce the simulation study in the
paper.

## Repository layout

- [main_code](main_code)
  Core method implementation: nuisance estimation, cross-fitting, empirical
  likelihood, and inference.
- [simulation](simulation)
  Simulation settings, result-table generation, and the three figure scripts
  used in the paper.

## Install

Create a Python environment and install the required packages:

```bash
python3 -m pip install numpy pandas scipy matplotlib torch
```

The code automatically uses `cuda` when available, otherwise falls back to
`mps` or `cpu`.

## Reproduce the paper figures

Run from the repository root:

```bash
python3 simulation/experiment.py --replications 300 --force
python3 simulation/Figure1.py
python3 simulation/Figure2.py
python3 simulation/Figure3.py
```

## Output locations

- Raw simulation results: [simulation/output/Result_S1.csv](simulation/output/Result_S1.csv),
  [simulation/output/Result_S2.csv](simulation/output/Result_S2.csv)
- Generated figures: [simulation/output/figures](simulation/output/figures)

More detail is provided in [simulation/README.md](simulation/README.md).

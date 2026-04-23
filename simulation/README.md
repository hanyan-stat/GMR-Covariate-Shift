# Simulation Folder

This folder only contains the code needed for the three main-text simulation
figures in the paper.

## Files

- [design.py](design.py)
  Simulation settings, output paths, data-generating processes, and oracle
  quantities.
- [experiment.py](experiment.py)
  Method wrappers and result-table generation for the five simulation methods.
  It can also be run directly to generate the two result CSV files.
- [plot_style.py](plot_style.py)
  Plotting code for Figure 1, Figure 2, and Figure 3. The plotting functions
  read `Result_S1.csv` and `Result_S2.csv`.
- [Figure1.py](Figure1.py)
  Reads the two result tables and generates Figure 1.
- [Figure2.py](Figure2.py)
  Reads `Result_S1.csv` and generates Figure 2.
- [Figure3.py](Figure3.py)
  Reads `Result_S1.csv` and generates Figure 3.

## Output

- [output/Result_S1.csv](output/Result_S1.csv)
  Raw simulation results for Scenario S1.
- [output/Result_S2.csv](output/Result_S2.csv)
  Raw simulation results for Scenario S2.
- [output/figures](output/figures)
  PNG files for the three main-text figures.

## Recommended usage

1. Generate or refresh the two result tables:

```bash
python3 simulation/experiment.py --replications 300 --force
```

2. Draw the three figures from the saved CSV files:

```bash
python3 simulation/Figure1.py
python3 simulation/Figure2.py
python3 simulation/Figure3.py
```

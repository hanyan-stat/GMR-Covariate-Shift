"""Draw Figure 2 from Result_S1.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.plot_style import make_figure2
from simulation.design import figures_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 in the paper.")
    parser.add_argument(
        "--s1-csv",
        type=Path,
        default=ROOT / "simulation" / "output" / "Result_S1.csv",
        help="Path to the S1 result table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.s1_csv.exists():
        raise FileNotFoundError(f"Missing S1 result table: {args.s1_csv}")
    make_figure2(args.s1_csv, figures_dir(ROOT))


if __name__ == "__main__":
    main()

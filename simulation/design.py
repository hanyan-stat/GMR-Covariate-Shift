"""Simulation design, paths, data generation, and oracle quantities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, special, stats

from main_code.moments import mean_moment


@dataclass(frozen=True)
class SimulationConfig:
    replications: int
    n_p_values: Sequence[int]
    alpha_values: Sequence[str]
    scenario: str
    output_csv: Path
    n_q_ratio: float = 0.2
    n_q_fixed: Optional[int] = None
    dimension: int = 50
    k_folds: int = 4
    seed: int = 20260409
    truth_method: str = "closed_form"

    def n_q(self, n_p: int) -> int:
        if self.n_q_fixed is not None:
            return self.n_q_fixed
        return int(round(n_p * self.n_q_ratio))


def output_dir(base: Path) -> Path:
    return base / "simulation" / "output"


def results_dir(base: Path) -> Path:
    return output_dir(base)


def figures_dir(base: Path) -> Path:
    return output_dir(base) / "figures"


def default_s1_config(base: Path, replications: int = 300) -> SimulationConfig:
    return SimulationConfig(
        replications=replications,
        n_p_values=(1000, 2500, 5000, 10000),
        alpha_values=("inf",),
        scenario="s1",
        output_csv=results_dir(base) / "Result_S1.csv",
    )


def default_s2_config(base: Path, replications: int = 300) -> SimulationConfig:
    return SimulationConfig(
        replications=replications,
        n_p_values=(10000,),
        alpha_values=("4", "8", "16", "32"),
        scenario="s2",
        output_csv=results_dir(base) / "Result_S2.csv",
        n_q_fixed=2000,
    )


@dataclass
class SampleBundle:
    x_source: np.ndarray
    y_source: np.ndarray
    x_target: np.ndarray
    y_target: np.ndarray
    scenario: str
    alpha: str


def mean_function(x1: np.ndarray) -> np.ndarray:
    return np.abs(x1) - 2.0 * np.sqrt(np.abs(x1))


def scale_function(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.2 * np.abs(x[:, 1]) + 0.3 * np.abs(x[:, 3])


def sigma_from_alpha(alpha) -> float:
    if alpha in ("inf", math.inf):
        return 1.0
    alpha_val = float(alpha)
    return math.sqrt(alpha_val / (alpha_val - 1.0))


def generate_covariates_s1(n: int, d: int, rng: np.random.Generator, target: bool) -> np.ndarray:
    x = rng.uniform(0.0, 1.0, size=(n, d))
    if target:
        x[:, 0] = rng.beta(6.0 / 5.0, 6.0 / 5.0, size=n)
    return x


def generate_covariates_s2(
    n: int,
    d: int,
    alpha,
    rng: np.random.Generator,
    target: bool,
) -> np.ndarray:
    x = rng.normal(size=(n, d))
    if target:
        x[:, 0] *= sigma_from_alpha(alpha)
    return x


def generate_response(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(size=x.shape[0])
    return mean_function(x[:, 0]) + scale_function(x) * eps


def simulate_sample(
    scenario: str,
    n_p: int,
    n_q: int,
    d: int,
    alpha: str,
    seed: int,
) -> SampleBundle:
    rng = np.random.default_rng(seed)
    if scenario.lower() == "s1":
        x_source = generate_covariates_s1(n_p, d, rng, target=False)
        x_target = generate_covariates_s1(n_q, d, rng, target=True)
    elif scenario.lower() == "s2":
        x_source = generate_covariates_s2(n_p, d, alpha, rng, target=False)
        x_target = generate_covariates_s2(n_q, d, alpha, rng, target=True)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    y_source = generate_response(x_source, rng)
    y_target = generate_response(x_target, rng)
    return SampleBundle(
        x_source=x_source,
        y_source=y_source,
        x_target=x_target,
        y_target=y_target,
        scenario=scenario.lower(),
        alpha=str(alpha),
    )


def density_ratio_s1(x: np.ndarray) -> np.ndarray:
    return stats.beta.pdf(x[:, 0], a=6.0 / 5.0, b=6.0 / 5.0)


def density_ratio_s2(x: np.ndarray, alpha) -> np.ndarray:
    sigma = sigma_from_alpha(alpha)
    quad = 0.5 * (1.0 - sigma ** -2) * x[:, 0] ** 2
    return sigma ** -1 * np.exp(quad)


def density_ratio(x: np.ndarray, scenario: str, alpha: str) -> np.ndarray:
    if scenario == "s1":
        return density_ratio_s1(x)
    if scenario == "s2":
        return density_ratio_s2(x, alpha)
    raise ValueError(f"unknown scenario: {scenario}")


def oracle_m_theta(x: np.ndarray, theta: float) -> np.ndarray:
    x1 = x[:, 0]
    m = mean_function(x1)
    e = scale_function(x)
    part1 = x1 * (m - x1 * theta)
    z = (x1 * theta - m) / e
    part2 = x1 * (0.5 - stats.norm.cdf(z))
    return np.column_stack([part1, part2])


def theta_truth_closed_form(scenario: str, alpha: str) -> float:
    if scenario == "s2":
        return 0.0
    a = b = 6.0 / 5.0
    ex2 = a * (a + 1.0) / ((a + b) * (a + b + 1.0))
    ex32 = special.gamma(a + 1.5) * special.gamma(a + b)
    ex32 /= special.gamma(a) * special.gamma(a + b + 1.5)
    return (ex2 - 2.0 * ex32) / ex2


def theta_truth_monte_carlo(
    scenario: str,
    alpha: str,
    n_draws: int = 100000,
    seed: int = 20260409,
) -> float:
    rng = np.random.default_rng(seed)
    d = 50
    if scenario == "s1":
        x = generate_covariates_s1(n_draws, d, rng, target=True)
    else:
        x = generate_covariates_s2(n_draws, d, alpha, rng, target=True)
    y = generate_response(x, rng)

    def objective(theta: float) -> float:
        gbar = mean_moment(x, y, theta)
        return float(gbar @ gbar)

    result = optimize.minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(result.x)


def theta_truth(scenario: str, alpha: str, method: str = "closed_form") -> float:
    if method == "monte_carlo":
        return theta_truth_monte_carlo(scenario, alpha)
    return theta_truth_closed_form(scenario, alpha)

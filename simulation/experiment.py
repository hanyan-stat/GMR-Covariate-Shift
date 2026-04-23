"""Method wrappers and result-table generation for the simulation study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from main_code.config import ELConfig, MIFitConfig, NetworkConfig
from main_code.config import default_el_config, default_mi_config, default_network_config
from main_code.crossfit import CrossFittedNuisance, fit_cross_fitted_nuisance
from main_code.empirical_likelihood import (
    build_objective,
    estimate_theta,
    fixed_theta_stat,
    profile_ci,
    wilks_reject,
)
from main_code.inference import sandwich_ci
from main_code.moments import moment_vector

from simulation.design import (
    SampleBundle,
    SimulationConfig,
    default_s1_config,
    default_s2_config,
    density_ratio,
    oracle_m_theta,
    simulate_sample,
    theta_truth,
)


@dataclass
class MethodResult:
    method: str
    theta_hat: float
    ci_low: float
    ci_high: float
    theta_error: float
    theta_coverage: float
    theta_ci_length: float
    wilks_stat_p2: Optional[float]
    wilks_reject_95_p2: Optional[float]

    def to_row(self, replication: int, alpha: str, n_p: int, n_q: int) -> dict[str, object]:
        return {
            "replication": replication,
            "alpha": alpha,
            "nP": n_p,
            "nQ": n_q,
            "method": self.method,
            "theta1_error": round(self.theta_error, 4),
            "theta1_coverage": int(self.theta_coverage),
            "theta1_ci_length": round(self.theta_ci_length, 4),
            "wilks_stat_p2": "" if self.wilks_stat_p2 is None else round(self.wilks_stat_p2, 4),
            "wilks_reject_95_p2": ""
            if self.wilks_reject_95_p2 is None
            else round(self.wilks_reject_95_p2, 4),
        }


class MomentProvider:
    sample_size: int

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def moment_rows(self, theta: float) -> np.ndarray:
        return np.vstack(self.moment_groups(theta))

    def mean_moment(self, theta: float) -> np.ndarray:
        groups = self.moment_groups(theta)
        out = groups[0].mean(axis=0)
        for group in groups[1:]:
            out = out + group.mean(axis=0)
        return out


class OMDDWProvider(MomentProvider):
    def __init__(self, sample: SampleBundle, nuisance: CrossFittedNuisance):
        self.sample = sample
        self.nuisance = nuisance
        self.sample_size = len(sample.x_source) + len(sample.x_target)

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        a, b = self.nuisance.orthogonal_parts(theta)
        return a, b


class OMOracleProvider(MomentProvider):
    def __init__(self, sample: SampleBundle):
        self.sample = sample
        self.sample_size = len(sample.x_source) + len(sample.x_target)

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        r = density_ratio(self.sample.x_source, self.sample.scenario, self.sample.alpha)
        m_source = oracle_m_theta(self.sample.x_source, theta)
        m_target = oracle_m_theta(self.sample.x_target, theta)
        g_source = moment_vector(self.sample.x_source, self.sample.y_source, theta)
        a = r[:, None] * (g_source - m_source)
        return a, m_target


class DRWProvider(MomentProvider):
    def __init__(self, sample: SampleBundle, nuisance: CrossFittedNuisance):
        self.sample = sample
        self.r_source = nuisance.source_density_ratio()
        self.sample_size = len(sample.x_source)

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        g = moment_vector(self.sample.x_source, self.sample.y_source, theta)
        return (self.r_source[:, None] * g,)


class SourceProvider(MomentProvider):
    def __init__(self, sample: SampleBundle):
        self.sample = sample
        self.sample_size = len(sample.x_source)

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        return (moment_vector(self.sample.x_source, self.sample.y_source, theta),)


class TargetProvider(MomentProvider):
    def __init__(self, sample: SampleBundle):
        self.sample = sample
        self.sample_size = len(sample.x_target)

    def moment_groups(self, theta: float) -> tuple[np.ndarray, ...]:
        return (moment_vector(self.sample.x_target, self.sample.y_target, theta),)


def fit_all_methods(
    sample: SampleBundle,
    theta0: float,
    seed: int,
    k_folds: int,
    dr_config: NetworkConfig,
    mi_config: MIFitConfig,
    el_config: ELConfig,
) -> list[MethodResult]:
    nuisance = fit_cross_fitted_nuisance(
        x_source=sample.x_source,
        y_source=sample.y_source,
        x_target=sample.x_target,
        seed=seed,
        k_folds=k_folds,
        dr_config=dr_config,
        mi_config=mi_config,
    )

    providers = [
        ("OM-DDW", OMDDWProvider(sample, nuisance), "wilks"),
        ("OM-oracle", OMOracleProvider(sample), "wilks"),
        ("DRW", DRWProvider(sample, nuisance), "normal"),
        ("Source-unadjusted", SourceProvider(sample), "wilks_skip"),
        ("Target-labeled", TargetProvider(sample), "wilks"),
    ]

    results: list[MethodResult] = []
    for name, provider, mode in providers:
        objective = build_objective(provider)
        theta_hat, el_hat = estimate_theta(provider, el_config, objective=objective)
        if mode == "normal":
            ci_low, ci_high = sandwich_ci(provider, theta_hat, el_config)
            wilks = fixed_theta_stat(theta0, theta_hat, provider, objective=objective)
            wilks_rej = wilks_reject(wilks)
        elif mode == "wilks_skip":
            ci_low, ci_high = profile_ci(provider, theta_hat, el_hat, el_config, objective=objective)
            wilks = None
            wilks_rej = None
        else:
            ci_low, ci_high = profile_ci(provider, theta_hat, el_hat, el_config, objective=objective)
            wilks = fixed_theta_stat(theta0, theta_hat, provider, objective=objective)
            wilks_rej = wilks_reject(wilks)
        results.append(
            MethodResult(
                method=name,
                theta_hat=theta_hat,
                ci_low=ci_low,
                ci_high=ci_high,
                theta_error=theta_hat - theta0,
                theta_coverage=float(ci_low <= theta0 <= ci_high),
                theta_ci_length=ci_high - ci_low,
                wilks_stat_p2=wilks,
                wilks_reject_95_p2=wilks_rej,
            )
        )
    return results


RESULT_COLUMNS = [
    "replication",
    "alpha",
    "nP",
    "nQ",
    "method",
    "theta1_error",
    "theta1_coverage",
    "theta1_ci_length",
    "wilks_stat_p2",
    "wilks_reject_95_p2",
]


def write_results_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_config(
    config: SimulationConfig,
    dr_config: NetworkConfig,
    mi_config: MIFitConfig,
    el_config: ELConfig,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for alpha in config.alpha_values:
        theta0 = theta_truth(config.scenario, alpha, method=config.truth_method)
        for n_p in config.n_p_values:
            n_q = config.n_q(n_p)
            for replication in range(1, config.replications + 1):
                alpha_code = sum(ord(ch) for ch in str(alpha))
                seed = config.seed + 100000 * replication + 1000 * n_p + alpha_code
                sample = simulate_sample(
                    scenario=config.scenario,
                    n_p=n_p,
                    n_q=n_q,
                    d=config.dimension,
                    alpha=alpha,
                    seed=seed,
                )
                fitted = fit_all_methods(
                    sample=sample,
                    theta0=theta0,
                    seed=seed,
                    k_folds=config.k_folds,
                    dr_config=dr_config,
                    mi_config=mi_config,
                    el_config=el_config,
                )
                rows.extend(
                    result.to_row(replication=replication, alpha=alpha, n_p=n_p, n_q=n_q)
                    for result in fitted
                )
    write_results_csv(rows, config.output_csv)
    return rows


def ensure_results(
    config: SimulationConfig,
    dr_config: NetworkConfig,
    mi_config: MIFitConfig,
    el_config: ELConfig,
    force: bool = False,
) -> Path:
    if force or not config.output_csv.exists():
        run_config(config, dr_config, mi_config, el_config)
    return config.output_csv


def ensure_result_tables(
    base: Path,
    dr_config: NetworkConfig,
    mi_config: MIFitConfig,
    el_config: ELConfig,
    replications: int = 300,
    force: bool = False,
) -> tuple[Path, Path]:
    s1 = ensure_results(default_s1_config(base, replications), dr_config, mi_config, el_config, force=force)
    s2 = ensure_results(default_s2_config(base, replications), dr_config, mi_config, el_config, force=force)
    return s1, s2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Result_S1.csv and Result_S2.csv for the simulation study."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=300,
        help="Number of Monte Carlo replications for each setting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate results even if the output CSV files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_result_tables(
        base=ROOT,
        dr_config=default_network_config(),
        mi_config=default_mi_config(),
        el_config=default_el_config(),
        replications=args.replications,
        force=args.force,
    )


if __name__ == "__main__":
    main()

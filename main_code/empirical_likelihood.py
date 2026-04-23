from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .config import ELConfig


@dataclass
class ELResult:
    value: float
    lam: np.ndarray
    alphas: np.ndarray


def _solve_alpha(rows: np.ndarray, lam: np.ndarray) -> float:
    proj = rows @ lam
    if not np.all(np.isfinite(proj)):
        raise RuntimeError("invalid alpha projection")
    lower = max(1e-8, float(-proj.min()) + 1e-8)

    def fn(alpha: float) -> float:
        return float(np.sum(1.0 / (alpha + proj)) - 1.0)

    lower_value = fn(lower)
    if not np.isfinite(lower_value):
        raise RuntimeError("invalid alpha lower bracket")
    if lower_value <= 0.0:
        return lower

    upper = max(float(len(rows)), lower * 2.0)
    value = fn(upper)
    for _ in range(60):
        if np.isfinite(value) and value <= 0.0:
            break
        upper *= 2.0
        value = fn(upper)
    if not np.isfinite(value) or value > 0.0:
        raise RuntimeError("alpha root bracketing failed")
    root = optimize.root_scalar(fn, bracket=(lower, upper), method="brentq")
    return float(root.root)


def _group_contribution(rows: np.ndarray, lam: np.ndarray) -> tuple[float, np.ndarray, float]:
    alpha = _solve_alpha(rows, lam)
    denom = alpha + rows @ lam
    moment = (rows / denom[:, None]).sum(axis=0)
    value = float(np.log(denom).sum())
    return alpha, moment, value


def solve_el(groups: tuple[np.ndarray, ...], start: np.ndarray | None = None) -> ELResult:
    dim = groups[0].shape[1]
    if start is None:
        start = np.zeros(dim, dtype=float)

    def residual(lam: np.ndarray) -> np.ndarray:
        total = np.zeros(dim, dtype=float)
        try:
            for rows in groups:
                _, moment, _ = _group_contribution(rows, lam)
                total += moment
            return total
        except RuntimeError:
            return np.full(dim, 1e6, dtype=float)

    starts = [np.asarray(start, dtype=float), np.zeros(dim, dtype=float)]
    for scale in (0.05, 0.1):
        for axis in range(dim):
            plus = np.zeros(dim, dtype=float)
            minus = np.zeros(dim, dtype=float)
            plus[axis] = scale
            minus[axis] = -scale
            starts.extend([plus, minus])

    best: ELResult | None = None
    best_norm = np.inf
    for lam0 in starts:
        root = optimize.root(residual, x0=lam0, method="hybr")
        lam = np.asarray(root.x, dtype=float)
        norm = float(np.linalg.norm(residual(lam)))
        if not np.isfinite(norm):
            continue
        if norm < best_norm:
            try:
                alphas = []
                value = 0.0
                for rows in groups:
                    alpha, _, part = _group_contribution(rows, lam)
                    alphas.append(alpha)
                    value += part
                best = ELResult(value=value, lam=lam, alphas=np.asarray(alphas, dtype=float))
                best_norm = norm
            except RuntimeError:
                continue
        if root.success and norm < 1e-8:
            break

    if best is None or best_norm > 1e-5:
        raise RuntimeError("empirical likelihood solve failed")
    return best


class ELObjective:
    def __init__(self, provider) -> None:
        self.provider = provider
        self.cache: dict[float, ELResult] = {}

    def _closest_start(self, theta: float) -> np.ndarray | None:
        if not self.cache:
            return None
        key = min(self.cache, key=lambda item: abs(item - theta))
        return self.cache[key].lam

    def solve(self, theta: float) -> ELResult:
        key = round(float(theta), 10)
        if key in self.cache:
            return self.cache[key]
        result = solve_el(self.provider.moment_groups(float(theta)), start=self._closest_start(float(theta)))
        self.cache[key] = result
        return result

    def value(self, theta: float) -> float:
        try:
            return self.solve(theta).value
        except Exception:
            return np.inf


def build_objective(provider) -> ELObjective:
    return ELObjective(provider)


def estimate_theta(provider, config: ELConfig, objective: ELObjective | None = None) -> tuple[float, float]:
    objective = build_objective(provider) if objective is None else objective
    grid = np.linspace(config.theta_bounds[0], config.theta_bounds[1], config.grid_size)
    norms = np.array([np.linalg.norm(provider.mean_moment(float(theta))) for theta in grid], dtype=float)
    best_pos = int(np.argmin(norms))

    if best_pos == 0:
        left, right = float(grid[0]), float(grid[1])
    elif best_pos == len(grid) - 1:
        left, right = float(grid[-2]), float(grid[-1])
    else:
        left, right = float(grid[best_pos - 1]), float(grid[best_pos + 1])

    result = optimize.minimize_scalar(
        objective.value,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 60},
    )
    if not result.success or not np.isfinite(result.fun):
        theta_grid = grid[[max(best_pos - 1, 0), best_pos, min(best_pos + 1, len(grid) - 1)]]
        values = np.array([objective.value(float(theta)) for theta in theta_grid], dtype=float)
        theta_hat = float(theta_grid[int(np.argmin(values))])
        el_hat = float(values.min())
        return theta_hat, el_hat

    theta_hat = float(result.x)
    el_hat = float(result.fun)
    return theta_hat, el_hat


def profile_ci(
    provider,
    theta_hat: float,
    el_hat: float,
    config: ELConfig,
    objective: ELObjective | None = None,
) -> tuple[float, float]:
    objective = build_objective(provider) if objective is None else objective
    objective.solve(theta_hat)
    cutoff = stats.chi2.ppf(config.ci_level, df=1) / 2.0

    def boundary_function(theta: float) -> float:
        return float(objective.value(theta) - el_hat - cutoff)

    left = _find_boundary(
        boundary_function,
        theta_hat,
        config.theta_bounds[0],
        direction="left",
        grid_size=config.grid_size,
    )
    right = _find_boundary(
        boundary_function,
        theta_hat,
        config.theta_bounds[1],
        direction="right",
        grid_size=config.grid_size,
    )
    return left, right


def _find_boundary(
    fn,
    center: float,
    bound: float,
    direction: str,
    grid_size: int,
) -> float:
    if direction == "left":
        grid = np.linspace(bound, center, grid_size)
    else:
        grid = np.linspace(center, bound, grid_size)
    values = np.array([fn(float(point)) for point in grid], dtype=float)
    if direction == "left":
        values = values[::-1]
        grid = grid[::-1]
    for i in range(len(grid) - 1):
        left, right = float(grid[i]), float(grid[i + 1])
        f_left, f_right = values[i], values[i + 1]
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left <= 0.0 <= f_right:
            root = optimize.root_scalar(fn, bracket=(left, right), method="brentq")
            return float(root.root)
    return float(bound)


def fixed_theta_stat(
    theta0: float,
    theta_hat: float,
    provider,
    objective: ELObjective | None = None,
) -> float:
    objective = build_objective(provider) if objective is None else objective
    objective.solve(theta_hat)
    return 2.0 * (float(objective.value(theta0)) - float(objective.value(theta_hat)))


def wilks_reject(statistic: float, level: float = 0.95, df: int = 2) -> float:
    return float(statistic > stats.chi2.ppf(level, df=df))

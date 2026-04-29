from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .config import ELConfig


def sandwich_ci(provider, theta_hat: float, config: ELConfig) -> tuple[float, float]:
    step = config.finite_diff_step
    moments = provider.moment_rows(theta_hat)
    omega = np.cov(moments, rowvar=False, bias=False)
    if omega.ndim == 0:
        omega = np.array([[float(omega)]])
    omega_inv = np.linalg.pinv(omega)

    g_plus = provider.moment_rows(theta_hat + step).mean(axis=0)
    g_minus = provider.moment_rows(theta_hat - step).mean(axis=0)
    jac = ((g_plus - g_minus) / (2.0 * step)).reshape(-1, 1)
    information = float(jac.T @ omega_inv @ jac)
    variance = 1.0 / max(information * provider.sample_size, 1e-8)
    half_length = stats.norm.ppf(0.5 + config.ci_level / 2.0) * math.sqrt(variance)
    return theta_hat - half_length, theta_hat + half_length

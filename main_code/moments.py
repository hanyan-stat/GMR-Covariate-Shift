from __future__ import annotations

import numpy as np


def moment_vector(x: np.ndarray, y: np.ndarray, theta: float) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    g1 = x[..., 0] * (y - x[..., 0] * theta)
    g2 = x[..., 0] * (0.5 - (y <= x[..., 0] * theta).astype(float))
    return np.stack([g1, g2], axis=-1)


def mean_moment(x: np.ndarray, y: np.ndarray, theta: float) -> np.ndarray:
    return moment_vector(x, y, theta).mean(axis=0)

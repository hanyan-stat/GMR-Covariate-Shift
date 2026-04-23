from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    out = scale.copy()
    out[out < 1e-8] = 1.0
    return out


def fit_x_standardizer(x_source: np.ndarray, x_target: np.ndarray) -> Standardizer:
    pooled = np.vstack([x_source, x_target])
    return Standardizer(mean=pooled.mean(axis=0), scale=_safe_scale(pooled.std(axis=0)))


def fit_y_standardizer(y_source: np.ndarray) -> Standardizer:
    y = np.asarray(y_source).reshape(-1, 1)
    std = y.std(axis=0)
    return Standardizer(mean=y.mean(axis=0), scale=_safe_scale(std))


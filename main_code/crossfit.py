from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MIFitConfig, NetworkConfig
from .moments import moment_vector
from .nuisance_dr import RatioFit, candidate_upper_bounds, fit_ratio_model
from .nuisance_mi import (
    MIFit,
    MThetaCache,
    build_m_theta_cache,
    evaluate_m_theta_from_cache,
    fit_mi_density,
)
from .preprocess import Standardizer, fit_x_standardizer


@dataclass
class DRFoldFit:
    fit: RatioFit
    x_scaler: Standardizer

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_std = self.x_scaler.transform(x)
        return self.fit.predict(x_std)


@dataclass
class FoldNuisance:
    holdout_index: np.ndarray
    dr: DRFoldFit
    mi: MIFit
    holdout_cache: MThetaCache
    target_cache: MThetaCache


class CrossFittedNuisance:
    def __init__(
        self,
        x_source: np.ndarray,
        y_source: np.ndarray,
        x_target: np.ndarray,
        folds: list[FoldNuisance],
    ) -> None:
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target
        self.folds = folds
        self._r_source: Optional[np.ndarray] = None
        self._om_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def source_density_ratio(self) -> np.ndarray:
        if self._r_source is None:
            out = np.zeros(len(self.x_source), dtype=float)
            for fold in self.folds:
                out[fold.holdout_index] = fold.dr.predict(self.x_source[fold.holdout_index])
            self._r_source = out
        return self._r_source

    def orthogonal_parts(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(theta), 10)
        if key in self._om_cache:
            return self._om_cache[key]

        a = np.zeros((len(self.x_source), 2), dtype=float)
        for fold in self.folds:
            idx = fold.holdout_index
            r = fold.dr.predict(self.x_source[idx])
            g = moment_vector(self.x_source[idx], self.y_source[idx], theta)
            m = evaluate_m_theta_from_cache(fold.holdout_cache, theta)
            a[idx] = r[:, None] * (g - m)
        b = np.mean(
            [evaluate_m_theta_from_cache(fold.target_cache, theta) for fold in self.folds],
            axis=0,
        )
        self._om_cache[key] = (a, b)
        return a, b


def _make_folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    perm = rng.permutation(n)
    return [chunk.astype(int) for chunk in np.array_split(perm, k)]


def fit_cross_fitted_nuisance(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    seed: int,
    k_folds: int,
    dr_config: NetworkConfig,
    mi_config: MIFitConfig,
) -> CrossFittedNuisance:
    rng = np.random.default_rng(seed)
    folds = _make_folds(len(x_source), k_folds, rng)
    fitted: list[FoldNuisance] = []
    for fold_id, holdout in enumerate(folds):
        mask = np.ones(len(x_source), dtype=bool)
        mask[holdout] = False
        x_train = x_source[mask]
        y_train = y_source[mask]

        x_scaler = fit_x_standardizer(x_train, x_target)
        dr_fit = fit_ratio_model(
            source_features=x_scaler.transform(x_train),
            target_features=x_scaler.transform(x_target),
            seed=seed + 100 * (fold_id + 1),
            config=dr_config,
            upper_bounds=candidate_upper_bounds(len(x_train)),
            normalize=True,
        )
        mi_fit = fit_mi_density(
            x_source=x_train,
            y_source=y_train,
            x_target=x_target,
            seed=seed + 1000 * (fold_id + 1),
            config=mi_config,
        )
        holdout_cache = build_m_theta_cache(mi_fit, x_source[holdout])
        target_cache = build_m_theta_cache(mi_fit, x_target)
        fitted.append(
            FoldNuisance(
                holdout_index=holdout,
                dr=DRFoldFit(fit=dr_fit, x_scaler=x_scaler),
                mi=mi_fit,
                holdout_cache=holdout_cache,
                target_cache=target_cache,
            )
        )
    return CrossFittedNuisance(
        x_source=x_source,
        y_source=y_source,
        x_target=x_target,
        folds=fitted,
    )

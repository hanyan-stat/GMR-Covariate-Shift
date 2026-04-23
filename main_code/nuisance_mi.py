from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .config import MIFitConfig
from .nuisance_dr import RatioFit, RatioNet, resolve_torch_device
from .preprocess import Standardizer, fit_x_standardizer, fit_y_standardizer


@dataclass
class MIFit:
    rho_left: RatioFit
    rho_right: RatioFit
    x_scaler: Standardizer
    y_scaler: Standardizer
    x_source_raw: np.ndarray
    y_reference_raw: np.ndarray
    config: MIFitConfig


@dataclass
class MThetaCache:
    x1: np.ndarray
    y_sorted: np.ndarray
    weight_sum: np.ndarray
    weighted_y_sum: np.ndarray
    cumulative_weights: np.ndarray
    scale: float


def _split_train_validation(
    n: int, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 5 or fraction <= 0.0:
        idx = np.arange(n)
        return idx, idx[:0]
    n_val = max(1, int(round(n * fraction)))
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def _fixed_pseudo_pairs(
    x_half: np.ndarray,
    y_other: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    choice = rng.choice(len(y_other), size=len(x_half), replace=len(y_other) < len(x_half))
    return np.column_stack([x_half, y_other[choice]])


def _train_half_model(
    x_half: np.ndarray,
    y_half: np.ndarray,
    y_other: np.ndarray,
    config: MIFitConfig,
    seed: int,
) -> RatioFit:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(config.network.device)
    upper = config.upper_clip
    model = RatioNet(
        input_dim=x_half.shape[1] + 1,
        widths=tuple(config.network.widths),
        lower=config.network.lower_clip,
        upper=upper,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.network.learning_rate,
        weight_decay=config.network.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.network.lr_factor,
        patience=config.network.lr_patience,
    )

    train_idx, val_idx = _split_train_validation(
        len(x_half), config.network.validation_fraction, rng
    )
    x_train = x_half[train_idx]
    y_train = y_half[train_idx]
    x_val = x_half[val_idx] if len(val_idx) else x_train
    y_val = y_half[val_idx] if len(val_idx) else y_train
    pseudo_val = _fixed_pseudo_pairs(x_val, y_other, rng)
    real_val = np.column_stack([x_val, y_val])

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            pseudo = torch.as_tensor(pseudo_val, dtype=torch.float32, device=device)
            real = torch.as_tensor(real_val, dtype=torch.float32, device=device)
            loss = model(pseudo).square().mean() - 2.0 * model(real).mean()
        return float(loss.cpu().item())

    best_state = copy.deepcopy(model.state_dict())
    best_loss = validation_loss()
    stale_epochs = 0
    steps_per_epoch = max(1, int(np.ceil(len(x_train) / config.network.batch_size)))

    for _ in range(config.network.max_epochs):
        model.train()
        for _ in range(steps_per_epoch):
            batch_idx = rng.integers(
                0, len(x_train), size=min(config.network.batch_size, len(x_train))
            )
            other_idx = rng.integers(
                0, len(y_other), size=min(config.network.batch_size, len(x_train))
            )
            real_batch = np.column_stack([x_train[batch_idx], y_train[batch_idx]])
            pseudo_batch = np.column_stack([x_train[batch_idx], y_other[other_idx]])
            pseudo_tensor = torch.as_tensor(pseudo_batch, dtype=torch.float32, device=device)
            real_tensor = torch.as_tensor(real_batch, dtype=torch.float32, device=device)
            loss = model(pseudo_tensor).square().mean() - 2.0 * model(real_tensor).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.network.grad_clip_norm)
            optimizer.step()
        val_loss = validation_loss()
        scheduler.step(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.network.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    return RatioFit(
        model=model,
        normalization=1.0,
        upper_clip=upper,
        feature_dim=x_half.shape[1] + 1,
        config=config.network,
    )


def fit_mi_density(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    seed: int,
    config: MIFitConfig,
) -> MIFit:
    rng = np.random.default_rng(seed)
    x_scaler = fit_x_standardizer(x_source, x_target)
    y_scaler = fit_y_standardizer(y_source)
    x_std = x_scaler.transform(x_source)
    y_std = y_scaler.transform(y_source.reshape(-1, 1)).ravel()
    perm = rng.permutation(len(x_source))
    midpoint = len(perm) // 2
    left_idx = perm[:midpoint]
    right_idx = perm[midpoint:]
    if len(left_idx) == 0 or len(right_idx) == 0:
        raise ValueError("source sample too small for mutual-information fitting")

    if config.max_reference_samples is None or config.max_reference_samples >= len(y_source):
        y_reference_raw = y_source
    else:
        ref_idx = rng.choice(len(y_source), size=config.max_reference_samples, replace=False)
        y_reference_raw = y_source[np.sort(ref_idx)]

    rho_left = _train_half_model(
        x_half=x_std[left_idx],
        y_half=y_std[left_idx],
        y_other=y_std[right_idx],
        config=config,
        seed=seed + 17,
    )
    rho_right = _train_half_model(
        x_half=x_std[right_idx],
        y_half=y_std[right_idx],
        y_other=y_std[left_idx],
        config=config,
        seed=seed + 31,
    )
    return MIFit(
        rho_left=rho_left,
        rho_right=rho_right,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_source_raw=x_source,
        y_reference_raw=y_reference_raw,
        config=config,
    )


def _predict_rho_block(fit: MIFit, x_std: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    features = np.column_stack([x_std, y_std])
    pred_left = fit.rho_left.predict(features, batch_size=fit.config.batch_size)
    pred_right = fit.rho_right.predict(features, batch_size=fit.config.batch_size)
    return 0.5 * (pred_left + pred_right)


def build_m_theta_cache(fit: MIFit, x_eval: np.ndarray) -> MThetaCache:
    x_eval = np.asarray(x_eval)
    y_raw = fit.y_reference_raw
    y_std = fit.y_scaler.transform(y_raw.reshape(-1, 1)).ravel()
    order = np.argsort(y_raw)
    y_sorted = y_raw[order]
    y_sorted_std = y_std[order]

    x_eval_std = fit.x_scaler.transform(x_eval)
    weights = np.zeros((len(x_eval), len(y_sorted)), dtype=float)
    x_batch_size = fit.config.eval_batch_size
    for start in range(0, len(x_eval), x_batch_size):
        stop = min(len(x_eval), start + x_batch_size)
        x_block = x_eval_std[start:stop]
        repeated_x = np.repeat(x_block, repeats=len(y_sorted_std), axis=0)
        tiled_y = np.tile(y_sorted_std, len(x_block))
        rho = _predict_rho_block(fit, repeated_x, tiled_y).reshape(len(x_block), -1)
        weights[start:stop] = rho

    return MThetaCache(
        x1=np.asarray(x_eval[:, 0], dtype=float),
        y_sorted=y_sorted,
        weight_sum=weights.sum(axis=1),
        weighted_y_sum=weights @ y_sorted,
        cumulative_weights=np.cumsum(weights, axis=1),
        scale=1.0 / len(y_sorted),
    )


def evaluate_m_theta_from_cache(cache: MThetaCache, theta: float) -> np.ndarray:
    threshold = cache.x1 * theta
    idx = np.searchsorted(cache.y_sorted, threshold, side="right") - 1
    cdf = np.zeros(len(cache.x1), dtype=float)
    valid = idx >= 0
    if np.any(valid):
        cdf[valid] = cache.cumulative_weights[np.nonzero(valid)[0], idx[valid]]

    m1 = cache.x1 * (cache.weighted_y_sum - cache.x1 * theta * cache.weight_sum)
    m2 = cache.x1 * (0.5 * cache.weight_sum - cdf)
    return cache.scale * np.column_stack([m1, m2])


def evaluate_m_theta(
    fit: MIFit,
    x_eval: np.ndarray,
    theta: float,
    cache: Optional[MThetaCache] = None,
) -> np.ndarray:
    local_cache = cache if cache is not None else build_m_theta_cache(fit, x_eval)
    return evaluate_m_theta_from_cache(local_cache, theta)

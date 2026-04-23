from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .config import NetworkConfig


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class RatioNet(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...], lower: float, upper: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for width in widths:
            layers.append(nn.Linear(dim, width))
            layers.append(nn.ReLU())
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.lower = lower
        self.upper = upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.softplus(self.body(x)).squeeze(-1)
        return torch.clamp(out, min=self.lower, max=self.upper)


@dataclass
class RatioFit:
    model: RatioNet
    normalization: float
    upper_clip: float
    feature_dim: int
    config: NetworkConfig

    def predict(self, features: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.config.batch_size
        device = resolve_torch_device(self.config.device)
        self.model.to(device)
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                stop = min(len(features), start + batch_size)
                x = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
                pred = self.model(x).cpu().numpy()
                outputs.append(pred)
        return self.normalization * np.concatenate(outputs, axis=0)


def candidate_upper_bounds(n_source: int) -> list[float]:
    vals = [10.0]
    vals.extend(10.0 * n_source ** power for power in (0.2, 0.4, 0.6, 0.8))
    out = []
    for val in vals:
        if not out or abs(val - out[-1]) > 1e-8:
            out.append(float(val))
    return out


def _split_train_validation(
    n: int, fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 5 or fraction <= 0.0:
        idx = np.arange(n)
        return idx, idx[:0]
    n_val = max(1, int(round(n * fraction)))
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def _loss_from_arrays(model: RatioNet, source: np.ndarray, target: np.ndarray) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_source = torch.as_tensor(source, dtype=torch.float32, device=device)
        x_target = torch.as_tensor(target, dtype=torch.float32, device=device)
        loss = model(x_source).square().mean() - 2.0 * model(x_target).mean()
    return float(loss.cpu().item())


def _train_single_model(
    source: np.ndarray,
    target: np.ndarray,
    upper_clip: float,
    config: NetworkConfig,
    seed: int,
) -> tuple[RatioNet, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_torch_device(config.device)
    model = RatioNet(
        input_dim=source.shape[1],
        widths=tuple(config.widths),
        lower=config.lower_clip,
        upper=upper_clip,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    src_train_idx, src_val_idx = _split_train_validation(
        len(source), config.validation_fraction, rng
    )
    tgt_train_idx, tgt_val_idx = _split_train_validation(
        len(target), config.validation_fraction, rng
    )
    source_train = source[src_train_idx]
    target_train = target[tgt_train_idx]
    source_val = source[src_val_idx] if len(src_val_idx) else source_train
    target_val = target[tgt_val_idx] if len(tgt_val_idx) else target_train

    best_state = copy.deepcopy(model.state_dict())
    best_loss = _loss_from_arrays(model, source_val, target_val)
    stale_epochs = 0
    steps_per_epoch = max(1, int(np.ceil(max(len(source_train), len(target_train)) / config.batch_size)))

    for _ in range(config.max_epochs):
        model.train()
        for _ in range(steps_per_epoch):
            src_idx = rng.integers(0, len(source_train), size=min(config.batch_size, len(source_train)))
            tgt_idx = rng.integers(0, len(target_train), size=min(config.batch_size, len(target_train)))
            src_batch = torch.as_tensor(source_train[src_idx], dtype=torch.float32, device=device)
            tgt_batch = torch.as_tensor(target_train[tgt_idx], dtype=torch.float32, device=device)
            loss = model(src_batch).square().mean() - 2.0 * model(tgt_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
        val_loss = _loss_from_arrays(model, source_val, target_val)
        scheduler.step(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    return model, best_loss


def fit_ratio_model(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int,
    config: NetworkConfig,
    upper_bounds: list[float],
    normalize: bool,
) -> RatioFit:
    best_model: Optional[RatioNet] = None
    best_loss = np.inf
    best_upper = upper_bounds[0]
    for offset, upper in enumerate(upper_bounds):
        model, loss = _train_single_model(
            source=source_features,
            target=target_features,
            upper_clip=upper,
            config=config,
            seed=seed + 1009 * offset,
        )
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_upper = upper
    if best_model is None:
        raise RuntimeError("ratio training failed")

    fit = RatioFit(
        model=best_model,
        normalization=1.0,
        upper_clip=best_upper,
        feature_dim=source_features.shape[1],
        config=config,
    )
    if normalize:
        pred_mean = fit.predict(source_features).mean()
        fit.normalization = 1.0 / max(pred_mean, 1e-8)
    return fit

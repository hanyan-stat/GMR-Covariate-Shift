from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class NetworkConfig:
    widths: Sequence[int] = (256, 128, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 300
    validation_fraction: float = 0.1
    lr_patience: int = 15
    lr_factor: float = 0.5
    early_stopping_patience: int = 30
    grad_clip_norm: float = 5.0
    lower_clip: float = 1e-4
    device: str = "auto"


@dataclass(frozen=True)
class MIFitConfig:
    network: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(learning_rate=5e-4)
    )
    upper_clip: float = 50.0
    batch_size: int = 256
    eval_batch_size: int = 128
    max_reference_samples: Optional[int] = None


@dataclass(frozen=True)
class ELConfig:
    theta_bounds: tuple[float, float] = (-4.0, 4.0)
    grid_size: int = 41
    ci_level: float = 0.95
    finite_diff_step: float = 1e-4


def default_network_config() -> NetworkConfig:
    return NetworkConfig(
        widths=(128, 64, 32),
        learning_rate=8e-4,
        batch_size=256,
        max_epochs=40,
        validation_fraction=0.1,
        lr_patience=5,
        early_stopping_patience=10,
    )


def default_mi_config() -> MIFitConfig:
    return MIFitConfig(
        network=NetworkConfig(
            widths=(32, 16),
            learning_rate=5e-4,
            batch_size=256,
            max_epochs=8,
            validation_fraction=0.2,
            lr_patience=2,
            early_stopping_patience=4,
        ),
        eval_batch_size=128,
        max_reference_samples=512,
    )


def default_el_config() -> ELConfig:
    return ELConfig()

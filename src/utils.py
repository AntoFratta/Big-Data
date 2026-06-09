"""Shared reproducibility and reconstruction-inference utilities."""

import random
from typing import Callable

import numpy as np
import torch

from config import DEVICE, RANDOM_SEED
from losses import mae_reconstruction, mse_reconstruction
from model import Autoencoder


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Configure deterministic random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_reconstruction_function(loss_fn: str) -> Callable:
    """Return the per-sample reconstruction function for a supported loss."""
    if loss_fn == "mae":
        return mae_reconstruction
    if loss_fn == "mse":
        return mse_reconstruction
    raise ValueError(f"Unsupported reconstruction loss: {loss_fn!r}.")


def compute_reconstruction_errors(
    model: Autoencoder,
    features: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
) -> np.ndarray:
    """Compute per-sample reconstruction errors in device-sized batches."""
    reconstruction_fn = get_reconstruction_function(loss_fn)
    model.eval()
    model.to(DEVICE)

    batches = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            batches.append(reconstruction_fn(batch, model(batch)).cpu().numpy())

    if not batches:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(batches)


def compute_best_reconstruction(
    autoencoders: dict[int, Autoencoder],
    features: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum error and best-reconstructing class for each sample."""
    min_errors = np.full(len(features), np.inf, dtype=np.float32)
    best_classes = np.zeros(len(features), dtype=np.int64)

    for cls, model in autoencoders.items():
        errors = compute_reconstruction_errors(
            model,
            features,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )
        better = errors < min_errors
        min_errors[better] = errors[better]
        best_classes[better] = cls

    return min_errors, best_classes


def apply_class_thresholds(
    min_errors: np.ndarray,
    best_classes: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Apply the class-specific novelty threshold to each best reconstruction."""
    selected_thresholds = np.fromiter(
        (thresholds[int(cls)] for cls in best_classes),
        dtype=np.float64,
        count=len(best_classes),
    )
    return np.where(min_errors > selected_thresholds, -1, best_classes)

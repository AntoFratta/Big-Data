"""
Loss functions used by the novelty-detection experiments.

The reconstruction losses return per-sample errors. The custom losses implement
the thesis objective for Experiment 3, where the target class is encouraged to
reconstruct well and the other known classes are discouraged from doing so.
"""

import torch


_EPS = 1e-3


def mae_reconstruction(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return the per-sample MAE reconstruction error."""
    return (x - x_hat).abs().mean(dim=1)


def mse_reconstruction(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return the per-sample MSE reconstruction error."""
    return ((x - x_hat) ** 2).mean(dim=1)


def custom_loss_mae(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    Compute the MAE-based custom loss used for Experiment 3.

    Samples belonging to target_class are optimized with the reconstruction
    error. Samples from other known classes use the reciprocal reconstruction
    error, penalizing good reconstructions by the wrong autoencoder.
    """
    errors = mae_reconstruction(x, x_hat)
    target_mask = (labels == target_class).float()
    loss_per_sample = target_mask * errors + (1.0 - target_mask) / (errors + _EPS)
    return loss_per_sample.mean()


def custom_loss_mse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """Compute the MSE-based custom loss used for Experiment 3."""
    errors = mse_reconstruction(x, x_hat)
    target_mask = (labels == target_class).float()
    loss_per_sample = target_mask * errors + (1.0 - target_mask) / (errors + _EPS)
    return loss_per_sample.mean()

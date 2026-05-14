"""
losses.py — Funzioni di loss per gli esperimenti di novelty detection.

Eq. 3.1  MAE di ricostruzione (per-sample, poi media sul batch)
Eq. 3.2  MSE di ricostruzione (per-sample, poi media sul batch)
Eq. 3.3  Custom loss MAE  (usata nell'Esperimento 3)
Eq. 3.4  Custom loss MSE  (usata nell'Esperimento 3)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Costante epsilon per evitare divisione per zero nelle custom loss
# ---------------------------------------------------------------------------
_EPS = 1e-3


# ---------------------------------------------------------------------------
# Eq. 3.1 — MAE di ricostruzione
# ---------------------------------------------------------------------------
def mae_reconstruction(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Calcola la MAE di ricostruzione per ogni campione nel batch.

    Parameters
    ----------
    x     : (N, D) — input originale
    x_hat : (N, D) — ricostruzione dell'autoencoder

    Returns
    -------
    errors : (N,) — MAE per ogni campione
    """
    return (x - x_hat).abs().mean(dim=1)


# ---------------------------------------------------------------------------
# Eq. 3.2 — MSE di ricostruzione
# ---------------------------------------------------------------------------
def mse_reconstruction(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Calcola la MSE di ricostruzione per ogni campione nel batch.

    Parameters
    ----------
    x     : (N, D) — input originale
    x_hat : (N, D) — ricostruzione dell'autoencoder

    Returns
    -------
    errors : (N,) — MSE per ogni campione
    """
    return ((x - x_hat) ** 2).mean(dim=1)


# ---------------------------------------------------------------------------
# Eq. 3.3 — Custom loss MAE (Esperimento 3)
# ---------------------------------------------------------------------------
def custom_loss_mae(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    Custom loss basata su MAE (Eq. 3.3 della tesi).

    Per ogni campione i nel batch:
      - se labels[i] == target_class  →  loss_i = MAE(x_i, x̂_i)
      - altrimenti                    →  loss_i = 1 / (MAE(x_i, x̂_i) + eps)

    Parameters
    ----------
    x            : (N, D) — input originale
    x_hat        : (N, D) — ricostruzione
    labels       : (N,)   — classe reale di ogni campione
    target_class : int    — classe a cui questo autoencoder è specializzato

    Returns
    -------
    loss : scalare — media della loss sul batch
    """
    errors = mae_reconstruction(x, x_hat)          # (N,)

    mask = (labels == target_class).float()         # 1 se same class, 0 altrimenti

    loss_per_sample = mask * errors + (1.0 - mask) * (1.0 / (errors + _EPS))

    return loss_per_sample.mean()


# ---------------------------------------------------------------------------
# Eq. 3.4 — Custom loss MSE (Esperimento 3)
# ---------------------------------------------------------------------------
def custom_loss_mse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """
    Custom loss basata su MSE (Eq. 3.4 della tesi).

    Per ogni campione i nel batch:
      - se labels[i] == target_class  →  loss_i = MSE(x_i, x̂_i)
      - altrimenti                    →  loss_i = 1 / (MSE(x_i, x̂_i) + eps)

    Parameters
    ----------
    x            : (N, D) — input originale
    x_hat        : (N, D) — ricostruzione
    labels       : (N,)   — classe reale di ogni campione
    target_class : int    — classe a cui questo autoencoder è specializzato

    Returns
    -------
    loss : scalare — media della loss sul batch
    """
    errors = mse_reconstruction(x, x_hat)          # (N,)

    mask = (labels == target_class).float()         # 1 se same class, 0 altrimenti

    loss_per_sample = mask * errors + (1.0 - mask) * (1.0 / (errors + _EPS))

    return loss_per_sample.mean()

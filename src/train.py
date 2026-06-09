"""
Training utilities for the three PLAsTiCC novelty-detection experiments.

Experiment 1 trains one global autoencoder on all known classes. Experiment 2
trains one autoencoder per known class. Experiment 3 keeps the same per-class
architectures and trains each model with the custom objective.
"""

import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    EXP1_CONFIG,
    EXP2_CONFIG,
    EXP3_CONFIG,
    KNOWN_CLASSES,
    LEARNING_RATE,
    MODELS_DIR,
    RESULTS_DIR,
    SIGMA_FACTOR,
    WEIGHT_DECAY,
)
from losses import (
    custom_loss_mae,
    custom_loss_mse,
)
from model import Autoencoder
from utils import (
    apply_class_thresholds,
    compute_best_reconstruction,
    compute_reconstruction_errors,
    get_reconstruction_function,
)


def _make_loader(X: np.ndarray, y: np.ndarray = None, shuffle: bool = True) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
    else:
        dataset = TensorDataset(X_t)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_autoencoder(
    model: Autoencoder,
    loader: DataLoader,
    loss_fn: str = "mae",
    epochs: int = EPOCHS,
) -> list:
    """Train an autoencoder with MAE or MSE reconstruction loss."""
    recon_fn = get_reconstruction_function(loss_fn)

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            x_batch = batch[0].to(DEVICE)

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = recon_fn(x_batch, x_hat).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        avg_loss = total_loss / n_samples
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}]  Loss ({loss_fn.upper()}): {avg_loss:.6f}")

    return history


def train_autoencoder_custom(
    model: Autoencoder,
    loader: DataLoader,
    target_class: int,
    loss_fn: str = "mae",
    epochs: int = EPOCHS,
) -> list:
    """Train an autoencoder with the custom MAE or MSE objective."""
    custom_fn = custom_loss_mae if loss_fn == "mae" else custom_loss_mse

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = custom_fn(x_batch, x_hat, y_batch, target_class)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        avg_loss = total_loss / n_samples
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1}/{epochs}]  "
                f"Custom Loss ({loss_fn.upper()}): {avg_loss:.6f}"
            )

    return history


def compute_threshold(
    model: Autoencoder,
    X_class: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
    sigma_factor: float = SIGMA_FACTOR,
) -> float:
    """
    Compute the novelty threshold for one autoencoder.

    threshold = mean(training_errors) + sigma_factor * std(training_errors)
    """
    errors = compute_reconstruction_errors(
        model,
        X_class,
        loss_fn=loss_fn,
        batch_size=batch_size,
    )

    return float(errors.mean() + sigma_factor * errors.std())


def predict_novelty(
    autoencoders: dict,
    thresholds: dict,
    X_test: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
) -> tuple:
    """
    Apply the per-class autoencoder decision rule.

    Each sample is assigned to the autoencoder with the lowest reconstruction
    error, unless that error exceeds the corresponding class threshold.
    """
    min_err, best_cls = compute_best_reconstruction(
        autoencoders,
        X_test,
        loss_fn=loss_fn,
        batch_size=batch_size,
    )

    predictions = apply_class_thresholds(min_err, best_cls, thresholds)

    return predictions, min_err


def run_experiment_1(
    data: dict,
    loss_fn: str = "mae",
    save_results: bool = True,
    evaluate: bool = True,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT 1 - Global AE  |  Loss: {loss_fn.upper()}")
    print(f"{'=' * 60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    known_mask = np.isin(y_train, KNOWN_CLASSES)
    X_known = X_train[known_mask]
    y_known = y_train[known_mask]

    loader = _make_loader(X_known, y_known, shuffle=True)

    cfg = EXP1_CONFIG
    model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

    history = train_autoencoder(model, loader, loss_fn=loss_fn)

    threshold = compute_threshold(model, X_known, loss_fn=loss_fn)
    print(f"  Global threshold: {threshold:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"exp1_{loss_fn}.pt")

    if not evaluate:
        return {
            "experiment": 1,
            "loss_fn": loss_fn,
            "threshold": threshold,
            "history": history,
        }

    train_errors = compute_reconstruction_errors(model, X_known, loss_fn=loss_fn)
    errors = compute_reconstruction_errors(model, X_test, loss_fn=loss_fn)

    predictions = np.where(errors > threshold, -1, 0)

    results = {
        "experiment": 1,
        "loss_fn": loss_fn,
        "threshold": threshold,
        "history": history,
        "train_errors": train_errors.tolist(),
        "errors": errors.tolist(),
        "predictions": predictions.tolist(),
        "y_test": y_test.tolist(),
    }

    if save_results:
        _save_results(results, f"exp1_{loss_fn}")
    return results


def run_experiment_2(
    data: dict,
    loss_fn: str = "mae",
    save_results: bool = True,
    evaluate: bool = True,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT 2 - Per-class AE  |  Loss: {loss_fn.upper()}")
    print(f"{'=' * 60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    autoencoders = {}
    thresholds = {}
    histories = {}

    for cls in KNOWN_CLASSES:
        print(f"\n  -> Class {cls}")

        X_cls = X_train[y_train == cls]
        if len(X_cls) == 0:
            print(f"    WARNING: no samples found for class {cls}; skipping.")
            continue

        cfg = EXP2_CONFIG[cls][loss_fn]
        model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

        loader = _make_loader(X_cls, shuffle=True)
        history = train_autoencoder(model, loader, loss_fn=loss_fn)
        threshold = compute_threshold(model, X_cls, loss_fn=loss_fn)

        print(f"    Class {cls} threshold: {threshold:.6f}")

        autoencoders[cls] = model
        thresholds[cls] = threshold
        histories[cls] = history

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / f"exp2_{loss_fn}_class{cls}.pt")

    if not evaluate:
        return {
            "experiment": 2,
            "loss_fn": loss_fn,
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "histories": {str(k): v for k, v in histories.items()},
        }, autoencoders

    predictions, min_errors = predict_novelty(autoencoders, thresholds, X_test, loss_fn)

    results = {
        "experiment": 2,
        "loss_fn": loss_fn,
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "histories": {str(k): v for k, v in histories.items()},
        "predictions": predictions.tolist(),
        "min_errors": min_errors.tolist(),
        "y_test": y_test.tolist(),
    }

    if save_results:
        _save_results(results, f"exp2_{loss_fn}")
    return results, autoencoders


def run_experiment_3(
    data: dict,
    loss_fn: str = "mae",
    save_results: bool = True,
    evaluate: bool = True,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT 3 - Custom Loss  |  Loss: {loss_fn.upper()}")
    print(f"{'=' * 60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    known_mask = np.isin(y_train, KNOWN_CLASSES)
    X_known = X_train[known_mask]
    y_known = y_train[known_mask]

    full_loader = _make_loader(X_known, y_known, shuffle=True)

    autoencoders = {}
    thresholds = {}
    histories = {}

    for cls in KNOWN_CLASSES:
        print(f"\n  -> Class {cls}")

        cfg = EXP3_CONFIG[cls][loss_fn]
        model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

        history = train_autoencoder_custom(
            model,
            full_loader,
            target_class=cls,
            loss_fn=loss_fn,
        )

        X_cls = X_known[y_known == cls]
        threshold = compute_threshold(model, X_cls, loss_fn=loss_fn)

        print(f"    Class {cls} threshold: {threshold:.6f}")

        autoencoders[cls] = model
        thresholds[cls] = threshold
        histories[cls] = history

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / f"exp3_{loss_fn}_class{cls}.pt")

    if not evaluate:
        return {
            "experiment": 3,
            "loss_fn": loss_fn,
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "histories": {str(k): v for k, v in histories.items()},
        }, autoencoders

    predictions, min_errors = predict_novelty(autoencoders, thresholds, X_test, loss_fn)

    results = {
        "experiment": 3,
        "loss_fn": loss_fn,
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "histories": {str(k): v for k, v in histories.items()},
        "predictions": predictions.tolist(),
        "min_errors": min_errors.tolist(),
        "y_test": y_test.tolist(),
    }

    if save_results:
        _save_results(results, f"exp3_{loss_fn}")
    return results, autoencoders


def _save_results(results: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {path}")

"""
train.py — Training e valutazione degli autoencoder per novelty detection.

Tre esperimenti:
  Exp 1 — un unico AE globale addestrato su tutte le classi note (loss MAE o MSE)
  Exp 2 — un AE per ogni classe nota, addestrato SOLO sui campioni di quella classe
  Exp 3 — un AE per ogni classe nota, addestrato su TUTTI i campioni con custom loss

Soglia per novelty detection:
  threshold_k = mean(err_k su campioni della classe k nel train) + 3 * std(...)
"""

import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    SIGMA_FACTOR,
    MODELS_DIR,
    RESULTS_DIR,
    KNOWN_CLASSES,
    EXP1_CONFIG,
    EXP2_CONFIG,
    EXP3_CONFIG,
)
from data_loader import prepare_data
from model import Autoencoder
from losses import (
    mae_reconstruction,
    mse_reconstruction,
    custom_loss_mae,
    custom_loss_mse,
)


# ---------------------------------------------------------------------------
# Utility: costruisce un DataLoader da array numpy
# ---------------------------------------------------------------------------
def _make_loader(X: np.ndarray, y: np.ndarray = None, shuffle: bool = True) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
    else:
        dataset = TensorDataset(X_t)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def _compute_reconstruction_errors_batched(
    model: Autoencoder,
    X: np.ndarray,
    recon_fn,
    batch_size: int = 8192,
) -> np.ndarray:
    """Calcola gli errori di ricostruzione senza caricare tutto X su DEVICE."""
    model.eval()
    model.to(DEVICE)

    all_errors = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(DEVICE)
            x_hat = model(X_batch)
            errors = recon_fn(X_batch, x_hat).cpu().numpy()
            all_errors.append(errors)

    return np.concatenate(all_errors)


# ---------------------------------------------------------------------------
# Training di un singolo autoencoder (Exp 1 e Exp 2)
# loss_fn deve essere 'mae' oppure 'mse'
# ---------------------------------------------------------------------------
def train_autoencoder(
    model: Autoencoder,
    loader: DataLoader,
    loss_fn: str = "mae",
    epochs: int = EPOCHS,
) -> list:
    """
    Addestra il modello minimizzando MAE o MSE di ricostruzione.
    Il DataLoader può avere tuple (X,) oppure (X, y): in entrambi i casi
    si usa solo X.

    Returns
    -------
    history : lista delle loss medie per epoca
    """
    recon_fn = mae_reconstruction if loss_fn == "mae" else mse_reconstruction

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples  = 0

        for batch in loader:
            x_batch = batch[0].to(DEVICE)   # ignora y se presente

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = recon_fn(x_batch, x_hat).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples  += x_batch.size(0)

        avg_loss = total_loss / n_samples
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  Loss ({loss_fn.upper()}): {avg_loss:.6f}")

    return history


# ---------------------------------------------------------------------------
# Training con custom loss (Exp 3)
# Il DataLoader DEVE contenere tuple (X, y)
# ---------------------------------------------------------------------------
def train_autoencoder_custom(
    model: Autoencoder,
    loader: DataLoader,
    target_class: int,
    loss_fn: str = "mae",
    epochs: int = EPOCHS,
) -> list:
    """
    Addestra con la custom loss (Eq. 3.3 o 3.4).

    Parameters
    ----------
    target_class : classe a cui questo AE è specializzato
    loss_fn      : 'mae' → Eq. 3.3 | 'mse' → Eq. 3.4
    """
    custom_fn = custom_loss_mae if loss_fn == "mae" else custom_loss_mse

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples  = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = custom_fn(x_batch, x_hat, y_batch, target_class)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples  += x_batch.size(0)

        avg_loss = total_loss / n_samples
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  Custom Loss ({loss_fn.upper()}): {avg_loss:.6f}")

    return history


# ---------------------------------------------------------------------------
# Calcolo della soglia (threshold) per novelty detection
# threshold_k = mean(err_k su campioni classe k nel train) + SIGMA_FACTOR * std
# ---------------------------------------------------------------------------
def compute_threshold(
    model: Autoencoder,
    X_class: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
    sigma_factor: float = SIGMA_FACTOR,
) -> float:
    """
    Calcola la soglia di novelty per un autoencoder dato.

    Parameters
    ----------
    X_class : np.ndarray (N_k, D) — campioni del train della classe k (già scalati)
    loss_fn : 'mae' o 'mse'
    sigma_factor : moltiplicatore della deviazione standard nella soglia

    Returns
    -------
    threshold : float
    """
    recon_fn = mae_reconstruction if loss_fn == "mae" else mse_reconstruction

    errors = _compute_reconstruction_errors_batched(
        model, X_class, recon_fn, batch_size=batch_size
    )

    mu    = errors.mean()
    sigma = errors.std()
    return float(mu + sigma_factor * sigma)


# ---------------------------------------------------------------------------
# Inferenza: per ogni campione di test calcola gli errori di ricostruzione
# su tutti gli autoencoder e applica la regola di decisione.
# ---------------------------------------------------------------------------
def predict_novelty(
    autoencoders: dict,
    thresholds: dict,
    X_test: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 8192,
) -> tuple:
    """
    Applica la regola di novelty detection sull'intero test set.

    Parameters
    ----------
    autoencoders : {class_id: Autoencoder}
    thresholds   : {class_id: float}
    X_test       : np.ndarray (N_test, D)
    loss_fn      : 'mae' o 'mse'

    Returns
    -------
    predictions  : np.ndarray (N_test,) — classe predetta o -1 (novelty)
    min_errors   : np.ndarray (N_test,) — errore minimo sul miglior AE
    """
    recon_fn = mae_reconstruction if loss_fn == "mae" else mse_reconstruction

    class_ids = list(autoencoders.keys())

    min_err = np.full(len(X_test), np.inf, dtype=np.float32)
    best_cls = np.zeros(len(X_test), dtype=np.int64)

    for cls in class_ids:
        model = autoencoders[cls]
        errors = _compute_reconstruction_errors_batched(
            model, X_test, recon_fn, batch_size=batch_size
        )
        better_mask = errors < min_err
        min_err[better_mask] = errors[better_mask]
        best_cls[better_mask] = cls

    # Regola di novelty
    predictions = np.where(
        min_err > np.array([thresholds[c] for c in best_cls]),
        -1,          # novelty
        best_cls,    # classe predetta
    )

    return predictions, min_err


# ---------------------------------------------------------------------------
# ESPERIMENTO 1 — singolo AE globale
# ---------------------------------------------------------------------------
def run_experiment_1(data: dict, loss_fn: str = "mae") -> dict:
    print(f"\n{'='*60}")
    print(f"ESPERIMENTO 1 — AE globale  |  Loss: {loss_fn.upper()}")
    print(f"{'='*60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # Filtra solo le classi note per il training
    known_mask = np.isin(y_train, KNOWN_CLASSES)
    X_known    = X_train[known_mask]
    y_known    = y_train[known_mask]

    loader = _make_loader(X_known, y_known, shuffle=True)

    cfg   = EXP1_CONFIG
    model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

    history = train_autoencoder(model, loader, loss_fn=loss_fn)

    # Threshold calcolata su tutti i campioni noti del train
    threshold = compute_threshold(model, X_known, loss_fn=loss_fn)
    print(f"  Threshold globale: {threshold:.6f}")

    # Salvataggio modello
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"exp1_{loss_fn}.pt")

    # Errori di ricostruzione sul train (serve per il plot KDE)
    recon_fn = mae_reconstruction if loss_fn == "mae" else mse_reconstruction
    X_known_t = torch.tensor(X_known, dtype=torch.float32).to(DEVICE)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        train_errors = recon_fn(X_known_t, model(X_known_t)).cpu().numpy()
        errors       = recon_fn(X_test_t,  model(X_test_t)).cpu().numpy()

    predictions = np.where(errors > threshold, -1, 0)  # 0 = "known" genericamente

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

    _save_results(results, f"exp1_{loss_fn}")
    return results


# ---------------------------------------------------------------------------
# ESPERIMENTO 2 — un AE per classe, addestrato solo sui campioni di quella classe
# ---------------------------------------------------------------------------
def run_experiment_2(data: dict, loss_fn: str = "mae") -> dict:
    print(f"\n{'='*60}")
    print(f"ESPERIMENTO 2 — AE per classe  |  Loss: {loss_fn.upper()}")
    print(f"{'='*60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    autoencoders: dict = {}
    thresholds:   dict = {}
    histories:    dict = {}

    for cls in KNOWN_CLASSES:
        print(f"\n  → Classe {cls}")

        # Campioni della sola classe cls
        mask    = (y_train == cls)
        X_cls   = X_train[mask]

        if len(X_cls) == 0:
            print(f"    ATTENZIONE: nessun campione per la classe {cls}, skip.")
            continue

        cfg   = EXP2_CONFIG[cls][loss_fn]
        model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

        loader   = _make_loader(X_cls, shuffle=True)
        history  = train_autoencoder(model, loader, loss_fn=loss_fn)
        threshold = compute_threshold(model, X_cls, loss_fn=loss_fn)

        print(f"    Threshold classe {cls}: {threshold:.6f}")

        autoencoders[cls] = model
        thresholds[cls]   = threshold
        histories[cls]    = history

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / f"exp2_{loss_fn}_class{cls}.pt")

    # Predizione sul test set
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

    _save_results(results, f"exp2_{loss_fn}")
    return results, autoencoders


# ---------------------------------------------------------------------------
# ESPERIMENTO 3 — un AE per classe, addestrato su TUTTO il train con custom loss
# ---------------------------------------------------------------------------
def run_experiment_3(data: dict, loss_fn: str = "mae") -> dict:
    print(f"\n{'='*60}")
    print(f"ESPERIMENTO 3 — Custom Loss  |  Loss: {loss_fn.upper()}")
    print(f"{'='*60}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # Filtra solo le classi note per il training (le novelty non devono entrare)
    known_mask = np.isin(y_train, KNOWN_CLASSES)
    X_known    = X_train[known_mask]
    y_known    = y_train[known_mask]

    # DataLoader con (X, y) su TUTTE le classi note
    full_loader = _make_loader(X_known, y_known, shuffle=True)

    autoencoders: dict = {}
    thresholds:   dict = {}
    histories:    dict = {}

    for cls in KNOWN_CLASSES:
        print(f"\n  → Classe {cls}")

        cfg   = EXP3_CONFIG[cls][loss_fn]
        model = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])

        history = train_autoencoder_custom(
            model, full_loader, target_class=cls, loss_fn=loss_fn
        )

        # Threshold: calcolata solo sui campioni di training della classe cls
        mask  = (y_known == cls)
        X_cls = X_known[mask]
        threshold = compute_threshold(model, X_cls, loss_fn=loss_fn)

        print(f"    Threshold classe {cls}: {threshold:.6f}")

        autoencoders[cls] = model
        thresholds[cls]   = threshold
        histories[cls]    = history

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / f"exp3_{loss_fn}_class{cls}.pt")

    # Predizione sul test set
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

    _save_results(results, f"exp3_{loss_fn}")
    return results, autoencoders


# ---------------------------------------------------------------------------
# Utility: salva risultati in JSON
# ---------------------------------------------------------------------------
def _save_results(results: dict, name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Risultati salvati in: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Caricamento dati...")
    data = prepare_data()
    print(f"  Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    # Esegui tutti e tre gli esperimenti per entrambe le loss
    for loss in ("mae", "mse"):
        run_experiment_1(data, loss_fn=loss)
        run_experiment_2(data, loss_fn=loss)
        run_experiment_3(data, loss_fn=loss)

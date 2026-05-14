"""
evaluate.py — Visualizzazione e valutazione per novelty detection.

Funzioni principali:
  build_error_matrix            — matrice errori medi (classi × AE)
  plot_error_distribution_exp1  — Exp 1: KDE train/test/novelty + soglia
  plot_error_heatmap            — Exp 2/3: heatmap errori medi
  plot_min_error_distribution   — distribuzione min_errors post-predizione
  compute_novelty_summary       — statistiche novelty detection
  print_novelty_summary         — stampa formattata del riepilogo
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import DEVICE, KNOWN_CLASSES, NOVELTY_CLASSES, RESULTS_DIR, PLOTS_DIR
from losses import mae_reconstruction, mse_reconstruction
from model import Autoencoder

# ---------------------------------------------------------------------------
# Stile globale
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
# PLOTS_DIR è importato da config: OUTPUT_DIR / "plots"


# ---------------------------------------------------------------------------
# Utility interna: errori di ricostruzione di un AE su X
# ---------------------------------------------------------------------------
def _compute_errors(
    model: Autoencoder,
    X: np.ndarray,
    loss_fn: str = "mae",
    batch_size: int = 4096,
) -> np.ndarray:
    recon_fn = mae_reconstruction if loss_fn == "mae" else mse_reconstruction
    model.eval()
    model.to(DEVICE)

    all_errors = []
    for start in range(0, len(X), batch_size):
        X_batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            errors = recon_fn(X_batch, model(X_batch)).cpu().numpy()
        all_errors.append(errors)

    return np.concatenate(all_errors)  # (N,)


# ---------------------------------------------------------------------------
# Matrice errori medi (righe=classi reali, colonne=AE)
# ---------------------------------------------------------------------------
def build_error_matrix(
    autoencoders: Dict[int, Autoencoder],
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: str = "mae",
) -> tuple:
    """
    Calcola la matrice degli errori medi di ricostruzione.

    Returns
    -------
    matrix     : np.ndarray (n_row_classes, n_ae_classes)
    row_labels : List[int] — classi reali uniche in y (righe)
    col_labels : List[int] — classi degli AE (colonne)
    """
    ae_classes  = sorted(autoencoders.keys())
    row_classes = sorted(int(c) for c in np.unique(y))

    matrix = np.full((len(row_classes), len(ae_classes)), np.nan, dtype=np.float32)

    for j, ae_cls in enumerate(ae_classes):
        errors = _compute_errors(autoencoders[ae_cls], X, loss_fn)
        for i, real_cls in enumerate(row_classes):
            mask = y == real_cls
            if mask.sum() > 0:
                matrix[i, j] = errors[mask].mean()

    return matrix, row_classes, ae_classes


# ---------------------------------------------------------------------------
# Exp 1 — distribuzione errori (KDE)
# ---------------------------------------------------------------------------
def plot_error_distribution_exp1(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    loss_fn: str = "mae",
    title_suffix: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """
    KDE degli errori di ricostruzione (Exp 1).
    Separa: train known, test known, test novelty.
    Aggiunge la linea verticale della soglia.
    Corrisponde alla figura 4.1 del PDF.
    """
    known_mask   = np.isin(y_test, KNOWN_CLASSES)
    novelty_mask = np.isin(y_test, NOVELTY_CLASSES)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(train_errors,           ax=ax, label="Train (known)",  color="steelblue",   fill=True, alpha=0.35)
    sns.kdeplot(test_errors[known_mask], ax=ax, label="Test (known)",  color="forestgreen", fill=True, alpha=0.35)

    if novelty_mask.sum() > 0:
        sns.kdeplot(test_errors[novelty_mask], ax=ax, label="Test (novelty)", color="crimson", fill=True, alpha=0.35)

    ax.axvline(
        threshold, color="black", linestyle="--", linewidth=1.5,
        label=f"Threshold = {threshold:.4f}",
    )

    ax.set_xlabel(f"Errore di ricostruzione ({loss_fn.upper()})")
    ax.set_ylabel("Densità")
    ax.set_title(f"Exp 1 — Distribuzione errori {loss_fn.upper()}{title_suffix}")
    ax.legend()
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Exp 2/3 — heatmap errori medi
# ---------------------------------------------------------------------------
def plot_error_heatmap(
    matrix: np.ndarray,
    row_labels: List[int],
    col_labels: List[int],
    title: str = "Heatmap errori di ricostruzione",
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap con righe=classi reali e colonne=AE.
    Un valore basso sulla diagonale indica specializzazione.
    Corrisponde alle figure 4.2–4.9 del PDF.
    """
    df = pd.DataFrame(
        matrix,
        index=[f"C{c}" for c in row_labels],
        columns=[f"AE{c}" for c in col_labels],
    )

    h = max(6, len(row_labels) * 0.55)
    w = max(10, len(col_labels) * 0.8)

    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        df, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, cbar_kws={"label": "Errore medio"},
    )
    ax.set_title(title)
    ax.set_xlabel("Autoencoder")
    ax.set_ylabel("Classe reale")
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Distribuzione min_errors (output di predict_novelty)
# ---------------------------------------------------------------------------
def plot_min_error_distribution(
    min_errors: np.ndarray,
    y_test: np.ndarray,
    thresholds: Dict[int, float],
    loss_fn: str = "mae",
    title_suffix: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """
    KDE degli errori minimi (min su tutti gli AE).
    Separa campioni noti e novelty reali.
    Aggiunge la soglia mediana come riferimento visivo.
    """
    known_mask   = np.isin(y_test, KNOWN_CLASSES)
    novelty_mask = np.isin(y_test, NOVELTY_CLASSES)

    fig, ax = plt.subplots(figsize=(10, 5))

    if known_mask.sum() > 0:
        sns.kdeplot(min_errors[known_mask],   ax=ax, label="Known (reale)",   color="steelblue", fill=True, alpha=0.35)
    if novelty_mask.sum() > 0:
        sns.kdeplot(min_errors[novelty_mask], ax=ax, label="Novelty (reale)", color="crimson",   fill=True, alpha=0.35)

    median_thr = float(np.median(list(thresholds.values())))
    ax.axvline(
        median_thr, color="black", linestyle="--", linewidth=1.5,
        label=f"Threshold mediana AE (solo rif. visivo) = {median_thr:.4f}",
    )

    ax.set_xlabel(f"Min errore di ricostruzione ({loss_fn.upper()})")
    ax.set_ylabel("Densità")
    ax.set_title(f"Distribuzione min_errors — {loss_fn.upper()}{title_suffix}")
    ax.legend()
    plt.tight_layout()

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Riepilogo novelty detection
# ---------------------------------------------------------------------------
def compute_novelty_summary(
    predictions: np.ndarray,
    y_test: np.ndarray,
    min_errors: np.ndarray,
    thresholds: Dict[int, float],
) -> dict:
    """
    Calcola le statistiche di novelty detection.

    Returns
    -------
    dict con:
      n_total, n_novelty_predicted, novelty_rate_pct,
      novelty_recall_pct, false_novelty_rate_pct,
      per_class_novelty_recall_pct, thresholds
    """
    summary: dict = {}

    n_total   = len(predictions)
    n_novelty = int((predictions == -1).sum())

    summary["n_total"]             = n_total
    summary["n_novelty_predicted"] = n_novelty
    summary["novelty_rate_pct"]    = round(100 * n_novelty / n_total, 2)

    # Recall delle novelty reali
    real_novelty = np.isin(y_test, NOVELTY_CLASSES)
    if real_novelty.sum() > 0:
        tp = int(((predictions == -1) & real_novelty).sum())
        summary["novelty_recall_pct"] = round(100 * tp / real_novelty.sum(), 2)
    else:
        summary["novelty_recall_pct"] = None

    # Tasso di falsi novelty (classi note marcate come novelty)
    real_known = np.isin(y_test, KNOWN_CLASSES)
    if real_known.sum() > 0:
        fp = int(((predictions == -1) & real_known).sum())
        summary["false_novelty_rate_pct"] = round(100 * fp / real_known.sum(), 2)
    else:
        summary["false_novelty_rate_pct"] = None

    # Recall per singola classe novelty
    per_class: dict = {}
    for cls in NOVELTY_CLASSES:
        mask = y_test == cls
        if mask.sum() > 0:
            recall = ((predictions == -1) & mask).sum() / mask.sum()
            per_class[str(cls)] = round(100 * float(recall), 2)
    summary["per_class_novelty_recall_pct"] = per_class

    summary["thresholds"] = {str(k): round(float(v), 6) for k, v in thresholds.items()}

    # Accuracy sulle sole classi note.
    # Exp 1 usa 0 come etichetta generica "known", quindi non produce una
    # classificazione tra le classi note e l'accuracy sarebbe fuorviante.
    known_mask_real = np.isin(y_test, KNOWN_CLASSES)
    known_and_classified = known_mask_real & (predictions != -1)
    predicts_known_classes = np.all(
        np.isin(predictions[known_and_classified], KNOWN_CLASSES)
    )
    if known_and_classified.sum() > 0 and predicts_known_classes:
        correct = (predictions[known_and_classified] == y_test[known_and_classified]).sum()
        summary["known_accuracy_pct"] = round(100 * int(correct) / int(known_and_classified.sum()), 2)
    else:
        summary["known_accuracy_pct"] = None

    return summary


def print_novelty_summary(summary: dict, label: str = "") -> None:
    """Stampa formattata del riepilogo novelty detection."""
    header = f"  NOVELTY SUMMARY — {label}" if label else "  NOVELTY SUMMARY"
    print(f"\n{'='*55}")
    print(header)
    print(f"{'='*55}")
    print(f"  Campioni totali          : {summary['n_total']}")
    print(f"  Predetti come novelty    : {summary['n_novelty_predicted']} ({summary['novelty_rate_pct']}%)")

    recall = summary.get("novelty_recall_pct")
    fnr    = summary.get("false_novelty_rate_pct")
    print(f"  Novelty recall           : {recall}%" if recall is not None else "  Novelty recall           : N/A")
    print(f"  False novelty rate       : {fnr}%"    if fnr    is not None else "  False novelty rate       : N/A")

    per_cls = summary.get("per_class_novelty_recall_pct", {})
    if per_cls:
        print("  Recall per classe novelty:")
        for cls, val in per_cls.items():
            print(f"    Classe {cls}: {val}%")

    acc = summary.get("known_accuracy_pct")
    print(f"  Known class accuracy     : {acc}%" if acc is not None else "  Known class accuracy     : N/A")

    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Utility interna: salva e mostra il plot
# ---------------------------------------------------------------------------
def _save_and_show(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Salvato: {save_path}")
    plt.close(fig)

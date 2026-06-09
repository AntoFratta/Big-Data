"""
Evaluation and visualization utilities for PLAsTiCC novelty detection.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import KNOWN_CLASSES, NOVELTY_CLASSES
from model import Autoencoder
from utils import compute_reconstruction_errors


sns.set_theme(style="whitegrid", context="notebook")


def build_error_matrix(
    autoencoders: Dict[int, Autoencoder],
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: str = "mae",
) -> tuple:
    """
    Build the mean reconstruction-error matrix.

    Rows represent true classes and columns represent the class-specific
    autoencoders.
    """
    ae_classes = sorted(autoencoders.keys())
    row_classes = sorted(int(c) for c in np.unique(y))

    matrix = np.full((len(row_classes), len(ae_classes)), np.nan, dtype=np.float32)

    for j, ae_cls in enumerate(ae_classes):
        errors = compute_reconstruction_errors(
            autoencoders[ae_cls],
            X,
            loss_fn=loss_fn,
        )
        for i, real_cls in enumerate(row_classes):
            mask = y == real_cls
            if mask.sum() > 0:
                matrix[i, j] = errors[mask].mean()

    return matrix, row_classes, ae_classes


def plot_error_distribution_exp1(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    loss_fn: str = "mae",
    title_suffix: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """Plot the reconstruction-error distributions for Experiment 1."""
    known_mask = np.isin(y_test, KNOWN_CLASSES)
    novelty_mask = np.isin(y_test, NOVELTY_CLASSES)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(train_errors, ax=ax, label="Train (known)", color="steelblue", fill=True, alpha=0.35)
    sns.kdeplot(test_errors[known_mask], ax=ax, label="Test (known)", color="forestgreen", fill=True, alpha=0.35)

    if novelty_mask.sum() > 0:
        sns.kdeplot(test_errors[novelty_mask], ax=ax, label="Test (novelty)", color="crimson", fill=True, alpha=0.35)

    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold:.4f}",
    )

    ax.set_xlabel(f"Reconstruction error ({loss_fn.upper()})")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Experiment 1 - Reconstruction-error distribution "
        f"({loss_fn.upper()}){title_suffix}"
    )
    ax.legend()
    plt.tight_layout()

    _save_and_close(fig, save_path)


def plot_error_heatmap(
    matrix: np.ndarray,
    row_labels: List[int],
    col_labels: List[int],
    title: str = "Reconstruction-error heatmap",
    save_path: Optional[Path] = None,
) -> None:
    """Plot the mean reconstruction-error matrix as a heatmap."""

    def _compact_value(value: float) -> str:
        value = float(value)
        abs_value = abs(value)
        if np.isnan(value):
            return ""
        if abs_value >= 10000:
            return f"{value / 1000:.0f}k"
        if abs_value >= 1000:
            return f"{value / 1000:.1f}k"
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    df = pd.DataFrame(
        matrix,
        index=[f"C{c}" for c in row_labels],
        columns=[f"AE{c}" for c in col_labels],
    )
    labels = df.map(_compact_value)

    h = max(6, len(row_labels) * 0.55)
    w = max(12, len(col_labels) * 0.9)

    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        df,
        ax=ax,
        annot=labels,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"label": "Mean error"},
        annot_kws={"fontsize": 8},
    )
    ax.set_title(title)
    ax.set_xlabel("Autoencoder")
    ax.set_ylabel("True class")
    plt.tight_layout()

    _save_and_close(fig, save_path)


def plot_min_error_distribution(
    min_errors: np.ndarray,
    y_test: np.ndarray,
    thresholds: Dict[int, float],
    loss_fn: str = "mae",
    title_suffix: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the distribution of the minimum reconstruction error across AEs.

    The vertical threshold is the median per-class threshold and is shown only
    as a visual reference, because the decision rule uses class-specific values.
    """
    known_mask = np.isin(y_test, KNOWN_CLASSES)
    novelty_mask = np.isin(y_test, NOVELTY_CLASSES)

    fig, ax = plt.subplots(figsize=(10, 5))

    if known_mask.sum() > 0:
        sns.kdeplot(min_errors[known_mask], ax=ax, label="Known (true)", color="steelblue", fill=True, alpha=0.35)
    if novelty_mask.sum() > 0:
        sns.kdeplot(min_errors[novelty_mask], ax=ax, label="Novelty (true)", color="crimson", fill=True, alpha=0.35)

    median_threshold = float(np.median(list(thresholds.values())))
    ax.axvline(
        median_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Median AE threshold (visual reference) = {median_threshold:.4f}",
    )

    ax.set_xlabel(f"Minimum reconstruction error ({loss_fn.upper()})")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Minimum reconstruction-error distribution "
        f"({loss_fn.upper()}){title_suffix}"
    )
    ax.legend()
    plt.tight_layout()

    _save_and_close(fig, save_path)


def compute_novelty_summary(
    predictions: np.ndarray,
    y_test: np.ndarray,
    min_errors: np.ndarray,
    thresholds: Dict[int, float],
) -> dict:
    """Compute the main novelty-detection metrics."""
    summary = {}

    n_total = len(predictions)
    n_novelty = int((predictions == -1).sum())

    summary["n_total"] = n_total
    summary["n_novelty_predicted"] = n_novelty
    summary["novelty_rate_pct"] = round(100 * n_novelty / n_total, 2)

    real_novelty = np.isin(y_test, NOVELTY_CLASSES)
    if real_novelty.sum() > 0:
        tp = int(((predictions == -1) & real_novelty).sum())
        summary["novelty_recall_pct"] = round(100 * tp / real_novelty.sum(), 2)
    else:
        summary["novelty_recall_pct"] = None

    real_known = np.isin(y_test, KNOWN_CLASSES)
    if real_known.sum() > 0:
        fp = int(((predictions == -1) & real_known).sum())
        summary["false_novelty_rate_pct"] = round(100 * fp / real_known.sum(), 2)
    else:
        summary["false_novelty_rate_pct"] = None

    per_class = {}
    for cls in NOVELTY_CLASSES:
        mask = y_test == cls
        if mask.sum() > 0:
            recall = ((predictions == -1) & mask).sum() / mask.sum()
            per_class[str(cls)] = round(100 * float(recall), 2)
    summary["per_class_novelty_recall_pct"] = per_class

    summary["thresholds"] = {str(k): round(float(v), 6) for k, v in thresholds.items()}

    known_mask_real = np.isin(y_test, KNOWN_CLASSES)
    known_and_classified = known_mask_real & (predictions != -1)
    predicts_known_classes = np.all(
        np.isin(predictions[known_and_classified], KNOWN_CLASSES)
    )
    if known_and_classified.sum() > 0 and predicts_known_classes:
        correct = (predictions[known_and_classified] == y_test[known_and_classified]).sum()
        summary["known_accuracy_pct"] = round(
            100 * int(correct) / int(known_and_classified.sum()),
            2,
        )
    else:
        summary["known_accuracy_pct"] = None

    return summary


def print_novelty_summary(summary: dict, label: str = "") -> None:
    """Print the novelty-detection metrics in a compact console format."""
    header = f"  NOVELTY SUMMARY - {label}" if label else "  NOVELTY SUMMARY"
    print(f"\n{'=' * 55}")
    print(header)
    print(f"{'=' * 55}")
    print(f"  Total samples             : {summary['n_total']}")
    print(
        "  Predicted as novelty      : "
        f"{summary['n_novelty_predicted']} ({summary['novelty_rate_pct']}%)"
    )

    recall = summary.get("novelty_recall_pct")
    fnr = summary.get("false_novelty_rate_pct")
    print(f"  Novelty recall            : {recall}%" if recall is not None else "  Novelty recall            : N/A")
    print(f"  False novelty rate        : {fnr}%" if fnr is not None else "  False novelty rate        : N/A")

    per_cls = summary.get("per_class_novelty_recall_pct", {})
    if per_cls:
        print("  Per-class novelty recall:")
        for cls, val in per_cls.items():
            print(f"    Class {cls}: {val}%")

    acc = summary.get("known_accuracy_pct")
    print(f"  Known class accuracy      : {acc}%" if acc is not None else "  Known class accuracy      : N/A")

    print(f"{'=' * 55}\n")


def _save_and_close(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)

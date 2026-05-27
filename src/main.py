"""
main.py — Pipeline completa per novelty detection su PLAsTiCC.

Esegue in sequenza:
  1. Caricamento dati
  2. Esperimento 1 (AE globale)      — MAE e MSE
  3. Esperimento 2 (AE per classe)   — MAE e MSE
  4. Esperimento 3 (Custom loss)     — MAE e MSE
  5. Grafici e riepilogo per ogni esperimento
"""

import json
import numpy as np
import random
import torch

from config import KNOWN_CLASSES, PLOTS_DIR, RANDOM_SEED, RESULTS_DIR
from data_loader import prepare_data
from train import run_experiment_1, run_experiment_2, run_experiment_3
import threshold_sensitivity
from evaluate import (
    build_error_matrix,
    plot_error_distribution_exp1,
    plot_error_heatmap,
    plot_min_error_distribution,
    compute_novelty_summary,
    print_novelty_summary,
)


# ---------------------------------------------------------------------------
# Helper: filtra i campioni noti dal train per le heatmap
# ---------------------------------------------------------------------------
def _known_train(data: dict):
    mask = np.isin(data["y_train"], KNOWN_CLASSES)
    return data["X_train"][mask], data["y_train"][mask]


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _final_loss_stats(histories: dict) -> dict:
    final_losses = {str(cls): float(history[-1]) for cls, history in histories.items()}
    values = np.array(list(final_losses.values()), dtype=float)
    return {
        "final_loss_per_class": final_losses,
        "final_loss_mean": float(values.mean()),
        "final_loss_std": float(values.std()),
        "final_loss_min": float(values.min()),
        "final_loss_max": float(values.max()),
    }


def _save_summary(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_summary.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Summary salvato in: {path}")


def _training_metadata_from_result(res: dict) -> dict:
    if "history" in res:
        return {
            "history": res["history"],
            "final_loss": float(res["history"][-1]),
        }

    if "histories" in res:
        return {
            "histories": res["histories"],
            **_final_loss_stats(res["histories"]),
        }

    return {}


# ---------------------------------------------------------------------------
# ESPERIMENTO 1
# ---------------------------------------------------------------------------
def pipeline_exp1(data: dict, loss_fn: str) -> None:
    print(f"\n{'#'*60}")
    print(f"# PIPELINE EXP 1 — {loss_fn.upper()}")
    print(f"{'#'*60}")

    res = run_experiment_1(data, loss_fn=loss_fn)

    train_errors = np.array(res["train_errors"])
    test_errors  = np.array(res["errors"])
    predictions  = np.array(res["predictions"])
    y_test       = np.array(res["y_test"])
    threshold    = res["threshold"]

    # --- Plot distribuzione errori ---
    plot_error_distribution_exp1(
        train_errors=train_errors,
        test_errors=test_errors,
        y_test=y_test,
        threshold=threshold,
        loss_fn=loss_fn,
        title_suffix=f" — Exp 1",
        save_path=PLOTS_DIR / f"exp1_{loss_fn}_distribution.png",
    )

    # --- Riepilogo novelty ---
    summary = compute_novelty_summary(
        predictions=predictions,
        y_test=y_test,
        min_errors=test_errors,
        thresholds={"global": threshold},
    )
    print_novelty_summary(summary, label=f"Exp 1 — {loss_fn.upper()}")

    _save_summary(
        f"exp1_{loss_fn}",
        {
            "experiment": 1,
            "loss_fn": loss_fn,
            "history": res["history"],
            "final_loss": float(res["history"][-1]),
            "threshold": float(threshold),
            "summary_metrics": summary,
        },
    )


# ---------------------------------------------------------------------------
# ESPERIMENTO 2
# ---------------------------------------------------------------------------
def pipeline_exp2(data: dict, loss_fn: str) -> None:
    print(f"\n{'#'*60}")
    print(f"# PIPELINE EXP 2 — {loss_fn.upper()}")
    print(f"{'#'*60}")

    res, autoencoders = run_experiment_2(data, loss_fn=loss_fn)

    predictions = np.array(res["predictions"])
    min_errors  = np.array(res["min_errors"])
    y_test      = np.array(res["y_test"])
    thresholds  = {int(k): v for k, v in res["thresholds"].items()}

    X_known, y_known = _known_train(data)

    # --- Heatmap sul training set (solo classi note) ---
    mat_train, rows_train, cols = build_error_matrix(
        autoencoders, X_known, y_known, loss_fn=loss_fn
    )
    plot_error_heatmap(
        mat_train, rows_train, cols,
        title=f"Exp 2 — Errori medi TRAIN ({loss_fn.upper()})",
        save_path=PLOTS_DIR / f"exp2_{loss_fn}_train_heatmap.png",
    )

    # --- Heatmap sul test set (classi note + novelty) ---
    mat_test, rows_test, _ = build_error_matrix(
        autoencoders, data["X_test"], y_test, loss_fn=loss_fn
    )
    plot_error_heatmap(
        mat_test, rows_test, cols,
        title=f"Exp 2 — Errori medi TEST ({loss_fn.upper()})",
        save_path=PLOTS_DIR / f"exp2_{loss_fn}_test_heatmap.png",
    )

    # --- Distribuzione min_errors ---
    plot_min_error_distribution(
        min_errors=min_errors,
        y_test=y_test,
        thresholds=thresholds,
        loss_fn=loss_fn,
        title_suffix=" — Exp 2",
        save_path=PLOTS_DIR / f"exp2_{loss_fn}_min_errors.png",
    )

    # --- Riepilogo novelty ---
    summary = compute_novelty_summary(
        predictions=predictions,
        y_test=y_test,
        min_errors=min_errors,
        thresholds=thresholds,
    )
    print_novelty_summary(summary, label=f"Exp 2 — {loss_fn.upper()}")

    _save_summary(
        f"exp2_{loss_fn}",
        {
            "experiment": 2,
            "loss_fn": loss_fn,
            "histories": res["histories"],
            **_final_loss_stats(res["histories"]),
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "summary_metrics": summary,
        },
    )


# ---------------------------------------------------------------------------
# ESPERIMENTO 3
# ---------------------------------------------------------------------------
def pipeline_exp3(data: dict, loss_fn: str) -> None:
    print(f"\n{'#'*60}")
    print(f"# PIPELINE EXP 3 — Custom Loss {loss_fn.upper()}")
    print(f"{'#'*60}")

    res, autoencoders = run_experiment_3(data, loss_fn=loss_fn)

    predictions = np.array(res["predictions"])
    min_errors  = np.array(res["min_errors"])
    y_test      = np.array(res["y_test"])
    thresholds  = {int(k): v for k, v in res["thresholds"].items()}

    X_known, y_known = _known_train(data)

    # --- Heatmap sul training set ---
    mat_train, rows_train, cols = build_error_matrix(
        autoencoders, X_known, y_known, loss_fn=loss_fn
    )
    plot_error_heatmap(
        mat_train, rows_train, cols,
        title=f"Exp 3 — Errori medi TRAIN custom loss ({loss_fn.upper()})",
        save_path=PLOTS_DIR / f"exp3_{loss_fn}_train_heatmap.png",
    )

    # --- Heatmap sul test set ---
    mat_test, rows_test, _ = build_error_matrix(
        autoencoders, data["X_test"], y_test, loss_fn=loss_fn
    )
    plot_error_heatmap(
        mat_test, rows_test, cols,
        title=f"Exp 3 — Errori medi TEST custom loss ({loss_fn.upper()})",
        save_path=PLOTS_DIR / f"exp3_{loss_fn}_test_heatmap.png",
    )

    # --- Distribuzione min_errors ---
    plot_min_error_distribution(
        min_errors=min_errors,
        y_test=y_test,
        thresholds=thresholds,
        loss_fn=loss_fn,
        title_suffix=" — Exp 3",
        save_path=PLOTS_DIR / f"exp3_{loss_fn}_min_errors.png",
    )

    # --- Riepilogo novelty ---
    summary = compute_novelty_summary(
        predictions=predictions,
        y_test=y_test,
        min_errors=min_errors,
        thresholds=thresholds,
    )
    print_novelty_summary(summary, label=f"Exp 3 — {loss_fn.upper()}")

    _save_summary(
        f"exp3_{loss_fn}",
        {
            "experiment": 3,
            "loss_fn": loss_fn,
            "histories": res["histories"],
            **_final_loss_stats(res["histories"]),
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "summary_metrics": summary,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    set_random_seed()

    print("=" * 60)
    print("  PLAsTiCC — Novelty Detection Pipeline")
    print("=" * 60)

    print("\nCaricamento dati...")
    data = prepare_data()
    print(f"  Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    training_metadata = {}

    for loss_fn in ("mae", "mse"):
        res1 = run_experiment_1(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp1_{loss_fn}"] = _training_metadata_from_result(res1)

        res2, _ = run_experiment_2(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp2_{loss_fn}"] = _training_metadata_from_result(res2)

        res3, _ = run_experiment_3(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp3_{loss_fn}"] = _training_metadata_from_result(res3)

    print("\nTraining completato.")
    print("Avvio valutazione con soglie media + 3σ, + 2σ, + 1σ...")

    threshold_sensitivity.TRAINING_METADATA = training_metadata
    threshold_sensitivity.main(data=data)

    print("\nPipeline completata.")
    print(f"  Modelli  → outputs/models/")
    print(f"  Risultati → outputs/results/sigma_*/")
    print(f"  Grafici  → outputs/plots/sigma_*/")


if __name__ == "__main__":
    main()

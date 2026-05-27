"""
Evaluate saved autoencoders with multiple threshold factors.

This script does not retrain models. It loads the models saved by main.py and
recomputes predictions/plots for thresholds of the form:

    threshold = mean(train_errors) + sigma_factor * std(train_errors)

Outputs are separated by sigma factor under outputs/results/sigma_* and
outputs/plots/sigma_*.
"""

import json
from pathlib import Path
import random

import numpy as np
import torch

from config import (
    DEVICE,
    EXP1_CONFIG,
    EXP2_CONFIG,
    EXP3_CONFIG,
    KNOWN_CLASSES,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from data_loader import prepare_data
from evaluate import (
    build_error_matrix,
    compute_novelty_summary,
    plot_error_distribution_exp1,
    plot_error_heatmap,
    plot_min_error_distribution,
    print_novelty_summary,
)
from losses import mae_reconstruction, mse_reconstruction
from model import Autoencoder
from train import _compute_reconstruction_errors_batched


SIGMA_FACTORS = (3, 2, 1)
BATCH_EVAL_SIZE = 8192

# Save complete per-sample outputs as requested for the threshold comparison.
SAVE_DETAILED_RESULTS = True
TRAINING_METADATA = {}


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _known_train(data: dict) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(data["y_train"], KNOWN_CLASSES)
    return data["X_train"][mask], data["y_train"][mask]


def _recon_fn(loss_fn: str):
    return mae_reconstruction if loss_fn == "mae" else mse_reconstruction


def _load_model(config: dict, path: Path) -> Autoencoder:
    model = Autoencoder(
        config["layer_dims"],
        config["activation"],
        config["use_dropout"],
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def _load_class_autoencoders(experiment: int, loss_fn: str) -> dict[int, Autoencoder]:
    config = EXP2_CONFIG if experiment == 2 else EXP3_CONFIG
    autoencoders = {}

    for cls in KNOWN_CLASSES:
        path = MODELS_DIR / f"exp{experiment}_{loss_fn}_class{cls}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved model: {path}")
        autoencoders[cls] = _load_model(config[cls][loss_fn], path)

    return autoencoders


def _threshold_stats_from_errors(errors: np.ndarray) -> dict:
    return {
        "mean": float(errors.mean()),
        "std": float(errors.std()),
    }


def _thresholds_from_stats(stats: dict, sigma_factor: float) -> dict:
    return {
        cls: values["mean"] + sigma_factor * values["std"]
        for cls, values in stats.items()
    }


def _threshold_stats_by_class(
    autoencoders: dict[int, Autoencoder],
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss_fn: str,
) -> dict:
    recon_fn = _recon_fn(loss_fn)
    stats = {}

    for cls, model in autoencoders.items():
        X_cls = X_train[y_train == cls]
        errors = _compute_reconstruction_errors_batched(
            model,
            X_cls,
            recon_fn,
            batch_size=BATCH_EVAL_SIZE,
        )
        stats[cls] = _threshold_stats_from_errors(errors)

    return stats


def _best_errors_and_classes(
    autoencoders: dict[int, Autoencoder],
    X_test: np.ndarray,
    loss_fn: str,
) -> tuple[np.ndarray, np.ndarray]:
    recon_fn = _recon_fn(loss_fn)
    min_errors = np.full(len(X_test), np.inf, dtype=np.float32)
    best_classes = np.zeros(len(X_test), dtype=np.int64)

    for cls, model in autoencoders.items():
        errors = _compute_reconstruction_errors_batched(
            model,
            X_test,
            recon_fn,
            batch_size=BATCH_EVAL_SIZE,
        )
        better_mask = errors < min_errors
        min_errors[better_mask] = errors[better_mask]
        best_classes[better_mask] = cls

    return min_errors, best_classes


def _predict_from_thresholds(
    min_errors: np.ndarray,
    best_classes: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    selected_thresholds = np.array([thresholds[int(cls)] for cls in best_classes])
    return np.where(min_errors > selected_thresholds, -1, best_classes)


def _output_dirs(sigma_factor: float) -> tuple[Path, Path]:
    suffix = f"sigma_{int(sigma_factor)}"
    results_dir = RESULTS_DIR / suffix
    plots_dir = PLOTS_DIR / suffix
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")


def _save_outputs(
    name: str,
    sigma_factor: float,
    result_payload: dict,
) -> None:
    results_dir, _ = _output_dirs(sigma_factor)
    _save_json(results_dir / f"{name}_results.json", result_payload)


def _load_training_metadata(experiment: int, loss_fn: str) -> dict:
    """Load lightweight training metadata from the previous main.py run."""
    key = f"exp{experiment}_{loss_fn}"
    if key in TRAINING_METADATA:
        return TRAINING_METADATA[key]

    path = RESULTS_DIR / f"exp{experiment}_{loss_fn}_summary.json"
    if not path.exists():
        return {}

    with open(path, "r") as f:
        summary = json.load(f)

    metadata = {}
    for key in (
        "history",
        "histories",
        "final_loss",
        "final_loss_per_class",
        "final_loss_mean",
        "final_loss_std",
        "final_loss_min",
        "final_loss_max",
    ):
        if key in summary:
            metadata[key] = summary[key]

    return metadata


def _detailed_arrays(**arrays) -> dict:
    if not SAVE_DETAILED_RESULTS:
        return {}
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in arrays.items()
    }


def evaluate_exp1(data: dict, loss_fn: str) -> None:
    print(f"\nEvaluating Exp 1 - {loss_fn.upper()}")

    path = MODELS_DIR / f"exp1_{loss_fn}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved model: {path}")

    training_metadata = _load_training_metadata(1, loss_fn)
    model = _load_model(EXP1_CONFIG, path)
    recon_fn = _recon_fn(loss_fn)

    X_known, _ = _known_train(data)
    train_errors = _compute_reconstruction_errors_batched(
        model,
        X_known,
        recon_fn,
        batch_size=BATCH_EVAL_SIZE,
    )
    test_errors = _compute_reconstruction_errors_batched(
        model,
        data["X_test"],
        recon_fn,
        batch_size=BATCH_EVAL_SIZE,
    )

    stats = {"global": _threshold_stats_from_errors(train_errors)}

    for sigma_factor in SIGMA_FACTORS:
        _, plots_dir = _output_dirs(sigma_factor)
        threshold = stats["global"]["mean"] + sigma_factor * stats["global"]["std"]
        predictions = np.where(test_errors > threshold, -1, 0)

        thresholds = {"global": threshold}
        summary = compute_novelty_summary(
            predictions=predictions,
            y_test=data["y_test"],
            min_errors=test_errors,
            thresholds=thresholds,
        )
        print_novelty_summary(
            summary,
            label=f"Exp 1 - {loss_fn.upper()} - sigma {sigma_factor}",
        )

        plot_error_distribution_exp1(
            train_errors=train_errors,
            test_errors=test_errors,
            y_test=data["y_test"],
            threshold=threshold,
            loss_fn=loss_fn,
            title_suffix=f" - Exp 1 - sigma {sigma_factor}",
            save_path=plots_dir / f"exp1_{loss_fn}_distribution.png",
        )

        result_payload = {
            "experiment": 1,
            "loss_fn": loss_fn,
            "sigma_factor": sigma_factor,
            "threshold": float(threshold),
            "summary_metrics": summary,
            **training_metadata,
            **_detailed_arrays(
                train_errors=train_errors,
                errors=test_errors,
                predictions=predictions,
                y_test=data["y_test"],
            ),
        }
        _save_outputs(f"exp1_{loss_fn}", sigma_factor, result_payload)


def evaluate_class_experiment(data: dict, experiment: int, loss_fn: str) -> None:
    print(f"\nEvaluating Exp {experiment} - {loss_fn.upper()}")

    training_metadata = _load_training_metadata(experiment, loss_fn)
    autoencoders = _load_class_autoencoders(experiment, loss_fn)
    X_known, y_known = _known_train(data)

    threshold_stats = _threshold_stats_by_class(
        autoencoders,
        X_known,
        y_known,
        loss_fn,
    )
    min_errors, best_classes = _best_errors_and_classes(
        autoencoders,
        data["X_test"],
        loss_fn,
    )

    train_matrix, train_rows, cols = build_error_matrix(
        autoencoders,
        X_known,
        y_known,
        loss_fn=loss_fn,
    )
    test_matrix, test_rows, _ = build_error_matrix(
        autoencoders,
        data["X_test"],
        data["y_test"],
        loss_fn=loss_fn,
    )

    for sigma_factor in SIGMA_FACTORS:
        _, plots_dir = _output_dirs(sigma_factor)
        thresholds = _thresholds_from_stats(threshold_stats, sigma_factor)
        predictions = _predict_from_thresholds(min_errors, best_classes, thresholds)

        summary = compute_novelty_summary(
            predictions=predictions,
            y_test=data["y_test"],
            min_errors=min_errors,
            thresholds=thresholds,
        )
        print_novelty_summary(
            summary,
            label=f"Exp {experiment} - {loss_fn.upper()} - sigma {sigma_factor}",
        )

        plot_error_heatmap(
            train_matrix,
            train_rows,
            cols,
            title=f"Exp {experiment} - Train errors ({loss_fn.upper()}) - sigma {sigma_factor}",
            save_path=plots_dir / f"exp{experiment}_{loss_fn}_train_heatmap.png",
        )
        plot_error_heatmap(
            test_matrix,
            test_rows,
            cols,
            title=f"Exp {experiment} - Test errors ({loss_fn.upper()}) - sigma {sigma_factor}",
            save_path=plots_dir / f"exp{experiment}_{loss_fn}_test_heatmap.png",
        )
        plot_min_error_distribution(
            min_errors=min_errors,
            y_test=data["y_test"],
            thresholds=thresholds,
            loss_fn=loss_fn,
            title_suffix=f" - Exp {experiment} - sigma {sigma_factor}",
            save_path=plots_dir / f"exp{experiment}_{loss_fn}_min_errors.png",
        )

        result_payload = {
            "experiment": experiment,
            "loss_fn": loss_fn,
            "sigma_factor": sigma_factor,
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "summary_metrics": summary,
            **training_metadata,
            **_detailed_arrays(
                predictions=predictions,
                min_errors=min_errors,
                best_classes=best_classes,
                y_test=data["y_test"],
            ),
        }
        _save_outputs(
            f"exp{experiment}_{loss_fn}",
            sigma_factor,
            result_payload,
        )


def main(data: dict | None = None) -> None:
    set_random_seed()

    print("=" * 60)
    print("  Threshold sensitivity evaluation")
    print("=" * 60)
    print(f"  Sigma factors: {SIGMA_FACTORS}")
    print("  Saved models are loaded from outputs/models/")

    if data is None:
        print("\nLoading data...")
        data = prepare_data()
    else:
        print("\nUsing data already loaded by the training pipeline...")
    print(f"  Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    for loss_fn in ("mae", "mse"):
        evaluate_exp1(data, loss_fn)
        evaluate_class_experiment(data, experiment=2, loss_fn=loss_fn)
        evaluate_class_experiment(data, experiment=3, loss_fn=loss_fn)

    print("\nEvaluation completed.")
    print("  Results -> outputs/results/sigma_*/")
    print("  Plots   -> outputs/plots/sigma_*/")


if __name__ == "__main__":
    main()

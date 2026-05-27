"""
Full PLAsTiCC novelty-detection pipeline.

The script trains all autoencoders for MAE and MSE, then evaluates the saved
models with the configured threshold factors through threshold_sensitivity.py.
"""

import random

import numpy as np
import torch

import threshold_sensitivity
from config import RANDOM_SEED
from data_loader import prepare_data
from train import run_experiment_1, run_experiment_2, run_experiment_3


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


def _training_metadata_from_result(result: dict) -> dict:
    if "history" in result:
        return {
            "history": result["history"],
            "final_loss": float(result["history"][-1]),
        }

    if "histories" in result:
        return {
            "histories": result["histories"],
            **_final_loss_stats(result["histories"]),
        }

    return {}


def main() -> None:
    set_random_seed()

    print("=" * 60)
    print("  PLAsTiCC - Novelty Detection Pipeline")
    print("=" * 60)

    print("\nLoading data...")
    data = prepare_data()
    print(f"  Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    training_metadata = {}

    for loss_fn in ("mae", "mse"):
        result_exp1 = run_experiment_1(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp1_{loss_fn}"] = _training_metadata_from_result(
            result_exp1
        )

        result_exp2, _ = run_experiment_2(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp2_{loss_fn}"] = _training_metadata_from_result(
            result_exp2
        )

        result_exp3, _ = run_experiment_3(
            data,
            loss_fn=loss_fn,
            save_results=False,
            evaluate=False,
        )
        training_metadata[f"exp3_{loss_fn}"] = _training_metadata_from_result(
            result_exp3
        )

    print("\nTraining completed.")
    print("Starting evaluation with mean + 3 sigma, + 2 sigma, and + 1 sigma thresholds...")

    threshold_sensitivity.TRAINING_METADATA = training_metadata
    threshold_sensitivity.main(data=data)

    print("\nPipeline completed.")
    print("  Models  -> outputs/models/")
    print("  Results -> outputs/results/sigma_*/")
    print("  Plots   -> outputs/plots/sigma_*/")


if __name__ == "__main__":
    main()

from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_FEATURES_PATH = TRAIN_DIR / "dataset_augment_zeros.csv"
TRAIN_LABELS_PATH = TRAIN_DIR / "y_dataset_augment.csv"

TEST_FEATURES_PATH = TEST_DIR / "dataset_test_zeros.csv"
TEST_LABELS_PATH = TEST_DIR / "y_dataset_test.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

ID_COLUMN = "object_id"
TARGET_COLUMN = "class"

FEATURE_COLUMNS = [
    "host_photoz", "host_photoz_error", "length_scale", "max_mag",
    "pos_flux_ratio", "max_flux_ratio_red", "max_flux_ratio_blue",
    "min_flux_ratio_red", "min_flux_ratio_blue", "max_dt",
    "positive_width", "negative_width",
    "time_fwd_max_0.5", "time_fwd_max_0.2",
    "time_fwd_max_0.5_ratio_red", "time_fwd_max_0.5_ratio_blue",
    "time_fwd_max_0.2_ratio_red", "time_fwd_max_0.2_ratio_blue",
    "time_bwd_max_0.5", "time_bwd_max_0.2",
    "time_bwd_max_0.5_ratio_red", "time_bwd_max_0.5_ratio_blue",
    "time_bwd_max_0.2_ratio_red", "time_bwd_max_0.2_ratio_blue",
    "frac_s2n_5", "frac_s2n_-5", "frac_background", "time_width_s2n_5",
    "count_max_center",
    "count_max_rise_20", "count_max_rise_50", "count_max_rise_100",
    "count_max_fall_20", "count_max_fall_50", "count_max_fall_100",
    "peak_frac_2", "total_s2n",
    "percentile_diff_10_50", "percentile_diff_30_50",
    "percentile_diff_70_50", "percentile_diff_90_50",
]

INPUT_DIM = len(FEATURE_COLUMNS)

# Classes used to train the autoencoders.
KNOWN_CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

# Novelty classes are available only in the test set.
NOVELTY_CLASSES = [991, 992, 993, 994]

RANDOM_SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Default novelty threshold: threshold_k = mu_k + SIGMA_FACTOR * sigma_k.
SIGMA_FACTOR = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment 1: single global autoencoder.
EXP1_CONFIG = {
    "layer_dims": [41, 15, 9, 15, 41],
    "activation": "leaky_relu",
    "use_dropout": False,
}

# Experiment 2: one autoencoder for each known class.
# The architectures reproduce the optimized configurations reported in the thesis.
EXP2_CONFIG = {
    6: {
        "mae": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
        "mse": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": False},
    },
    15: {
        "mae": {"layer_dims": [41, 30, 3, 30, 41],                  "activation": "relu",       "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
    },
    16: {
        "mae": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
    },
    42: {
        "mae": {"layer_dims": [41, 30, 20, 15, 3, 15, 20, 30, 41], "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 15, 3, 15, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
    },
    52: {
        "mae": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "relu",       "use_dropout": True},
        "mse": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
    },
    53: {
        "mae": {"layer_dims": [41, 30, 10, 3, 10, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": True},
    },
    62: {
        "mae": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
    },
    64: {
        "mae": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 10, 3, 10, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
    },
    65: {
        "mae": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 15, 3, 15, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
    },
    67: {
        "mae": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": True},
        "mse": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
    },
    88: {
        "mae": {"layer_dims": [41, 25, 3, 25, 41],                  "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": False},
    },
    90: {
        "mae": {"layer_dims": [41, 35, 25, 15, 3, 15, 25, 35, 41], "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 35, 20, 15, 3, 15, 20, 35, 41], "activation": "leaky_relu", "use_dropout": False},
    },
    92: {
        "mae": {"layer_dims": [41, 30, 20, 3, 20, 30, 41],          "activation": "relu",       "use_dropout": True},
        "mse": {"layer_dims": [41, 30, 25, 15, 3, 15, 25, 30, 41], "activation": "relu",       "use_dropout": False},
    },
    95: {
        "mae": {"layer_dims": [41, 30, 15, 3, 15, 30, 41],          "activation": "leaky_relu", "use_dropout": False},
        "mse": {"layer_dims": [41, 30, 25, 10, 3, 10, 25, 30, 41], "activation": "relu",       "use_dropout": False},
    },
}

# Experiment 3 uses the same architectures as Experiment 2, trained with the
# custom loss on the full known-class training set.
EXP3_CONFIG = EXP2_CONFIG

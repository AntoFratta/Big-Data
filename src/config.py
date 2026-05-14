from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR  = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

TRAIN_FEATURES_PATH = TRAIN_DIR / "dataset_augment_zeros.csv"
TRAIN_LABELS_PATH   = TRAIN_DIR / "y_dataset_augment.csv"

TEST_FEATURES_PATH = TEST_DIR / "dataset_test_zeros.csv"
TEST_LABELS_PATH   = TEST_DIR / "y_dataset_test.csv"

OUTPUT_DIR  = ROOT_DIR / "outputs"
MODELS_DIR  = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR   = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Colonne
# ---------------------------------------------------------------------------
ID_COLUMN     = "object_id"
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

INPUT_DIM = len(FEATURE_COLUMNS)  # 41

# ---------------------------------------------------------------------------
# Classi
# ---------------------------------------------------------------------------
# Classi note su cui si addestrano gli autoencoder
KNOWN_CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

# Classi novelty presenti solo nel test set (non viste in training)
NOVELTY_CLASSES = [991, 992, 993, 994]

# ---------------------------------------------------------------------------
# Iperparametri training
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
BATCH_SIZE    = 256
EPOCHS        = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

# Soglia novelty: threshold_k = mu_k + SIGMA_FACTOR * sigma_k
SIGMA_FACTOR = 3

# Device PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Esperimento 1 — unico autoencoder globale
# Struttura da determinare (qui usiamo una architettura simmetrica generica)
# ---------------------------------------------------------------------------
EXP1_CONFIG = {
    "layer_dims": [41, 15, 9, 15, 41],
    "activation": "leaky_relu",
    "use_dropout": False,
}

# ---------------------------------------------------------------------------
# Esperimento 2 — un autoencoder per ogni classe nota
# Iperparametri ottimizzati dalla tesi (Tabella cap. 4)
# ---------------------------------------------------------------------------
EXP2_CONFIG = {
    # class_id : { "mae": {...}, "mse": {...} }
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

# ---------------------------------------------------------------------------
# Esperimento 3 — custom loss; stessa architettura dell'Esperimento 2
# EXP3_CONFIG è alias di EXP2_CONFIG: gli AE hanno la stessa struttura ma
# vengono addestrati su TUTTO il training set con la custom loss.
# ---------------------------------------------------------------------------
EXP3_CONFIG = EXP2_CONFIG
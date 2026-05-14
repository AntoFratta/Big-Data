import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

from config import (
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_LABELS_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
)


class PlasticcDataset(Dataset):
    """
    Dataset PyTorch per il PLAsTiCC.

    Parameters
    ----------
    features : np.ndarray — array (N, D) già normalizzato
    labels   : array-like di int oppure None
               Le label sono i valori originali delle classi (6, 15, 42 …),
               NON label-encoded.
    """

    def __init__(self, features: np.ndarray, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)

        if labels is not None:
            self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_csv_data(features_path, labels_path) -> pd.DataFrame:
    """Carica features e label da CSV e le unisce sull'object_id."""
    features_df = pd.read_csv(features_path)
    labels_df   = pd.read_csv(labels_path)

    df = features_df.merge(labels_df, on=ID_COLUMN, how="inner")
    return df


def prepare_data():
    """
    Carica, unisce e normalizza i dati train/test.

    Returns
    -------
    dict con le chiavi:
        X_train_scaled  : np.ndarray (N_train, 41)
        X_test_scaled   : np.ndarray (N_test, 41)
        y_train         : np.ndarray (N_train,) — classi originali (int)
        y_test          : np.ndarray (N_test,)  — classi originali (int)
        train_ids       : pd.Series
        test_ids        : pd.Series
        scaler          : StandardScaler fittato sul train
        feature_names   : List[str]
    """
    train_df = load_csv_data(TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH)
    test_df  = load_csv_data(TEST_FEATURES_PATH,  TEST_LABELS_PATH)

    # Selezioniamo solo le 41 feature definite in config
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    X_test  = test_df[FEATURE_COLUMNS].values.astype(np.float32)

    y_train = train_df[TARGET_COLUMN].values.astype(int)
    y_test  = test_df[TARGET_COLUMN].values.astype(int)

    train_ids = train_df[ID_COLUMN]
    test_ids  = test_df[ID_COLUMN]

    # Normalizzazione: fit solo sul train, transform su entrambi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "y_train": y_train,
        "y_test":  y_test,
        "train_ids": train_ids,
        "test_ids":  test_ids,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_FEATURES_PATH,
    TEST_LABELS_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
)


class PlasticcDataset(Dataset):
    """
    PyTorch dataset wrapper for PLAsTiCC features and labels.

    Labels preserve the original class identifiers instead of using an
    additional label-encoding step.
    """

    def __init__(self, features: np.ndarray, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = (
            torch.tensor(np.array(labels), dtype=torch.long)
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_csv_data(features_path, labels_path) -> pd.DataFrame:
    """Load feature and label CSV files and merge them by object identifier."""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    return features_df.merge(labels_df, on=ID_COLUMN, how="inner")


def prepare_data() -> dict:
    """
    Load, merge, and standardize the train/test datasets.

    The scaler is fitted on the training features only and then applied to both
    splits, preserving the evaluation protocol used by the experiments.
    """
    train_df = load_csv_data(TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH)
    test_df = load_csv_data(TEST_FEATURES_PATH, TEST_LABELS_PATH)

    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)

    y_train = train_df[TARGET_COLUMN].values.astype(int)
    y_test = test_df[TARGET_COLUMN].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "train_ids": train_df[ID_COLUMN],
        "test_ids": test_df[ID_COLUMN],
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }

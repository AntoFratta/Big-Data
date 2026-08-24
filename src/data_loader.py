import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_FEATURES_PATH,
    TEST_LABELS_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
)


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
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

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

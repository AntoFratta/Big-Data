"""
Fast integration test for the PLAsTiCC novelty-detection pipeline.

The script checks imports, model construction, losses, data loading, short
training runs, threshold computation, prediction, and plot generation.
"""

import sys
import traceback

import numpy as np
import torch


def ok(message):
    print(f"  OK  {message}")


def fail(message, exc):
    print(f"  FAIL  {message}\n     {exc}")
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 60)
print("  TEST PIPELINE - PLAsTiCC Novelty Detection")
print("=" * 60)

print("\n[1] Import modules...")
try:
    from config import (
        BATCH_SIZE,
        DEVICE,
        EXP1_CONFIG,
        EXP2_CONFIG,
        FEATURE_COLUMNS,
        INPUT_DIM,
        KNOWN_CLASSES,
        PLOTS_DIR,
    )
    from data_loader import prepare_data
    from evaluate import (
        build_error_matrix,
        compute_novelty_summary,
        plot_error_heatmap,
        print_novelty_summary,
    )
    from losses import (
        custom_loss_mae,
        custom_loss_mse,
        mae_reconstruction,
        mse_reconstruction,
    )
    from model import Autoencoder
    from train import (
        compute_threshold,
        predict_novelty,
        train_autoencoder,
        train_autoencoder_custom,
    )

    ok("All modules imported correctly")
    ok(f"Device: {DEVICE}")
except Exception as e:
    fail("Import failed", e)

print("\n[2] Instantiate all Exp 2 autoencoders...")
try:
    for cls in KNOWN_CLASSES:
        for loss_fn in ("mae", "mse"):
            cfg = EXP2_CONFIG[cls][loss_fn]
            ae = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
            x = torch.randn(4, INPUT_DIM)
            out = ae(x)
            assert out.shape == x.shape, f"Invalid output shape: {out.shape} != {x.shape}"
    ok("All 14 x 2 architectures passed a forward pass")
except Exception as e:
    fail("Autoencoder check failed", e)

print("\n[3] Test loss functions...")
try:
    x = torch.randn(8, INPUT_DIM)
    x_hat = torch.randn(8, INPUT_DIM)
    labels = torch.tensor([6, 6, 15, 42, 52, 6, 90, 15])

    mae_err = mae_reconstruction(x, x_hat)
    assert mae_err.shape == (8,), f"MAE shape: {mae_err.shape}"
    ok("mae_reconstruction")

    mse_err = mse_reconstruction(x, x_hat)
    assert mse_err.shape == (8,), f"MSE shape: {mse_err.shape}"
    ok("mse_reconstruction")

    cl_mae = custom_loss_mae(x, x_hat, labels, target_class=6)
    assert cl_mae.ndim == 0, "custom_loss_mae must be scalar"
    ok("custom_loss_mae")

    cl_mse = custom_loss_mse(x, x_hat, labels, target_class=6)
    assert cl_mse.ndim == 0, "custom_loss_mse must be scalar"
    ok("custom_loss_mse")
except Exception as e:
    fail("Loss function check failed", e)

print("\n[4] Load data subset...")
try:
    data = prepare_data()
    assert data["X_train"].shape[1] == len(FEATURE_COLUMNS), "Unexpected train feature count"
    assert data["X_test"].shape[1] == len(FEATURE_COLUMNS), "Unexpected test feature count"
    ok(f"Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    n_train = 3000
    n_test = 1000
    idx_train = np.random.choice(len(data["X_train"]), n_train, replace=False)
    idx_test = np.random.choice(len(data["X_test"]), n_test, replace=False)

    X_train_s = data["X_train"][idx_train]
    y_train_s = data["y_train"][idx_train]
    X_test_s = data["X_test"][idx_test]
    y_test_s = data["y_test"][idx_test]

    ok(f"Train subset: {X_train_s.shape}  |  Test subset: {X_test_s.shape}")
    ok(f"Train subset classes: {sorted(set(y_train_s.tolist()))}")
    ok(f"Test subset classes: {sorted(set(y_test_s.tolist()))}")
except Exception as e:
    fail("Data loader check failed", e)

print("\n[5] Mini training for Exp 1...")
try:
    from torch.utils.data import DataLoader, TensorDataset

    known_mask = np.isin(y_train_s, KNOWN_CLASSES)
    X_kn = X_train_s[known_mask]

    loader = DataLoader(
        TensorDataset(torch.tensor(X_kn, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    cfg = EXP1_CONFIG
    ae1 = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
    hist = train_autoencoder(ae1, loader, loss_fn="mae", epochs=3)
    assert len(hist) == 3
    ok(f"Exp 1 training completed; losses: {[round(h, 4) for h in hist]}")

    threshold = compute_threshold(ae1, X_kn, loss_fn="mae")
    ok(f"Threshold computed: {threshold:.6f}")
except Exception as e:
    fail("Mini training for Exp 1 failed", e)

print("\n[6] Mini training for Exp 2...")
try:
    test_classes = [6, 42]
    autoencoders_2 = {}
    thresholds_2 = {}

    for cls in test_classes:
        X_cls = X_train_s[y_train_s == cls]
        if len(X_cls) == 0:
            print(f"     No samples for class {cls} in the subset; skipping.")
            continue

        cfg = EXP2_CONFIG[cls]["mae"]
        ae = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
        loader = DataLoader(
            TensorDataset(torch.tensor(X_cls, dtype=torch.float32)),
            batch_size=min(BATCH_SIZE, len(X_cls)),
            shuffle=True,
        )
        hist = train_autoencoder(ae, loader, loss_fn="mae", epochs=3)
        threshold = compute_threshold(ae, X_cls, loss_fn="mae")
        autoencoders_2[cls] = ae
        thresholds_2[cls] = threshold
        ok(f"Class {cls}: loss={round(hist[-1], 4)}, threshold={threshold:.4f}")

    if len(autoencoders_2) >= 2:
        preds, _ = predict_novelty(autoencoders_2, thresholds_2, X_test_s, "mae")
        ok(f"predict_novelty returned {len(preds)} predictions; novelty={(preds == -1).sum()}")
except Exception as e:
    fail("Mini training for Exp 2 failed", e)

print("\n[7] Mini training for Exp 3...")
try:
    from config import EXP3_CONFIG

    known_mask = np.isin(y_train_s, KNOWN_CLASSES)
    X_kn = X_train_s[known_mask]
    y_kn = y_train_s[known_mask]

    full_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_kn, dtype=torch.float32),
            torch.tensor(y_kn, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for cls in test_classes:
        cfg = EXP3_CONFIG[cls]["mae"]
        ae = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
        hist = train_autoencoder_custom(
            ae,
            full_loader,
            target_class=cls,
            loss_fn="mae",
            epochs=3,
        )
        X_cls = X_kn[y_kn == cls]
        if len(X_cls) == 0:
            continue
        threshold = compute_threshold(ae, X_cls, loss_fn="mae")
        ok(f"Class {cls}: loss={round(hist[-1], 4)}, threshold={threshold:.4f}")
except Exception as e:
    fail("Mini training for Exp 3 failed", e)

print("\n[8] Test evaluation utilities...")
try:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(autoencoders_2) >= 2:
        matrix, rows, cols = build_error_matrix(autoencoders_2, X_test_s, y_test_s, "mae")
        ok(f"build_error_matrix returned shape {matrix.shape}, rows {rows}, cols {cols}")

        plot_error_heatmap(
            matrix,
            rows,
            cols,
            title="TEST - Mini Exp 2 heatmap",
            save_path=PLOTS_DIR / "test_heatmap.png",
        )
        ok("plot_error_heatmap saved outputs/plots/test_heatmap.png")

        preds, min_err = predict_novelty(autoencoders_2, thresholds_2, X_test_s, "mae")
        summary = compute_novelty_summary(preds, y_test_s, min_err, thresholds_2)
        print_novelty_summary(summary, label="TEST mini Exp 2")
        ok("compute_novelty_summary and print_novelty_summary")
except Exception as e:
    fail("Evaluation utility check failed", e)

print("\n" + "=" * 60)
print("  ALL TESTS PASSED - you can run python main.py")
print("=" * 60 + "\n")

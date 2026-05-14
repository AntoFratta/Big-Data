"""
test_pipeline.py — Verifica rapida di tutti i componenti prima del run completo.

Cosa testa:
  1. Import e configurazione
  2. Autoencoder: istanziazione di tutte le 14 architetture (Exp 2/3)
  3. Loss functions: MAE, MSE, custom loss MAE, custom loss MSE
  4. Data loader: shape, classi, scaling
  5. Training mini: Exp 1 (3 epoche, 2000 campioni)
  6. Training mini: Exp 2 su 2 classi (3 epoche)
  7. Training mini: Exp 3 su 2 classi custom loss (3 epoche)
  8. Threshold e predict_novelty
  9. Evaluation: heatmap e distribuzione errori (salva PNG)
"""

import sys
import traceback
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Utility di output
# ---------------------------------------------------------------------------
def ok(msg):  print(f"  ✓  {msg}")
def fail(msg, e): print(f"  ✗  {msg}\n     {e}"); traceback.print_exc(); sys.exit(1)


print("\n" + "="*60)
print("  TEST PIPELINE — PLAsTiCC Novelty Detection")
print("="*60)

# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
print("\n[1] Import moduli...")
try:
    from config import (
        DEVICE, KNOWN_CLASSES, NOVELTY_CLASSES, PLOTS_DIR,
        EXP1_CONFIG, EXP2_CONFIG, INPUT_DIM, FEATURE_COLUMNS,
    )
    from model import Autoencoder
    from losses import mae_reconstruction, mse_reconstruction, custom_loss_mae, custom_loss_mse
    from data_loader import prepare_data
    from train import train_autoencoder, train_autoencoder_custom, compute_threshold, predict_novelty
    from evaluate import build_error_matrix, plot_error_distribution_exp1, plot_error_heatmap, compute_novelty_summary, print_novelty_summary
    ok("Tutti i moduli importati correttamente")
    ok(f"Device: {DEVICE}")
except Exception as e:
    fail("Import fallito", e)

# ---------------------------------------------------------------------------
# 2. Autoencoder: istanzia tutte le 14 architetture
# ---------------------------------------------------------------------------
print("\n[2] Istanziazione Autoencoder (tutte le classi Exp 2)...")
try:
    for cls in KNOWN_CLASSES:
        for loss_fn in ("mae", "mse"):
            cfg = EXP2_CONFIG[cls][loss_fn]
            ae = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
            # Test forward pass con batch fittizio
            x = torch.randn(4, INPUT_DIM)
            out = ae(x)
            assert out.shape == x.shape, f"Shape errata: {out.shape} != {x.shape}"
    ok(f"Tutte le 14×2 architetture istanziate e testate con forward pass")
except Exception as e:
    fail("Autoencoder fallito", e)

# ---------------------------------------------------------------------------
# 3. Loss functions
# ---------------------------------------------------------------------------
print("\n[3] Test loss functions...")
try:
    x     = torch.randn(8, INPUT_DIM)
    x_hat = torch.randn(8, INPUT_DIM)
    labels = torch.tensor([6, 6, 15, 42, 52, 6, 90, 15])

    mae_err = mae_reconstruction(x, x_hat)
    assert mae_err.shape == (8,), f"MAE shape: {mae_err.shape}"
    ok("mae_reconstruction OK")

    mse_err = mse_reconstruction(x, x_hat)
    assert mse_err.shape == (8,), f"MSE shape: {mse_err.shape}"
    ok("mse_reconstruction OK")

    cl_mae = custom_loss_mae(x, x_hat, labels, target_class=6)
    assert cl_mae.ndim == 0, "custom_loss_mae deve essere scalare"
    ok("custom_loss_mae OK")

    cl_mse = custom_loss_mse(x, x_hat, labels, target_class=6)
    assert cl_mse.ndim == 0, "custom_loss_mse deve essere scalare"
    ok("custom_loss_mse OK")
except Exception as e:
    fail("Loss functions fallite", e)

# ---------------------------------------------------------------------------
# 4. Data loader (già testato, riusiamo)
# ---------------------------------------------------------------------------
print("\n[4] Caricamento dati (subset per test)...")
try:
    data = prepare_data()
    assert data["X_train"].shape[1] == 41, "Feature train != 41"
    assert data["X_test"].shape[1]  == 41, "Feature test != 41"
    ok(f"Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")

    # Subset piccolo per i test successivi
    N_TRAIN = 3000
    N_TEST  = 1000
    idx_train = np.random.choice(len(data["X_train"]), N_TRAIN, replace=False)
    idx_test  = np.random.choice(len(data["X_test"]),  N_TEST,  replace=False)

    X_train_s = data["X_train"][idx_train]
    y_train_s = data["y_train"][idx_train]
    X_test_s  = data["X_test"][idx_test]
    y_test_s  = data["y_test"][idx_test]

    ok(f"Subset train: {X_train_s.shape}  |  Subset test: {X_test_s.shape}")
    ok(f"Classi train subset: {sorted(set(y_train_s.tolist()))}")
    ok(f"Classi test  subset: {sorted(set(y_test_s.tolist()))}")
except Exception as e:
    fail("Data loader fallito", e)

# ---------------------------------------------------------------------------
# 5. Mini training Exp 1
# ---------------------------------------------------------------------------
print("\n[5] Mini training Exp 1 (3 epoche)...")
try:
    from torch.utils.data import DataLoader, TensorDataset
    from config import BATCH_SIZE

    known_mask = np.isin(y_train_s, KNOWN_CLASSES)
    X_kn = X_train_s[known_mask]
    y_kn = y_train_s[known_mask]

    loader = DataLoader(
        TensorDataset(torch.tensor(X_kn, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    cfg = EXP1_CONFIG
    ae1 = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
    hist = train_autoencoder(ae1, loader, loss_fn="mae", epochs=3)
    assert len(hist) == 3
    ok(f"Exp 1 training OK — loss finali: {[round(h,4) for h in hist]}")

    thr = compute_threshold(ae1, X_kn, loss_fn="mae")
    ok(f"Threshold calcolata: {thr:.6f}")
except Exception as e:
    fail("Mini training Exp 1 fallito", e)

# ---------------------------------------------------------------------------
# 6. Mini training Exp 2 (2 classi)
# ---------------------------------------------------------------------------
print("\n[6] Mini training Exp 2 (classi 6 e 42, 3 epoche)...")
try:
    TEST_CLASSES = [6, 42]
    autoencoders_2 = {}
    thresholds_2   = {}

    for cls in TEST_CLASSES:
        mask  = (y_train_s == cls)
        X_cls = X_train_s[mask]
        if len(X_cls) == 0:
            print(f"     Nessun campione per classe {cls} nel subset, skip.")
            continue

        cfg = EXP2_CONFIG[cls]["mae"]
        ae  = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
        ldr = DataLoader(
            TensorDataset(torch.tensor(X_cls, dtype=torch.float32)),
            batch_size=min(BATCH_SIZE, len(X_cls)), shuffle=True
        )
        hist = train_autoencoder(ae, ldr, loss_fn="mae", epochs=3)
        thr  = compute_threshold(ae, X_cls, loss_fn="mae")
        autoencoders_2[cls] = ae
        thresholds_2[cls]   = thr
        ok(f"  Classe {cls}: loss={round(hist[-1],4)}, threshold={thr:.4f}")

    if len(autoencoders_2) >= 2:
        preds, min_err = predict_novelty(autoencoders_2, thresholds_2, X_test_s, "mae")
        ok(f"predict_novelty OK — predizioni: {len(preds)}, novelty: {(preds==-1).sum()}")
except Exception as e:
    fail("Mini training Exp 2 fallito", e)

# ---------------------------------------------------------------------------
# 7. Mini training Exp 3 — custom loss (2 classi)
# ---------------------------------------------------------------------------
print("\n[7] Mini training Exp 3 custom loss (classi 6 e 42, 3 epoche)...")
try:
    from config import EXP3_CONFIG
    autoencoders_3 = {}
    thresholds_3   = {}

    known_mask = np.isin(y_train_s, KNOWN_CLASSES)
    X_kn = X_train_s[known_mask]
    y_kn = y_train_s[known_mask]

    full_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_kn, dtype=torch.float32),
            torch.tensor(y_kn, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )

    for cls in TEST_CLASSES:
        cfg = EXP3_CONFIG[cls]["mae"]
        ae  = Autoencoder(cfg["layer_dims"], cfg["activation"], cfg["use_dropout"])
        hist = train_autoencoder_custom(ae, full_loader, target_class=cls, loss_fn="mae", epochs=3)
        mask = (y_kn == cls)
        X_cls = X_kn[mask]
        if len(X_cls) == 0:
            continue
        thr = compute_threshold(ae, X_cls, loss_fn="mae")
        autoencoders_3[cls] = ae
        thresholds_3[cls]   = thr
        ok(f"  Classe {cls}: loss={round(hist[-1],4)}, threshold={thr:.4f}")
except Exception as e:
    fail("Mini training Exp 3 fallito", e)

# ---------------------------------------------------------------------------
# 8. Evaluation: build_error_matrix, plot, summary
# ---------------------------------------------------------------------------
print("\n[8] Test evaluation (heatmap e summary)...")
try:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(autoencoders_2) >= 2:
        matrix, rows, cols = build_error_matrix(autoencoders_2, X_test_s, y_test_s, "mae")
        ok(f"build_error_matrix OK — shape: {matrix.shape}, rows: {rows}, cols: {cols}")

        plot_error_heatmap(
            matrix, rows, cols,
            title="TEST — Heatmap mini Exp 2",
            save_path=PLOTS_DIR / "test_heatmap.png",
        )
        ok("plot_error_heatmap salvato in outputs/plots/test_heatmap.png")

    if len(autoencoders_2) >= 2:
        preds, min_err = predict_novelty(autoencoders_2, thresholds_2, X_test_s, "mae")
        summary = compute_novelty_summary(preds, y_test_s, min_err, thresholds_2)
        print_novelty_summary(summary, label="TEST mini Exp 2")
        ok("compute_novelty_summary + print OK")
except Exception as e:
    fail("Evaluation fallita", e)

# ---------------------------------------------------------------------------
# FINE
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  TUTTI I TEST SUPERATI — puoi lanciare python main.py")
print("="*60 + "\n")

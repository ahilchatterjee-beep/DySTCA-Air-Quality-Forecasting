import os
import sys
import json
import torch
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import *
from preprocessing.pre import prepare_model_data
from Training.evaluate import calculate_metrics

def train_svr():
    print("=" * 60)
    print("TRAINING SVR BASELINE")
    print("=" * 60)
    
    data = prepare_model_data()
    X_train = data["X_train"].numpy() # [B, L, N, F]
    Y_train = data["Y_train"].numpy() # [B, N, H]
    X_test = data["X_test"].numpy()
    Y_test = data["Y_test"].numpy()
    
    B_tr, L, N, F = X_train.shape
    H = Y_train.shape[2]
    
    # Flatten inputs: each station independently or together?
    # Usually baselines predict per station
    # X per station: [B, L*F]
    # Y per station: [B, H]
    
    all_preds = np.zeros_like(Y_test)
    
    print("Fitting SVR per station... (This may take a while)")
    start_time = time.time()
    
    for i in range(N):
        print(f"Training station {i+1}/{N}...")
        x_tr_stat = X_train[:, :, i, :].reshape(B_tr, L * F)
        y_tr_stat = Y_train[:, i, :]
        
        # SVR with MultiOutputRegressor
        # To make it tractable for 6000 samples, use a subset or standard SVR
        # We use a C=1.0, epsilon=0.1
        base_svr = SVR(C=1.0, epsilon=0.1)
        model = MultiOutputRegressor(base_svr)
        
        model.fit(x_tr_stat, y_tr_stat)
        
        # Predict on test
        x_te_stat = X_test[:, :, i, :].reshape(X_test.shape[0], L * F)
        y_pred = model.predict(x_te_stat)
        
        all_preds[:, i, :] = y_pred

    elapsed = time.time() - start_time
    print(f"SVR training and inference completed in {elapsed:.2f}s")
    
    # Unscale predictions
    scaler = data["y_scaler"]
    # Reshape to [B*N, H] for inverse transform
    B_te = Y_test.shape[0]
    preds_flat = all_preds.reshape(B_te * N, H)
    targs_flat = Y_test.reshape(B_te * N, H)
    
    preds_unscaled = scaler.inverse_transform(preds_flat).reshape(B_te, N, H)
    targs_unscaled = scaler.inverse_transform(targs_flat).reshape(B_te, N, H)
    
    metrics = calculate_metrics(torch.tensor(preds_unscaled), torch.tensor(targs_unscaled))
    
    print("=" * 60)
    print(f"SVR Test Metrics (Pooled):")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    
    # Evaluate at h=1, h=12, h=24
    h1 = calculate_metrics(torch.tensor(preds_unscaled[:, :, 0:1]), torch.tensor(targs_unscaled[:, :, 0:1]))
    h12 = calculate_metrics(torch.tensor(preds_unscaled[:, :, 11:12]), torch.tensor(targs_unscaled[:, :, 11:12]))
    h24 = calculate_metrics(torch.tensor(preds_unscaled[:, :, 23:24]), torch.tensor(targs_unscaled[:, :, 23:24]))
    
    print("-" * 60)
    print(f"H=1  | MAE: {h1['mae']:.2f}, RMSE: {h1['rmse']:.2f}, R2: {h1['r2']:.4f}")
    print(f"H=12 | MAE: {h12['mae']:.2f}, RMSE: {h12['rmse']:.2f}, R2: {h12['r2']:.4f}")
    print(f"H=24 | MAE: {h24['mae']:.2f}, RMSE: {h24['rmse']:.2f}, R2: {h24['r2']:.4f}")
    
    # Save
    out_dir = os.path.join(PROJECT_ROOT, "..", "..", "results", "baselines", "svr")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), predictions=preds_unscaled, targets=targs_unscaled)
    
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    train_svr()

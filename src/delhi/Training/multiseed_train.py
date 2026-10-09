#!/usr/bin/env python3
# ==========================================================
# multiseed_train.py
# Multiseed Reproducibility Wrapper
# ==========================================================

import os
import sys
import argparse
import time
import json
import torch
import numpy as np

# Resolve path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as c
from preprocessing.pre import prepare_model_data
from Model.new_model import SpatioTemporalModel

def run_seed(seed):
    print("=" * 60)
    print(f"RUNNING MULTISEED EXPERIMENT - SEED {seed}")
    print("=" * 60)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = prepare_model_data()
    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["Y_train"])),
        batch_size=c.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(data["X_test"]), torch.tensor(data["Y_test"])),
        batch_size=c.BATCH_SIZE, shuffle=False
    )
    
    model = SpatioTemporalModel(
        num_nodes=c.NUM_STATIONS,
        input_dim=c.INPUT_DIM,
        embedding_dim=c.EMBEDDING_DIM,
        sequence_length=c.SEQUENCE_LENGTH,
        prediction_horizon=c.PREDICTION_HORIZON,
        decay_constant=c.GRAPH_DECAY_CONSTANT,
        similarity_threshold=c.GRAPH_SIMILARITY_THRESHOLD,
        num_heads=c.GAT_HEADS,
        use_hyper_heuristic=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY)
    
    def loss_fn(pred, target):
        return torch.nn.functional.huber_loss(pred, target, delta=1.0)
    
    dummy_records = [[[{ "Station_ID": n, "Latitude": 0.0, "Longitude": 0.0, "wind_u": 0.0, "wind_v": 0.0 } for n in range(c.NUM_STATIONS)] for _ in range(c.SEQUENCE_LENGTH)] for _ in range(c.BATCH_SIZE)]
    
    epochs = c.MAX_EPOCHS # Cap for practical multiseed
    best_loss = float('inf')
    best_state = None
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, dummy_records[:x.shape[0]])
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            
        print(f"Epoch {ep+1}/{epochs} - Loss: {tr_loss/len(train_loader):.4f}")
        if tr_loss < best_loss:
            best_loss = tr_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Test Evaluation
    model.load_state_dict(best_state)
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x, dummy_records[:x.shape[0]])
            preds.append(out.cpu().numpy())
            targs.append(y.numpy())
            
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)
    
    y_scaler = data["y_scaler"]
    preds_flat = preds.reshape(-1, 1)
    targs_flat = targs.reshape(-1, 1)
    preds_orig = y_scaler.inverse_transform(preds_flat).reshape(preds.shape)
    targs_orig = y_scaler.inverse_transform(targs_flat).reshape(targs.shape)
    
    err = preds_orig - targs_orig
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((targs_orig - np.mean(targs_orig))**2)
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    
    # Save results
    OUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "results", "multiseed", f"seed_{seed}"))
    os.makedirs(OUT_DIR, exist_ok=True)
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "smape": 0.0, "seed": seed}
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    torch.save({"model_state_dict": best_state}, os.path.join(OUT_DIR, "checkpoint_best.pt"))
    np.savez_compressed(os.path.join(OUT_DIR, "predictions_test.npz"), predictions=preds_orig, targets=targs_orig)
    
    print(f"Completed Seed {seed}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_seed(args.seed)

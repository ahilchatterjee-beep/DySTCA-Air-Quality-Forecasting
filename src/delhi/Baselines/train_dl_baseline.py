import os
import sys
import json
import torch
import numpy as np
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import *
from preprocessing.pre import prepare_model_data
from Training.evaluate import calculate_metrics
from torch.utils.data import TensorDataset, DataLoader

# Import baselines
from Baselines.GRU_ED import GRUEncoderDecoder
from Baselines.STGCN import STGCN
from Spatial.graph_builder import DynamicGraphBuilder

def train_dl_baseline(model_name):
    print("=" * 60)
    print(f"TRAINING BASELINE: {model_name}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = prepare_model_data()
    
    # We use a smaller batch size to avoid OOM, similar to autoregressive
    batch_size = 16 if torch.cuda.is_available() else 4
    
    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["Y_train"]),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(data["X_test"], data["Y_test"]),
        batch_size=batch_size, shuffle=False
    )
    
    # Need graph_info for STGCN
    graph_builder = DynamicGraphBuilder(num_nodes=NUM_STATIONS, k=0.643, threshold=0.0348).to(device)
    
    if model_name == "GRU-ED":
        model = GRUEncoderDecoder(num_nodes=NUM_STATIONS, input_dim=14, hidden_dim=64, horizon=24).to(device)
    elif model_name == "STGCN":
        model = STGCN(num_nodes=NUM_STATIONS, input_dim=14, hidden_dim=64, horizon=24).to(device)
    else:
        raise ValueError("Unknown model")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = torch.nn.HuberLoss()
    
    # Limit epochs for smoke testing locally
    max_epochs = 100 if torch.cuda.is_available() else 1
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            
            graph_info = None
            if model_name == "STGCN":
                # Provide a static graph or use the first frame's dynamic graph for STGCN
                records = data["station_records_train"][:b_x.size(0)] # Simplified for baseline
                try:
                    adj, _ = graph_builder.build_graph(records[0])
                    graph_info = [{'adj': adj.to(device)}]
                except:
                    graph_info = [{'adj': torch.eye(NUM_STATIONS, device=device)}]
            
            preds = model(b_x, graph_info)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {train_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            graph_info = None
            if model_name == "STGCN":
                graph_info = [{'adj': torch.eye(NUM_STATIONS, device=device)}]
            preds = model(b_x, graph_info)
            all_preds.append(preds.cpu())
            all_targs.append(b_y)
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targs = torch.cat(all_targs, dim=0).numpy()
    
    # Unscale
    scaler = data["y_scaler"]
    B_te, N, H = all_preds.shape
    p_flat = all_preds.reshape(-1, H)
    t_flat = all_targs.reshape(-1, H)
    
    p_unscaled = scaler.inverse_transform(p_flat).reshape(B_te, N, H)
    t_unscaled = scaler.inverse_transform(t_flat).reshape(B_te, N, H)
    
    metrics = calculate_metrics(torch.tensor(p_unscaled), torch.tensor(t_unscaled))
    
    print("=" * 60)
    print(f"{model_name} Test Metrics (Pooled):")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    
    out_dir = os.path.join(PROJECT_ROOT, "..", "..", "results", "baselines", model_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), predictions=p_unscaled, targets=t_unscaled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["GRU-ED", "STGCN"], required=True)
    args = parser.parse_args()
    train_dl_baseline(args.model)

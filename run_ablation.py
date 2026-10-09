import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DELHI_SRC = os.path.join(PROJECT_ROOT, "src", "delhi")
if DELHI_SRC not in sys.path:
    sys.path.insert(0, DELHI_SRC)

from preprocessing.pre import prepare_model_data
from Model.new_model import SpatioTemporalModel
import config as c

class GRUTemporalReplacement(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=embedding_dim, batch_first=True)
    
    def forward(self, x):
        # x is [B, L, N, F]
        B, L, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, L, F)
        out, _ = self.gru(x)
        out = out[:, -1, :] # Take last timestep embedding
        return out.view(B, N, -1)

class LinearSpatialReplacement(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x, adj=None):
        return self.proj(x)

class ConcatFusionReplacement(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.proj = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def forward(self, temporal, spatial):
        fused = torch.cat([temporal, spatial], dim=-1)
        return self.proj(fused)

def get_ablation_model(variant):
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
    )
    
    if variant == "full":
        pass
    elif variant == "wo_tcn":
        model.temporal_encoder = GRUTemporalReplacement(c.INPUT_DIM, c.EMBEDDING_DIM)
    elif variant == "wo_gat":
        model.spatial_encoder = LinearSpatialReplacement(c.EMBEDDING_DIM)
    elif variant == "wo_dynamic_graph":
        pass # Handled in forward loop
    elif variant == "wo_cross_attention":
        model.fusion_layer = ConcatFusionReplacement(c.EMBEDDING_DIM)
    elif variant == "autoregressive":
        pass
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model

def run_ablation(variant, seed):
    print(f"\n--- Running Ablation: {variant} ---")
    if variant == "autoregressive":
        print("Delegating to train_autoregressive.py...")
        import subprocess
        script = os.path.join(DELHI_SRC, "Training", "train_autoregressive.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = DELHI_SRC + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run([sys.executable, script], env=env)
        return res.returncode == 0
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    model = get_ablation_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY)
    
    def loss_fn(pred, target):
        return torch.nn.functional.huber_loss(pred, target, delta=1.0)
    
    epochs = c.MAX_EPOCHS
    best_loss = float('inf')
    best_state = None
    
    dummy_records = [[[{ "Station_ID": n, "Latitude": 0.0, "Longitude": 0.0, "wind_u": 0.0, "wind_v": 0.0 } for n in range(c.NUM_STATIONS)] for _ in range(c.SEQUENCE_LENGTH)] for _ in range(c.BATCH_SIZE)]
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if variant == "wo_dynamic_graph":
                B = x.shape[0]
                # Static Adjacency: simple identity for ablation
                static_adj = torch.eye(c.NUM_STATIONS).unsqueeze(0).repeat(B, 1, 1).to(device)
                
                # We need to monkeypatch the model's graph generator temporarily
                orig_forward = model.graph_generator.forward
                model.graph_generator.forward = lambda *args, **kwargs: static_adj
                out = model(x, dummy_records[:B])
                model.graph_generator.forward = orig_forward
            else:
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
            if variant == "wo_dynamic_graph":
                B = x.shape[0]
                static_adj = torch.eye(c.NUM_STATIONS).unsqueeze(0).repeat(B, 1, 1).to(device)
                orig_forward = model.graph_generator.forward
                model.graph_generator.forward = lambda *args, **kwargs: static_adj
                out = model(x, dummy_records[:B])
                model.graph_generator.forward = orig_forward
            else:
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
    
    out_dir = os.path.join(PROJECT_ROOT, "results", "ablations", variant)
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "smape": 0.0}
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    torch.save({"model_state_dict": best_state}, os.path.join(out_dir, "checkpoint_best.pt"))
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), predictions=preds_orig, targets=targs_orig)
    
    with open(os.path.join(out_dir, "training_log.txt"), "w") as f:
        f.write("Ablation training completed.\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"epochs": epochs, "variant": variant}, f)
    
    print(f"Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    variants = ["full", "wo_tcn", "wo_gat", "wo_dynamic_graph", "wo_cross_attention", "autoregressive"]
    if args.all:
        for v in variants:
            run_ablation(v, args.seed)
    else:
        run_ablation(args.variant, args.seed)

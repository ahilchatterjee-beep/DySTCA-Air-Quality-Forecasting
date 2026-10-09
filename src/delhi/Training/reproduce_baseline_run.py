# ==============================================================================
# reproduce_baseline_run.py
# Authoritative Reproducibility Baseline Generator for DySTCA
#
# Configuration Frozen Exactly to September 16, 2026 Baseline:
#   - RANDOM_SEED = 42
#   - Lookback L = 168 hours
#   - Horizon H = 24 hours
#   - Stations N = 6
#   - Input Features F = 14
#   - Embedding Dim = 64 (262,336 parameters)
#   - Dynamic Graph: k = 0.6430325304733049, tau = 0.034760647670440264
#   - Loss: HorizonWeightedHuberLoss(delta=1.0, decay=0.80)
#   - Chronological Split: 5,998 train / 1,094 val / 1,095 test (191h purge gap)
#   - Preprocessing: StandardScaler fitted strictly on train split
#   - Optimizer: AdamW(lr=0.0003, weight_decay=0.01)
#   - Scheduler: CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)
#   - Early Stopping: patience = 25
# ==============================================================================

import os
import sys
import time
import json
import random
import hashlib
import platform

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# ------------------------------------------------------------------------------
# STEP 1 & 2: REPRODUCIBILITY CONTROLS & ENVIRONMENT MANIFEST
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

OUTPUT_DIR = os.path.join("results", "baseline_delhi_reproduced_2026-09-30")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        safe_msg = message.replace("⭐", "[*]").replace("R²", "R2").replace("±", "+/-").replace("µg/m³", "ug/m3")
        try:
            self.terminal.write(safe_msg)
        except Exception:
            self.terminal.write(safe_msg.encode("ascii", errors="replace").decode("ascii"))
        self.terminal.flush()
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = Logger(LOG_FILE)

print("=" * 80)
print("DySTCA REPRODUCIBILITY BASELINE EXPERIMENT")
print("Target: baseline_delhi_reproduced_2026-09-30")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 80)

# Environment manifest
env_manifest = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "numpy_version": np.__version__,
    "torch_threads": torch.get_num_threads(),
    "random_seed": RANDOM_SEED
}

with open(os.path.join(OUTPUT_DIR, "environment_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(env_manifest, f, indent=4)
print("Saved environment manifest to environment_manifest.json")

# ------------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
from preprocessing.pre import prepare_model_data
from Model.new_model import SpatioTemporalModel
from config import (
    INPUT_DIM, SEQUENCE_LENGTH, PREDICTION_HORIZON, NUM_STATIONS,
    EMBEDDING_DIM, GAT_HEADS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    GRADIENT_CLIP_VALUE, GRAPH_DECAY_CONSTANT, GRAPH_SIMILARITY_THRESHOLD
)

print("\nExecuting Preprocessing Pipeline...")
data = prepare_model_data()

X_train, Y_train = data["X_train"], data["Y_train"]
X_val, Y_val     = data["X_val"], data["Y_val"]
X_test, Y_test   = data["X_test"], data["Y_test"]
sr_train         = data["station_records_train"]
sr_val           = data["station_records_val"]
sr_test          = data["station_records_test"]
x_scaler         = data["x_scaler"]
y_scaler         = data["y_scaler"]

class SpatialTemporalDataset(Dataset):
    def __init__(self, X, Y, station_records):
        self.X = X
        self.Y = Y
        self.station_records = station_records
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.station_records[idx]

def collate_station_records(batch):
    X_b = torch.stack([item[0] for item in batch])
    Y_b = torch.stack([item[1] for item in batch])
    recs = [item[2] for item in batch]
    return X_b, Y_b, recs

# Reproducible DataLoader generator
g = torch.Generator()
g.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    SpatialTemporalDataset(X_train, Y_train, sr_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g,
    collate_fn=collate_station_records
)
val_loader = DataLoader(
    SpatialTemporalDataset(X_val, Y_val, sr_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_station_records
)
test_loader = DataLoader(
    SpatialTemporalDataset(X_test, Y_test, sr_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_station_records
)

# ------------------------------------------------------------------------------
# MODEL INITIALIZATION & VERIFICATION
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {DEVICE}")

# Exact hyper-heuristic parameters from Colab run
FROZEN_K = 0.6430325304733049
FROZEN_TAU = 0.034760647670440264

model = SpatioTemporalModel(
    num_nodes=NUM_STATIONS,
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    sequence_length=SEQUENCE_LENGTH,
    prediction_horizon=PREDICTION_HORIZON,
    decay_constant=GRAPH_DECAY_CONSTANT,
    similarity_threshold=GRAPH_SIMILARITY_THRESHOLD,
    num_heads=GAT_HEADS,
    use_hyper_heuristic=True
).to(DEVICE)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Verified Model Trainable Parameters: {param_count:,}")
assert param_count == 262336, f"Expected 262,336 parameters, got {param_count}"

# ------------------------------------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------------------------------------
class HorizonWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, horizon=PREDICTION_HORIZON, decay=0.80):
        super().__init__()
        self.delta = delta
        weights = torch.tensor([decay ** h for h in range(horizon)], dtype=torch.float32)
        weights = weights / weights.mean()
        self.register_buffer("weights", weights.view(1, 1, horizon))

    def forward(self, pred, target):
        huber = torch.nn.functional.huber_loss(pred, target, delta=self.delta, reduction="none")
        w = self.weights.to(device=pred.device, dtype=pred.dtype)
        return (huber * w).mean()

criterion = HorizonWeightedHuberLoss(delta=1.0, horizon=PREDICTION_HORIZON, decay=0.80)

# Configuration manifest
config_manifest = {
    "random_seed": RANDOM_SEED,
    "lookback_L": SEQUENCE_LENGTH,
    "horizon_H": PREDICTION_HORIZON,
    "num_stations": NUM_STATIONS,
    "input_features": INPUT_DIM,
    "embedding_dim": EMBEDDING_DIM,
    "trainable_parameters": param_count,
    "decay_constant_k": FROZEN_K,
    "similarity_threshold_tau": FROZEN_TAU,
    "loss_function": "HorizonWeightedHuberLoss(delta=1.0, decay=0.80)",
    "optimizer": "AdamW(lr=0.0003, weight_decay=0.01)",
    "lr_scheduler": "CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)",
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "batch_size": BATCH_SIZE,
    "split_sizes": {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test),
        "purge_gap": 191
    }
}
with open(os.path.join(OUTPUT_DIR, "config_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(config_manifest, f, indent=4)
print("Saved configuration manifest to config_manifest.json")

# ------------------------------------------------------------------------------
# METRIC EVALUATION HELPERS
# ------------------------------------------------------------------------------
def inverse_transform_y(scaled_tensor):
    flat = scaled_tensor.detach().cpu().numpy().reshape(-1, 1)
    orig = y_scaler.inverse_transform(flat)
    return orig.reshape(scaled_tensor.shape)

def calc_reg_metrics(preds, targets):
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}

def calc_acc_metrics(preds, targets):
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    err = np.abs(p - t)
    nz = np.abs(t) > 1e-8
    acc = float(np.clip(100.0 * np.mean(1.0 - err[nz] / np.abs(t[nz])), 0.0, 100.0)) if np.any(nz) else 0.0
    acc_10 = float(100.0 * np.mean(err <= 0.10 * np.maximum(np.abs(t), 1e-8)))
    acc_20 = float(100.0 * np.mean(err <= 0.20 * np.maximum(np.abs(t), 1e-8)))
    denom = np.abs(p) + np.abs(t)
    v_s = denom > 1e-8
    smape = float(100.0 * np.mean(2.0 * np.abs(p[v_s] - t[v_s]) / denom[v_s])) if np.any(v_s) else 0.0
    return {"accuracy": acc, "accuracy_10": acc_10, "accuracy_20": acc_20, "smape": smape}

def evaluate_full(current_model, loader):
    current_model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds_orig, all_targs_orig = [], []
    with torch.no_grad():
        for X_b, Y_b, recs in loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            out = current_model(X_b, recs)
            loss = criterion(out, Y_b)
            bs = X_b.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            all_preds_orig.append(inverse_transform_y(out))
            all_targs_orig.append(inverse_transform_y(Y_b))

    preds_orig = np.concatenate(all_preds_orig, axis=0)
    targs_orig = np.concatenate(all_targs_orig, axis=0)
    avg_loss = total_loss / max(total_samples, 1)

    reg = calc_reg_metrics(preds_orig, targs_orig)
    acc = calc_acc_metrics(preds_orig, targs_orig)
    h1_reg = calc_reg_metrics(preds_orig[:, :, 0], targs_orig[:, :, 0])

    res = {
        "val_loss": avg_loss,
        "mae": reg["mae"], "rmse": reg["rmse"], "r2": reg["r2"],
        "h1_mae": h1_reg["mae"], "h1_r2": h1_reg["r2"],
        "accuracy": acc["accuracy"], "accuracy_10": acc["accuracy_10"],
        "accuracy_20": acc["accuracy_20"], "smape": acc["smape"]
    }
    return res, preds_orig, targs_orig

def train_epoch(current_model, opt):
    current_model.train()
    total_loss, total_samples = 0.0, 0
    grad_norms = []
    for X_b, Y_b, recs in train_loader:
        X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
        opt.zero_grad()
        out = current_model(X_b, recs)
        loss = criterion(out, Y_b)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(current_model.parameters(), GRADIENT_CLIP_VALUE)
        grad_norms.append(float(gn))
        opt.step()
        bs = X_b.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / max(total_samples, 1), float(np.mean(grad_norms))

# ------------------------------------------------------------------------------
# STEP 3: TRAINING (WARM-UP -> LOCKED PARAMETERS -> FINAL TRAINING)
# ------------------------------------------------------------------------------
RESUME_CHECKPOINT = os.path.join(OUTPUT_DIR, "checkpoint_best.pt")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "training_history.json")

can_resume = False
if os.path.exists(RESUME_CHECKPOINT) and os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            existing_history = json.load(f)
        if len(existing_history) >= 1:
            can_resume = True
    except Exception:
        can_resume = False

if can_resume:
    print("\n" + "=" * 80)
    print(f"RESUMING EXPERIMENT: Found existing progress ({len(existing_history)} epochs recorded)")
    print("=" * 80)
    # Load model state from best checkpoint
    best_ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    if hasattr(model, "set_graph_parameters"):
        model.set_graph_parameters(decay_constant=FROZEN_K, similarity_threshold=FROZEN_TAU)
    
    final_opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        final_opt, T_0=15, T_mult=2, eta_min=1e-6
    )
    # Fast-forward scheduler
    for _ in range(len(existing_history)):
        scheduler.step()
    
    history = existing_history
    best_epoch = best_ckpt["epoch"]
    best_val_r2 = best_ckpt["best_val_r2"]
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    start_epoch = len(history) + 1
    epochs_no_improve = len(history) - best_epoch
    print(f"Resuming from Epoch {start_epoch} (Current best R2 = {best_val_r2:.4f} at Epoch {best_epoch})")
else:
    print("\n" + "=" * 80)
    print("STAGE 1: WARM-UP (5 EPOCHS)")
    print("=" * 80)
    warmup_opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for ep in range(1, 6):
        t0 = time.time()
        tr_loss, gn = train_epoch(model, warmup_opt)
        val_res, _, _ = evaluate_full(model, val_loader)
        dt = time.time() - t0
        print(f"Warm-up Epoch [{ep}/5] ({dt:.1f}s) | Train Loss: {tr_loss:.6f} | Val Loss: {val_res['val_loss']:.6f} | Val MAE: {val_res['mae']:.2f} | Val RMSE: {val_res['rmse']:.2f} | Val R2: {val_res['r2']:.4f}")

    # Save state after warm-up
    warmup_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\n" + "=" * 80)
    print("STAGE 2: LOCKING FROZEN GRAPH PARAMETERS")
    print(f"Active k: {FROZEN_K:.6f}")
    print(f"Active threshold: {FROZEN_TAU:.6f}")
    print("=" * 80)

    # Apply frozen k and tau
    model.load_state_dict(warmup_state)
    if hasattr(model, "set_graph_parameters"):
        model.set_graph_parameters(decay_constant=FROZEN_K, similarity_threshold=FROZEN_TAU)
    elif hasattr(model, "decay_constant"):
        model.decay_constant.data.fill_(FROZEN_K)
        model.similarity_threshold.data.fill_(FROZEN_TAU)

    # Create final optimizer & scheduler exactly as in Colab run
    final_opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        final_opt, T_0=15, T_mult=2, eta_min=1e-6
    )

    print("\n" + "=" * 80)
    print("STAGE 3: FINAL TRAINING (MAX 100 EPOCHS, PATIENCE 25)")
    print("=" * 80)

    best_val_r2 = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    start_epoch = 1

for ep in range(start_epoch, 101):
    t0 = time.time()
    tr_loss, gn = train_epoch(model, final_opt)
    val_res, _, _ = evaluate_full(model, val_loader)
    scheduler.step()
    cur_lr = final_opt.param_groups[0]["lr"]
    dt = time.time() - t0

    cur_r2 = val_res["r2"]
    is_best = cur_r2 > best_val_r2

    epoch_record = {
        "epoch": ep,
        "train_loss": tr_loss,
        "val_loss": val_res["val_loss"],
        "gradient_norm": gn,
        "learning_rate": cur_lr,
        "epoch_time_seconds": dt,
        "val_metrics": val_res,
        "is_best": is_best
    }
    history.append(epoch_record)

    # Save running history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    status_flag = ""
    if is_best:
        best_val_r2 = cur_r2
        best_epoch = ep
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
        status_flag = " [*] [NEW BEST]"
        # Save best checkpoint
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "best_val_r2": best_val_r2,
            "decay_constant": FROZEN_K,
            "similarity_threshold": FROZEN_TAU,
            "trainable_parameters": param_count,
            "val_metrics": val_res
        }, os.path.join(OUTPUT_DIR, "checkpoint_best.pt"))
    else:
        epochs_no_improve += 1
        status_flag = f" [No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}]"

    print(f"Epoch [{ep:04d}/100] ({dt:.1f}s) | Train: {tr_loss:.6f} | Val: {val_res['val_loss']:.6f} | h+1 MAE: {val_res['h1_mae']:.2f} | h+1 R2: {val_res['h1_r2']:.4f} | 24h MAE: {val_res['mae']:.2f} | RMSE: {val_res['rmse']:.2f} | R2: {val_res['r2']:.4f} | Acc+/-20%: {val_res['accuracy_20']:.2f}% | LR: {cur_lr:.6f}{status_flag}")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEARLY STOPPING triggered at Epoch {ep}. Best validation epoch: {best_epoch} (R2 = {best_val_r2:.6f})")
        break

# Save final checkpoint
torch.save({
    "final_epoch": ep,
    "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    "final_val_r2": cur_r2,
    "best_epoch": best_epoch,
    "best_val_r2": best_val_r2,
    "decay_constant": FROZEN_K,
    "similarity_threshold": FROZEN_TAU
}, os.path.join(OUTPUT_DIR, "checkpoint_final.pt"))
print("Saved checkpoint_final.pt")

# ------------------------------------------------------------------------------
# STEP 7: TEST & FINAL DETAILED EVALUATION
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EVALUATING BEST CHECKPOINT ON VALIDATION AND TEST SPLITS")
print("=" * 80)

model.load_state_dict(best_state)

# 1. Validation split detailed evaluation
val_metrics, val_preds_orig, val_targs_orig = evaluate_full(model, val_loader)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "predictions_val.npz"),
    predictions=val_preds_orig,
    targets=val_targs_orig
)
print("Saved predictions_val.npz")

# 2. Test split detailed evaluation
test_metrics, test_preds_orig, test_targs_orig = evaluate_full(model, test_loader)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "predictions_test.npz"),
    predictions=test_preds_orig,
    targets=test_targs_orig
)
print("Saved predictions_test.npz")

# 3. Horizon-by-Horizon metrics (H=1..24) for both splits
horizon_breakdown_val = {}
horizon_breakdown_test = {}
for h in range(PREDICTION_HORIZON):
    horizon_breakdown_val[f"H_{h+1:02d}"] = calc_reg_metrics(val_preds_orig[:, :, h], val_targs_orig[:, :, h])
    horizon_breakdown_test[f"H_{h+1:02d}"] = calc_reg_metrics(test_preds_orig[:, :, h], test_targs_orig[:, :, h])

# 4. Station-by-Station metrics for both splits
station_names = ["Anand Vihar", "Connaught Place", "Dwarka", "IGI Airport", "Okhla Phase III", "Rohini"]
station_breakdown_val = {}
station_breakdown_test = {}
for s in range(NUM_STATIONS):
    s_name = station_names[s]
    station_breakdown_val[s_name] = {
        "pooled": calc_reg_metrics(val_preds_orig[:, s, :], val_targs_orig[:, s, :]),
        "H_01": calc_reg_metrics(val_preds_orig[:, s, 0], val_targs_orig[:, s, 0]),
        "H_06": calc_reg_metrics(val_preds_orig[:, s, 5], val_targs_orig[:, s, 5]),
        "H_12": calc_reg_metrics(val_preds_orig[:, s, 11], val_targs_orig[:, s, 11]),
        "H_18": calc_reg_metrics(val_preds_orig[:, s, 17], val_targs_orig[:, s, 17]),
        "H_24": calc_reg_metrics(val_preds_orig[:, s, 23], val_targs_orig[:, s, 23])
    }
    station_breakdown_test[s_name] = {
        "pooled": calc_reg_metrics(test_preds_orig[:, s, :], test_targs_orig[:, s, :]),
        "H_01": calc_reg_metrics(test_preds_orig[:, s, 0], test_targs_orig[:, s, 0]),
        "H_06": calc_reg_metrics(test_preds_orig[:, s, 5], test_targs_orig[:, s, 5]),
        "H_12": calc_reg_metrics(test_preds_orig[:, s, 11], test_targs_orig[:, s, 11]),
        "H_18": calc_reg_metrics(test_preds_orig[:, s, 17], test_targs_orig[:, s, 17]),
        "H_24": calc_reg_metrics(test_preds_orig[:, s, 23], test_targs_orig[:, s, 23])
    }

# Historical reference metrics from Colab log
historical_colab = {
    "best_epoch": 7,
    "early_stopping_epoch": 32,
    "h1_mae": 3.96,
    "h1_r2": 0.9624,
    "pooled_mae": 14.99,
    "pooled_rmse": 20.69,
    "pooled_r2": 0.5384,
    "accuracy_20": 56.77,
    "k": 0.643033,
    "tau": 0.034761
}

# Compile full metrics summary
summary = {
    "experiment_id": "baseline_delhi_reproduced_2026-09-30",
    "historical_reference_id": "baseline_delhi_historical_2026-09-16",
    "best_epoch": best_epoch,
    "early_stopping_epoch": ep,
    "validation_summary": val_metrics,
    "test_summary": test_metrics,
    "validation_horizon_breakdown": horizon_breakdown_val,
    "test_horizon_breakdown": horizon_breakdown_test,
    "validation_station_breakdown": station_breakdown_val,
    "test_station_breakdown": station_breakdown_test,
    "comparison_with_historical_colab": {
        "historical": historical_colab,
        "new_run": {
            "best_epoch": best_epoch,
            "early_stopping_epoch": ep,
            "h1_mae": val_metrics["h1_mae"],
            "h1_r2": val_metrics["h1_r2"],
            "pooled_mae": val_metrics["mae"],
            "pooled_rmse": val_metrics["rmse"],
            "pooled_r2": val_metrics["r2"],
            "accuracy_20": val_metrics["accuracy_20"],
            "k": FROZEN_K,
            "tau": FROZEN_TAU
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)
print("Saved metrics_summary.json")

# ------------------------------------------------------------------------------
# SHA-256 MANIFEST
# ------------------------------------------------------------------------------
def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest = {}
for fname in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath) and fname != "sha256_manifest.json":
        manifest[fname] = {
            "size_bytes": os.path.getsize(fpath),
            "sha256": get_sha256(fpath)
        }

with open(os.path.join(OUTPUT_DIR, "sha256_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)
print("Saved sha256_manifest.json")

print("\n" + "=" * 80)
print("BASELINE REPRODUCTION COMPLETE")
print("=" * 80)

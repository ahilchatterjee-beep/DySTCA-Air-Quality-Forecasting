# ==========================================================
# Training/train_autoregressive.py
#
# Closed-Loop Autoregressive Feed-Forward Training & Evaluation
# with 168-Hour (1-Week) Sliding Temporal Context Window
#
# Architecture:
#   - Core Predictor: SpatioTemporalModel with prediction_horizon = 1
#   - At each step h in 1..24:
#       1. Model takes the current 168-hour history [t-167 ... t]
#       2. Predicts step t+1
#       3. Feeds predicted PM2.5 forward into feature matrix
#       4. Slides the 168-hour window forward by 1 hour
#   - Scheduled Sampling:
#       Gradually transitions from teacher forcing to 100% closed-loop
#       autoregressive feedback, teaching the model self-correction.
# ==========================================================

import os
import sys
import time
import copy
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    EMBEDDING_DIM,
    GAT_HEADS,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    NUM_STATIONS,
    INPUT_DIM
)

from preprocessing.pre import prepare_model_data
from Model.new_model import SpatioTemporalModel

# Set reproducibility seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("AUTOREGRESSIVE FEED-FORWARD SPATIO-TEMPORAL FORECASTING")
print("=" * 70)
print(f"Device             : {DEVICE}")
print(f"History Context    : {SEQUENCE_LENGTH} hours (1 Week)")
print(f"Forecast Horizon   : {PREDICTION_HORIZON} hours (Autoregressive Rollout)")
print(f"Stations           : {NUM_STATIONS}")
print(f"Batch Size         : {BATCH_SIZE}")
print(f"Learning Rate      : {LEARNING_RATE}")
print("=" * 70)

# ==========================================================
# 1. LOAD DATASET
# ==========================================================
data = prepare_model_data()

X_train = data["X_train"]
Y_train = data["Y_train"]
station_records_train = data["station_records_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]
station_records_val = data["station_records_val"]

X_test = data["X_test"]
Y_test = data["Y_test"]
station_records_test = data["station_records_test"]

target_scaler = data["target_scaler"]


class StationRecordDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, records):
        self.X = X
        self.Y = Y
        self.records = records

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.records[idx]


def collate_fn(batch):
    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    records = [item[2] for item in batch]
    return torch.stack(X_list, dim=0), torch.stack(Y_list, dim=0), records


train_loader = DataLoader(
    StationRecordDataset(X_train, Y_train, station_records_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False
)

val_loader = DataLoader(
    StationRecordDataset(X_val, Y_val, station_records_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False
)

test_loader = DataLoader(
    StationRecordDataset(X_test, Y_test, station_records_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False
)

# ==========================================================
# 2. AUTOREGRESSIVE MODEL (prediction_horizon = 1)
# ==========================================================
model = SpatioTemporalModel(
    num_nodes=NUM_STATIONS,
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    sequence_length=SEQUENCE_LENGTH,
    prediction_horizon=1,
    num_heads=4
).to(DEVICE)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Horizon-Weighted Loss with gentle discount
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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=MIN_LEARNING_RATE
)

# ==========================================================
# 3. AUTOREGRESSIVE ROLLOUT FORWARD PASS
# ==========================================================
def autoregressive_rollout(current_model, X_batch, station_records, teacher_forcing_ratio=0.0, true_targets=None):
    """
    Executes a 24-step autoregressive rollout:
      - Takes current 168-hour history [B, 168, N, F]
      - Loops h = 1..24:
          Predicts step h [B, N, 1]
          Feeds predicted PM2.5 forward into sliding window
          Drops oldest hour, appends new hour to maintain 168h context
    """
    B, T, N, F = X_batch.shape
    H = PREDICTION_HORIZON
    curr_X = X_batch.clone()
    predictions = []

    for step in range(H):
        # 1-step forward pass
        pred_step = current_model(curr_X, station_records)  # [B, N, 1]
        predictions.append(pred_step)

        # Determine what to feed forward (Scheduled Sampling during training)
        if current_model.training and teacher_forcing_ratio > 0.0 and true_targets is not None:
            if random.random() < teacher_forcing_ratio:
                feed_val = true_targets[:, :, step:step+1]
            else:
                feed_val = pred_step
        else:
            feed_val = pred_step  # 100% closed loop in evaluation

        # Construct new step features at position t+h
        # Replicates last known meteorological state with updated predicted PM2.5
        new_step_features = curr_X[:, -1:, :, :].clone()  # [B, 1, N, F]
        new_step_features[:, 0, :, 0] = feed_val.squeeze(-1)

        # Slide 168-hour context window forward
        curr_X = torch.cat([curr_X[:, 1:, :, :], new_step_features], dim=1)

    # [B, N, 24]
    return torch.cat(predictions, dim=-1)


def inverse_transform_y(scaled_tensor):
    flat = scaled_tensor.detach().cpu().numpy().reshape(-1, 1)
    orig = target_scaler.inverse_transform(flat)
    return orig.reshape(scaled_tensor.shape)


def calculate_metrics(preds_orig, targets_orig):
    err = preds_orig - targets_orig
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ==========================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==========================================================
def train_one_epoch(current_model, opt, teacher_forcing_ratio):
    current_model.train()
    total_loss = 0.0
    total_batches = 0

    for X_batch, Y_batch, records in train_loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        Y_batch = Y_batch.to(DEVICE, non_blocking=True)

        opt.zero_grad()
        rollout_out = autoregressive_rollout(
            current_model, X_batch, records,
            teacher_forcing_ratio=teacher_forcing_ratio,
            true_targets=Y_batch
        )

        loss = criterion(rollout_out, Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(current_model, loader):
    current_model.eval()
    total_loss = 0.0
    total_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch, records in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_batch = Y_batch.to(DEVICE, non_blocking=True)

            # 100% Closed-Loop Rollout: model receives NO future ground truth
            rollout_out = autoregressive_rollout(
                current_model, X_batch, records,
                teacher_forcing_ratio=0.0,
                true_targets=None
            )

            loss = criterion(rollout_out, Y_batch)
            total_loss += loss.item()
            total_batches += 1

            all_preds.append(inverse_transform_y(rollout_out))
            all_targets.append(inverse_transform_y(Y_batch))

    preds_orig = np.concatenate(all_preds, axis=0)
    targets_orig = np.concatenate(all_targets, axis=0)

    # Pooled 24h metrics
    overall = calculate_metrics(preds_orig, targets_orig)

    # 1h metrics
    h1 = calculate_metrics(preds_orig[:, :, 0], targets_orig[:, :, 0])
    overall["h1_mae"] = h1["mae"]
    overall["h1_r2"] = h1["r2"]

    return total_loss / max(total_batches, 1), overall, preds_orig, targets_orig


# ==========================================================
# 5. MAIN TRAINING LOOP
# ==========================================================
best_val_r2 = float("-inf")
best_epoch = 0
best_state = None
history = []
epochs_without_improvement = 0

print("")
print("=" * 70)
print("STARTING CLOSED-LOOP AUTOREGRESSIVE TRAINING")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    # Scheduled sampling: starts with 0.5 teacher forcing, decays to 0.0 by epoch 10
    tf_ratio = max(0.0, 0.5 * (1.0 - (epoch - 1) / 10.0))

    train_loss = train_one_epoch(model, optimizer, teacher_forcing_ratio=tf_ratio)
    val_loss, val_metrics, _, _ = evaluate(model, val_loader)

    scheduler.step()
    curr_lr = optimizer.param_groups[0]["lr"]

    improved = False
    if val_metrics["r2"] > best_val_r2 + EARLY_STOPPING_MIN_DELTA:
        best_val_r2 = val_metrics["r2"]
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "best_autoregressive_model.pt")
        epochs_without_improvement = 0
        improved = True
    else:
        epochs_without_improvement += 1

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mae": val_metrics["mae"],
        "rmse": val_metrics["rmse"],
        "r2": val_metrics["r2"],
        "h1_r2": val_metrics["h1_r2"],
        "h1_mae": val_metrics["h1_mae"],
        "tf_ratio": tf_ratio
    })

    elapsed = time.time() - t0
    marker = " ⭐ [BEST]" if improved else ""
    print(
        f"Epoch [{epoch:03d}/{MAX_EPOCHS:03d}] "
        f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
        f"h+1 R²: {val_metrics['h1_r2']:.4f} | 24h R²: {val_metrics['r2']:.4f} | "
        f"24h MAE: {val_metrics['mae']:.2f} µg/m³ | TF: {tf_ratio:.2f} | {elapsed:.1f}s{marker}"
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
        break

# Save training history
with open("autoregressive_training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# ==========================================================
# 6. FINAL TEST SET EVALUATION
# ==========================================================
print("")
print("=" * 70)
print("FINAL TEST EVALUATION (CLOSED-LOOP AUTOREGRESSIVE ROLLOUT)")
print("=" * 70)

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best checkpoint from Epoch {best_epoch}")

test_loss, test_metrics, test_preds, test_targets = evaluate(model, test_loader)

print(f"Test Loss (scaled Huber) : {test_loss:.6f}")
print(f"Test Overall 24-Hr MAE   : {test_metrics['mae']:.2f} µg/m³")
print(f"Test Overall 24-Hr RMSE  : {test_metrics['rmse']:.2f} µg/m³")
print(f"Test Overall 24-Hr R²    : {test_metrics['r2']:.4f}")

print("")
print("=" * 70)
print("HORIZON-BY-HORIZON BREAKDOWN (CLOSED-LOOP AUTOREGRESSIVE ROLLOUT)")
print("=" * 70)
print(f"{'Horizon':<10} | {'MAE (µg/m³)':<15} | {'RMSE (µg/m³)':<15} | {'R²':<10}")
print("-" * 58)

horizons_to_show = [0, 1, 2, 5, 11, 17, 23]
for h in horizons_to_show:
    p_h = test_preds[:, :, h]
    t_h = test_targets[:, :, h]
    m_h = calculate_metrics(p_h, t_h)
    print(f"h+{h+1:02d} hr     | {m_h['mae']:<15.4f} | {m_h['rmse']:<15.4f} | {m_h['r2']:<10.4f}")

h12_preds = test_preds[:, :, :12]
h12_targets = test_targets[:, :, :12]
m_12 = calculate_metrics(h12_preds, h12_targets)
print("-" * 58)
print(f"12-Hr Avg  | {m_12['mae']:<15.4f} | {m_12['rmse']:<15.4f} | {m_12['r2']:<10.4f}")
print(f"24-Hr Avg  | {test_metrics['mae']:<15.4f} | {test_metrics['rmse']:<15.4f} | {test_metrics['r2']:<10.4f}")
print("=" * 70)
print("AUTOREGRESSIVE FEED-FORWARD TRAINING PIPELINE COMPLETE.")

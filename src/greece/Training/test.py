# ==========================================================
# test.py
#
# FINAL TESTING FOR THE CURRENT MODEL
#
# Current preprocessing:
#   Input  : [samples, 24, 6, 8]
#   Target : [samples, 6, 1]
#
# Metrics:
#   - R²
#   - MAE
#   - RMSE
#
# IMPORTANT:
# This evaluates ONLY the untouched TEST set.
# ==========================================================


import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from preprocessing.pre import prepare_model_data
from Model.model import SpatioTemporalModel


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32

CHECKPOINT_PATH = "best_model.pt"

RESULTS_DIR = "test_results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# DEVICE INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("FINAL TESTING")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================================
# DATASET
# ==========================================================

class TestDataset(Dataset):

    def __init__(
        self,
        X,
        Y,
        station_records
    ):

        self.X = X
        self.Y = Y
        self.station_records = station_records

        if len(self.X) != len(self.Y):

            raise ValueError(
                "X and Y have different numbers "
                "of samples."
            )

        if (
            len(self.X)
            != len(self.station_records)
        ):

            raise ValueError(
                "X and station_records have "
                "different numbers of samples."
            )


    def __len__(self):

        return len(self.X)


    def __getitem__(self, index):

        return (
            self.X[index],
            self.Y[index],
            self.station_records[index]
        )


# ==========================================================
# COLLATE FUNCTION
# ==========================================================

def collate_fn(batch):

    X = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    Y = torch.stack(
        [
            item[1]
            for item in batch
        ]
    )

    station_records = [
        item[2]
        for item in batch
    ]

    return (
        X,
        Y,
        station_records
    )


# ==========================================================
# LOAD PREPROCESSED DATA
# ==========================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

data = prepare_model_data()


X_test = data["X_test"]
Y_test = data["Y_test"]

station_records_test = (
    data["station_records_test"]
)

target_scaler = data["target_scaler"]


# ==========================================================
# DATA INFORMATION
# ==========================================================

NUM_TEST_SAMPLES = len(X_test)

SEQUENCE_LENGTH = X_test.shape[1]

NUM_STATIONS = X_test.shape[2]

INPUT_DIM = X_test.shape[3]

TARGET_DIM = Y_test.shape[-1]


print("\nTest samples      :", NUM_TEST_SAMPLES)
print("Sequence length   :", SEQUENCE_LENGTH)
print("Number of stations:", NUM_STATIONS)
print("Input features    :", INPUT_DIM)
print("Target dimension  :", TARGET_DIM)

print(
    "X test shape      :",
    tuple(X_test.shape)
)

print(
    "Y test shape      :",
    tuple(Y_test.shape)
)


# ==========================================================
# TEST DATALOADER
# ==========================================================

test_dataset = TestDataset(
    X_test,
    Y_test,
    station_records_test
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)


# ==========================================================
# CREATE MODEL
# ==========================================================

print("\n" + "=" * 70)
print("CREATING MODEL")
print("=" * 70)


model = SpatioTemporalModel(
    num_nodes=NUM_STATIONS,
    input_dim=INPUT_DIM,
    embedding_dim=64,
    sequence_length=SEQUENCE_LENGTH,

    # These are fallback values.
    # The checkpoint will replace them
    # with the values selected during training.
    decay_constant=0.5,
    similarity_threshold=0.2,

    num_heads=8,

    use_hyper_heuristic=False
)


model = model.to(DEVICE)


print(
    "Total parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)


# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print("\n" + "=" * 70)
print("LOADING BEST CHECKPOINT")
print("=" * 70)


if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        "\nBest model checkpoint was not found:\n"
        f"{CHECKPOINT_PATH}\n\n"
        "Make sure train_updated.py saved "
        "best_model.pt during training."
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


# ----------------------------------------------------------
# CHECKPOINT FORMAT
# ----------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Checkpoint type:",
            "full training checkpoint"
        )

        if "best_epoch" in checkpoint:

            print(
                "Best epoch:",
                checkpoint["best_epoch"]
            )

        if "best_val_r2" in checkpoint:

            print(
                "Best validation R²:",
                checkpoint["best_val_r2"]
            )

        if (
            "decay_constant"
            in checkpoint
        ):

            print(
                "Decay constant:",
                checkpoint[
                    "decay_constant"
                ]
            )

        if (
            "similarity_threshold"
            in checkpoint
        ):

            print(
                "Similarity threshold:",
                checkpoint[
                    "similarity_threshold"
                ]
            )

        # --------------------------------------------------
        # Restore graph parameters
        # --------------------------------------------------

        if (
            "decay_constant"
            in checkpoint
            and
            "similarity_threshold"
            in checkpoint
        ):

            model.set_graph_parameters(
                decay_constant=float(
                    checkpoint[
                        "decay_constant"
                    ]
                ),

                similarity_threshold=float(
                    checkpoint[
                        "similarity_threshold"
                    ]
                )
            )

    else:

        # Raw state_dict
        model.load_state_dict(
            checkpoint
        )

        print(
            "Checkpoint type:",
            "raw model state_dict"
        )

else:

    raise ValueError(
        "Unsupported checkpoint format."
    )


print(
    "\nCheckpoint loaded successfully."
)


# ==========================================================
# EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("RUNNING FINAL TEST")
print("=" * 70)


model.eval()


all_predictions = []

all_targets = []


with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        X_batch = batch[0]

        Y_batch = batch[1]

        station_records = batch[2]


        # --------------------------------------------------
        # Move tensors to device
        # --------------------------------------------------

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        Y_batch = Y_batch.to(
            DEVICE,
            non_blocking=True
        )


        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        predictions = model(
            X_batch,
            station_records
        )


        # --------------------------------------------------
        # Store predictions and targets
        # --------------------------------------------------

        all_predictions.append(
            predictions.detach().cpu()
        )

        all_targets.append(
            Y_batch.detach().cpu()
        )


        if (
            (batch_index + 1) % 20 == 0
            or
            (batch_index + 1)
            == len(test_loader)
        ):

            print(
                f"Processed test batches: "
                f"{batch_index + 1}/"
                f"{len(test_loader)}"
            )


# ==========================================================
# COMBINE OUTPUTS
# ==========================================================

predictions_scaled = torch.cat(
    all_predictions,
    dim=0
)

targets_scaled = torch.cat(
    all_targets,
    dim=0
)


# ==========================================================
# CONVERT TO NUMPY
# ==========================================================

predictions_scaled = (
    predictions_scaled
    .numpy()
)

targets_scaled = (
    targets_scaled
    .numpy()
)


# ==========================================================
# INVERSE SCALING
#
# The preprocessing fitted the target scaler on:
#
#   Y_train.reshape(-1, 1)
#
# Therefore we do exactly the inverse operation here.
# ==========================================================

prediction_shape = (
    predictions_scaled.shape
)

target_shape = (
    targets_scaled.shape
)


predictions_original = (
    target_scaler
    .inverse_transform(
        predictions_scaled.reshape(
            -1,
            1
        )
    )
    .reshape(
        prediction_shape
    )
)


targets_original = (
    target_scaler
    .inverse_transform(
        targets_scaled.reshape(
            -1,
            1
        )
    )
    .reshape(
        target_shape
    )
)


# ==========================================================
# FLATTEN
#
# We have:
#
#   samples × stations × 1
#
# For overall metrics we treat every
# station prediction as an observation.
# ==========================================================

# ==========================================================
# SCALED METRIC VALUES
# ==========================================================

y_true = targets_scaled.reshape(-1)
y_pred = predictions_scaled.reshape(-1)

valid_scaled_mask = (
    np.isfinite(y_true)
    &
    np.isfinite(y_pred)
)

y_true = y_true[valid_scaled_mask]
y_pred= y_pred[valid_scaled_mask]


# ==========================================================
# REMOVE INVALID VALUES
# ==========================================================

valid_mask = (
    np.isfinite(y_true)
    &
    np.isfinite(y_pred)
)


y_true = y_true[
    valid_mask
]

y_pred = y_pred[
    valid_mask
]


if len(y_true) == 0:

    raise ValueError(
        "No valid predictions remained "
        "after removing NaN/Inf values."
    )

# ==========================================================
# PREDICTION ACCURACY
#
# Measures how close each prediction is to the
# actual PM2.5 value.
#
# Accuracy = (1 - |Actual - Predicted| / |Actual|) * 100
#
# Very small actual values are excluded because
# percentage error becomes unstable near zero.
# ==========================================================

ACCURACY_EPSILON = 1e-8

accuracy_valid_mask = (
    np.abs(y_true) > ACCURACY_EPSILON
)

accuracy_values = (
    1.0
    -
    (
        np.abs(
            y_true[accuracy_valid_mask]
            -
            y_pred[accuracy_valid_mask]
        )
        /
        np.abs(
            y_true[accuracy_valid_mask]
        )
    )
)

# Prevent individual predictions from producing
# negative accuracy percentages.
accuracy_values = np.clip(
    accuracy_values,
    0.0,
    1.0
)

ACCURACY = (
    np.mean(accuracy_values)
    * 100.0
)

print(
    f"Prediction Accuracy : "
    f"{ACCURACY:.2f}%"
)


# ==========================================================
# R²
# ==========================================================

ss_res = np.sum(
    (y_true - y_pred) ** 2
)

ss_tot = np.sum(
    (y_true - np.mean(y_true)) ** 2
)


if ss_tot == 0:

    R2 = float("nan")

else:

    R2 = (
        1.0
        -
        (
            ss_res
            /
            ss_tot
        )
    )


# ==========================================================
# MAE
# ==========================================================

MAE = np.mean(
    np.abs(
        y_true - y_pred
    )
)


# ==========================================================
# RMSE
# ==========================================================

RMSE = np.sqrt(
    np.mean(
        (
            y_true - y_pred
        ) ** 2
    )
)


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {ACCURACY:.2f}%"
)

print(
    f"R²       : {R2:.6f}"
)

print(
    f"MAE      : {MAE:.6f} µg/m³"
)

print(
    f"RMSE     : {RMSE:.6f} µg/m³"
)

print(
    f"\nEvaluated observations: "
    f"{len(y_true)}"
)

print("=" * 70)


# ==========================================================
# PER-STATION RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("PER-STATION TEST RESULTS")
print("=" * 70)


station_predictions = (
    predictions_original[:, :, 0]
)

station_targets = (
    targets_original[:, :, 0]
)


station_results = []


stations = data["stations"]


for station_index, station_name in enumerate(
    stations
):

    station_true = (
        station_targets[
            :,
            station_index
        ]
    )

    station_pred = (
        station_predictions[
            :,
            station_index
        ]
    )


    station_ss_res = np.sum(
        (
            station_true
            -
            station_pred
        ) ** 2
    )


    station_ss_tot = np.sum(
        (
            station_true
            -
            np.mean(station_true)
        ) ** 2
    )


    if station_ss_tot == 0:

        station_r2 = float("nan")

    else:

        station_r2 = (
            1.0
            -
            station_ss_res
            /
            station_ss_tot
        )


    station_mae = np.mean(
        np.abs(
            station_true
            -
            station_pred
        )
    )


    station_rmse = np.sqrt(
        np.mean(
            (
                station_true
                -
                station_pred
            ) ** 2
        )
    )


    station_results.append(
        {
            "station": station_name,
            "r2": float(station_r2),
            "mae": float(station_mae),
            "rmse": float(station_rmse)
        }
    )


    print(
        f"{station_name:<20} "
        f"R²={station_r2:.6f}  "
        f"MAE={station_mae:.4f}  "
        f"RMSE={station_rmse:.4f}"
    )


# ==========================================================
# SAVE NUMERICAL RESULTS
# ==========================================================

results = {

    "test_samples":
        int(NUM_TEST_SAMPLES),

    "stations":
        int(NUM_STATIONS),

    "sequence_length":
        int(SEQUENCE_LENGTH),

    "prediction_horizon":
        1,

    "r2":
        float(R2),

    "mae":
        float(MAE),

    "rmse":
        float(RMSE),
        
    "accuracy":
        float(ACCURACY),

    "per_station":
        station_results
}


results_path = os.path.join(
    RESULTS_DIR,
    "test_metrics.json"
)


with open(
    results_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==========================================================
# SAVE RAW PREDICTIONS
# ==========================================================

predictions_path = os.path.join(
    RESULTS_DIR,
    "test_predictions.npz"
)


np.savez(
    predictions_path,

    predictions=
        predictions_original,

    targets=
        targets_original
)


# ==========================================================
# SAVE SIMPLE CSV
# ==========================================================

rows = []


for sample_index in range(
    predictions_original.shape[0]
):

    for station_index in range(
        predictions_original.shape[1]
    ):

        rows.append(
            {
                "sample":
                    sample_index,

                "station":
                    stations[
                        station_index
                    ],

                "actual_pm25":
                    float(
                        targets_original[
                            sample_index,
                            station_index,
                            0
                        ]
                    ),

                "predicted_pm25":
                    float(
                        predictions_original[
                            sample_index,
                            station_index,
                            0
                        ]
                    )
            }
        )


import pandas as pd


results_df = pd.DataFrame(
    rows
)


csv_path = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)


results_df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)

print(
    "Metrics saved to:",
    results_path
)

print(
    "Predictions saved to:",
    predictions_path
)

print(
    "CSV saved to:",
    csv_path
)

print("=" * 70)
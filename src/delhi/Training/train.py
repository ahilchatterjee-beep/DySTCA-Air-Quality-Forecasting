# ==========================================================
# train.py
# SPATIO-TEMPORAL AIR QUALITY TRAINING
# WITH HYPER-HEURISTIC GRAPH OPTIMIZATION
#
# HH:
#   - searches GLOBAL k and threshold
#   - evaluates candidates using validation R²
#   - candidate parameters are constant across all stations
#   - candidate parameters are constant across all timesteps
#
# FINAL TRAINING:
#   - uses ONLY the best HH parameters
#   - HH does NOT modify parameters during final epochs
# ==========================================================

import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from preprocessing.pre_updated import prepare_model_data
from Model.model import SpatioTemporalModel


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32

# ----------------------------------------------------------
# FINAL TRAINING
# ----------------------------------------------------------

EPOCHS = 10

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


# ==========================================================
# HYPER-HEURISTIC CONFIGURATION
# ==========================================================

USE_HYPER_HEURISTIC = True

DECAY_RANGE = (
    0.01,
    1.0
)

SIMILARITY_THRESHOLD_RANGE = (
    0.01,
    1.0
)

NUM_CANDIDATES = 10

EXPLORATION_PROBABILITY = 0.30

EXPLOITATION_FRACTION = 0.15


# ----------------------------------------------------------
# IMPORTANT:
#
# HH candidate training is deliberately SHORT.
#
# We do NOT run 2500 epochs for every candidate.
# ----------------------------------------------------------

HH_CANDIDATE_EPOCHS = 5


# ----------------------------------------------------------
# Warm-up
#
# The neural network first learns a little before the HH
# starts comparing graph parameters.
# ----------------------------------------------------------

HYPER_HEURISTIC_WARMUP_EPOCHS = 5


# ==========================================================
# FINAL TRAINING CONTROLS
# ==========================================================

EARLY_STOPPING_PATIENCE = 50

EARLY_STOPPING_MIN_DELTA = 1e-5

SCHEDULER_PATIENCE = 40

SCHEDULER_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-6


# ==========================================================
# GRADIENT CLIPPING
# ==========================================================

GRADIENT_CLIPPING = True

MAX_GRAD_NORM = 1.0


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

HISTORY_FILE = (
    "training_history.json"
)


# ==========================================================
# RANDOM SEEDS
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# DEVICE
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "DEVICE INFORMATION"
)

print(
    "=" * 70
)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "CUDA unavailable. Using CPU."
    )

print(
    "=" * 70
)


# ==========================================================
# LOAD DATA
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "PREPARING MODEL DATA"
)

print(
    "=" * 70
)

data = prepare_model_data()


X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

X_test = data["X_test"]
Y_test = data["Y_test"]


feature_scaler = data["x_scaler"]

target_scaler = data["y_scaler"]


station_records_train = (
    data["station_records_train"]
)

station_records_val = (
    data["station_records_val"]
)

station_records_test = (
    data["station_records_test"]
)


# ==========================================================
# DATASET
# ==========================================================

class SpatialTemporalDataset(
    Dataset
):

    def __init__(
        self,
        X,
        Y,
        station_records
    ):

        self.X = X
        self.Y = Y

        self.station_records = (
            station_records
        )

        if len(self.X) != len(self.Y):

            raise ValueError(
                "X and Y sample counts "
                "do not match."
            )

        if (
            len(self.X)
            !=
            len(self.station_records)
        ):

            raise ValueError(
                "X and station-record "
                "counts do not match."
            )


    def __len__(self):

        return len(self.X)


    def __getitem__(
        self,
        index
    ):

        return (

            self.X[index],

            self.Y[index],

            self.station_records[index]
        )


# ==========================================================
# COLLATE
# ==========================================================

def collate_station_records(
    batch
):

    X_batch = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    Y_batch = torch.stack(
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

        X_batch,

        Y_batch,

        station_records
    )


# ==========================================================
# DATASETS
# ==========================================================

train_dataset = SpatialTemporalDataset(

    X_train,

    Y_train,

    station_records_train
)


val_dataset = SpatialTemporalDataset(

    X_val,

    Y_val,

    station_records_val
)


test_dataset = SpatialTemporalDataset(

    X_test,

    Y_test,

    station_records_test
)


# ==========================================================
# DATALOADERS
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_station_records,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_station_records,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_station_records,

    pin_memory=torch.cuda.is_available()
)


# ==========================================================
# DATASET INFORMATION
# ==========================================================

INPUT_DIM = X_train.shape[-1]

SEQUENCE_LENGTH = X_train.shape[1]

NUM_STATIONS = X_train.shape[2]


print(
    "\n" + "=" * 70
)

print(
    "DATASET INFORMATION"
)

print(
    "=" * 70
)

print(
    "Training samples   :",
    len(train_dataset)
)

print(
    "Validation samples :",
    len(val_dataset)
)

print(
    "Test samples       :",
    len(test_dataset)
)

print(
    "Sequence length    :",
    SEQUENCE_LENGTH
)

print(
    "Number of stations :",
    NUM_STATIONS
)

print(
    "Input features     :",
    INPUT_DIM
)

print(
    "=" * 70
)


# ==========================================================
# VALIDATE STATION RECORDS
# ==========================================================

def validate_station_records(

    X,

    station_records,

    split_name

):

    if len(X) != len(
        station_records
    ):

        raise ValueError(
            f"{split_name}: X and station "
            f"record counts differ."
        )


    for b in range(
        len(station_records)
    ):

        sample = station_records[b]


        if len(sample) != (
            SEQUENCE_LENGTH
        ):

            raise ValueError(

                f"{split_name}, sample {b}: "
                f"expected "
                f"{SEQUENCE_LENGTH} timesteps, "
                f"got {len(sample)}."
            )


        for t in range(
            SEQUENCE_LENGTH
        ):

            stations = sample[t]


            if len(stations) != (
                NUM_STATIONS
            ):

                raise ValueError(

                    f"{split_name}, "
                    f"sample {b}, timestep {t}: "
                    f"expected "
                    f"{NUM_STATIONS} stations, "
                    f"got {len(stations)}."
                )


    print(
        f"{split_name}: station records PASS"
    )


validate_station_records(
    X_train,
    station_records_train,
    "TRAIN"
)

validate_station_records(
    X_val,
    station_records_val,
    "VALIDATION"
)

validate_station_records(
    X_test,
    station_records_test,
    "TEST"
)


# ==========================================================
# CREATE MODEL
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "CREATING MODEL"
)

print(
    "=" * 70
)


model = SpatioTemporalModel(

    num_nodes=NUM_STATIONS,

    input_dim=INPUT_DIM,

    embedding_dim=64,

    sequence_length=SEQUENCE_LENGTH,

    # --------------------------------------------------
    # Initial values only.
    # HH will search for better values.
    # --------------------------------------------------

    decay_constant=0.5,

    similarity_threshold=0.2,

    num_heads=8,

    use_hyper_heuristic=(
        USE_HYPER_HEURISTIC
    ),

    decay_range=DECAY_RANGE,

    similarity_threshold_range=(
        SIMILARITY_THRESHOLD_RANGE
    ),

    num_candidates=NUM_CANDIDATES,

    exploration_probability=(
        EXPLORATION_PROBABILITY
    ),

    exploitation_fraction=(
        EXPLOITATION_FRACTION
    )
)


model = model.to(
    DEVICE
)


# ==========================================================
# PARAMETER COUNT
# ==========================================================

total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()
)


trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)


print(
    "Total parameters     :",
    total_parameters
)

print(
    "Trainable parameters :",
    trainable_parameters
)


# ==========================================================
# LOSS
# ==========================================================

criterion = nn.HuberLoss(
    delta=1.0
)


# ==========================================================
# INVERSE TRANSFORM
# ==========================================================

def inverse_transform_y(
    values
):

    values_np = (

        values

        .detach()

        .cpu()

        .numpy()
    )


    original_shape = (
        values_np.shape
    )


    flattened = (
        values_np.reshape(
            -1,
            1
        )
    )


    restored = (
        target_scaler.inverse_transform(
            flattened
        )
    )


    return restored.reshape(
        original_shape
    )


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    ).reshape(-1)


    targets = np.asarray(
        targets,
        dtype=np.float64
    ).reshape(-1)


    errors = (
        predictions - targets
    )


    absolute_errors = (
        np.abs(errors)
    )


    squared_errors = (
        errors ** 2
    )


    mae = np.mean(
        absolute_errors
    )


    rmse = np.sqrt(
        np.mean(
            squared_errors
        )
    )


    target_mean = np.mean(
        targets
    )


    ss_res = np.sum(
        squared_errors
    )


    ss_tot = np.sum(
        (
            targets - target_mean
        ) ** 2
    )


    if ss_tot > 0:

        r2 = (
            1.0 -
            ss_res / ss_tot
        )

    else:

        r2 = 0.0


    nonzero_mask = (
        np.abs(targets) > 1e-8
    )


    if np.any(
        nonzero_mask
    ):

        accuracy = (

            100.0 *

            np.mean(

                1.0 -

                (
                    absolute_errors[
                        nonzero_mask
                    ]

                    /

                    np.abs(
                        targets[
                            nonzero_mask
                        ]
                    )
                )
            )
        )

        accuracy = np.clip(
            accuracy,
            0.0,
            100.0
        )

    else:

        accuracy = 0.0


    tolerance_10 = (

        np.abs(errors)

        <=

        0.10 *

        np.maximum(
            np.abs(targets),
            1e-8
        )
    )


    tolerance_20 = (

        np.abs(errors)

        <=

        0.20 *

        np.maximum(
            np.abs(targets),
            1e-8
        )
    )


    accuracy_10 = (
        100.0 *
        np.mean(tolerance_10)
    )


    accuracy_20 = (
        100.0 *
        np.mean(tolerance_20)
    )


    denominator = (

        np.abs(predictions)

        +

        np.abs(targets)
    )


    valid_smape = (
        denominator > 1e-8
    )


    if np.any(
        valid_smape
    ):

        smape = (

            100.0 *

            np.mean(

                2.0 *

                np.abs(

                    predictions[
                        valid_smape
                    ]

                    -

                    targets[
                        valid_smape
                    ]
                )

                /

                denominator[
                    valid_smape
                ]
            )
        )

    else:

        smape = 0.0


    return {

        "mae": float(mae),

        "rmse": float(rmse),

        "r2": float(r2),

        "accuracy": float(
            accuracy
        ),

        "accuracy_10": float(
            accuracy_10
        ),

        "accuracy_20": float(
            accuracy_20
        ),

        "smape": float(
            smape
        )
    }


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(
    current_model,
    loader
):

    current_model.eval()

    total_loss = 0.0

    total_samples = 0

    all_predictions = []

    all_targets = []


    with torch.no_grad():

        for (

            X_batch,

            Y_batch,

            station_records

        ) in loader:


            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )


            Y_batch = Y_batch.to(
                DEVICE,
                non_blocking=True
            )


            output = current_model(

                X_batch,

                station_records
            )


            loss = criterion(

                output,

                Y_batch
            )


            batch_size = (
                X_batch.shape[0]
            )


            total_loss += (

                loss.item()

                *

                batch_size
            )


            total_samples += (
                batch_size
            )


            predictions_original = (
                inverse_transform_y(
                    output
                )
            )


            targets_original = (
                inverse_transform_y(
                    Y_batch
                )
            )


            all_predictions.append(
                predictions_original
            )


            all_targets.append(
                targets_original
            )


    average_loss = (

        total_loss /

        max(
            total_samples,
            1
        )
    )


    predictions = np.concatenate(
        all_predictions,
        axis=0
    )


    targets = np.concatenate(
        all_targets,
        axis=0
    )


    metrics = calculate_metrics(

        predictions,

        targets
    )


    return (

        average_loss,

        metrics,

        predictions,

        targets
    )


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    current_model,
    current_optimizer
):

    current_model.train()

    total_loss = 0.0

    total_samples = 0

    gradient_norms = []


    for (

        X_batch,

        Y_batch,

        station_records

    ) in train_loader:


        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )


        Y_batch = Y_batch.to(
            DEVICE,
            non_blocking=True
        )


        current_optimizer.zero_grad(
            set_to_none=True
        )


        output = current_model(

            X_batch,

            station_records
        )


        loss = criterion(

            output,

            Y_batch
        )


        loss.backward()


        if GRADIENT_CLIPPING:

            gradient_norm = (

                torch.nn.utils
                .clip_grad_norm_(
                    current_model.parameters(),
                    MAX_GRAD_NORM
                )
            )

        else:

            gradient_norm = 0.0


        gradient_norms.append(
            float(gradient_norm)
        )


        current_optimizer.step()


        batch_size = (
            X_batch.shape[0]
        )


        total_loss += (

            loss.item()

            *

            batch_size
        )


        total_samples += (
            batch_size
        )


    average_loss = (

        total_loss /

        max(
            total_samples,
            1
        )
    )


    average_gradient = np.mean(
        gradient_norms
    )


    return (

        average_loss,

        average_gradient
    )


# ==========================================================
# SAVE THE INITIAL MODEL STATE
# ==========================================================

initial_model_state = copy.deepcopy(
    model.state_dict()
)


# ==========================================================
# WARM-UP TRAINING
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "HYPER-HEURISTIC WARM-UP"
)

print(
    "=" * 70
)


warmup_optimizer = (
    torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )
)


warmup_best_state = (
    copy.deepcopy(
        model.state_dict()
    )
)


warmup_best_val_loss = (
    float("inf")
)


for warmup_epoch in range(

    1,

    HYPER_HEURISTIC_WARMUP_EPOCHS + 1

):


    train_loss, _ = (
        train_one_epoch(

            model,

            warmup_optimizer
        )
    )


    val_loss, metrics, _, _ = (
        evaluate(

            model,

            val_loader
        )
    )


    print(

        f"Warm-up Epoch "
        f"[{warmup_epoch}/"
        f"{HYPER_HEURISTIC_WARMUP_EPOCHS}]"

    )

    print(
        f"  Train Loss : "
        f"{train_loss:.6f}"
    )

    print(
        f"  Val Loss   : "
        f"{val_loss:.6f}"
    )

    print(
        f"  Val R²     : "
        f"{metrics['r2']:.6f}"
    )


    if (
        val_loss
        <
        warmup_best_val_loss
    ):

        warmup_best_val_loss = (
            val_loss
        )

        warmup_best_state = (
            copy.deepcopy(
                model.state_dict()
            )
        )


# ==========================================================
# RESTORE BEST WARM-UP STATE
# ==========================================================

model.load_state_dict(
    warmup_best_state
)


# ==========================================================
# SAVE THIS STATE
#
# EVERY HH CANDIDATE STARTS FROM EXACTLY THIS STATE.
# ==========================================================

hh_start_state = copy.deepcopy(
    model.state_dict()
)


# ==========================================================
# HYPER-HEURISTIC CANDIDATE EVALUATION
# ==========================================================

def evaluate_graph_candidate(
    candidate
):

    """
    Evaluate ONE HH candidate.

    Candidate:

        k
        similarity_threshold

    Procedure:

        1. Restore identical warm-up weights.
        2. Apply candidate graph parameters.
        3. Train for a SMALL number of epochs.
        4. Evaluate validation R².
        5. Return validation R².

    Therefore every candidate receives a fair
    comparison under the same starting neural
    network state.
    """


    print(
        "\n" + "-" * 70
    )

    print(
        "EVALUATING HH CANDIDATE"
    )

    print(
        "Decay constant       :",
        candidate[
            "decay_constant"
        ]
    )

    print(
        "Similarity threshold :",
        candidate[
            "similarity_threshold"
        ]
    )


    # ------------------------------------------------------
    # RESET MODEL
    # ------------------------------------------------------

    model.load_state_dict(
        hh_start_state
    )


    # ------------------------------------------------------
    # APPLY CANDIDATE
    # ------------------------------------------------------

    model.set_graph_parameters(

        decay_constant=(
            candidate[
                "decay_constant"
            ]
        ),

        similarity_threshold=(
            candidate[
                "similarity_threshold"
            ]
        )
    )


    # ------------------------------------------------------
    # FRESH OPTIMIZER
    # ------------------------------------------------------

    candidate_optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY
        )
    )


    # ------------------------------------------------------
    # SHORT CANDIDATE TRAINING
    # ------------------------------------------------------

    best_candidate_r2 = (
        float("-inf")
    )


    for candidate_epoch in range(

        1,

        HH_CANDIDATE_EPOCHS + 1

    ):


        candidate_train_loss, _ = (
            train_one_epoch(

                model,

                candidate_optimizer
            )
        )


        (
            candidate_val_loss,

            candidate_metrics,

            _,

            _

        ) = evaluate(

            model,

            val_loader
        )


        candidate_r2 = (
            candidate_metrics["r2"]
        )


        print(

            f"  HH Epoch "
            f"[{candidate_epoch}/"
            f"{HH_CANDIDATE_EPOCHS}] "
            f"Train Loss: "
            f"{candidate_train_loss:.6f} "
            f"| Val R²: "
            f"{candidate_r2:.6f}"
        )


        if candidate_r2 > (
            best_candidate_r2
        ):

            best_candidate_r2 = (
                candidate_r2
            )


    # ------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------
    #
    # Return R² because the NEW HyperHeuristic
    # maximizes R².
    # ------------------------------------------------------

    print(

        "Candidate validation R²:",
        best_candidate_r2
    )


    return float(
        best_candidate_r2
    )


# ==========================================================
# RUN HYPER-HEURISTIC
# ==========================================================

if USE_HYPER_HEURISTIC:

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING HYPER-HEURISTIC"
    )

    print(
        "=" * 70
    )

    print(
        "Candidates:",
        NUM_CANDIDATES
    )

    print(
        "Candidate epochs:",
        HH_CANDIDATE_EPOCHS
    )

    print(
        "Objective: MAXIMIZE validation R²"
    )

    print(
        "=" * 70
    )


    (
        best_candidate,

        best_score

    ) = model.optimize_graph_parameters(

        evaluate_graph_candidate
    )


    if best_candidate is None:

        raise RuntimeError(
            "Hyper-Heuristic failed "
            "to produce a candidate."
        )


else:

    best_candidate = {

        "decay_constant":
            model.decay_constant,

        "similarity_threshold":
            model.similarity_threshold
    }

    best_score = None


# ==========================================================
# RESTORE WARM-UP MODEL
#
# VERY IMPORTANT:
#
# Candidate training must NOT leak into final training.
# ==========================================================

model.load_state_dict(
    hh_start_state
)


# ==========================================================
# APPLY FINAL BEST PARAMETERS
# ==========================================================

model.apply_hyper_heuristic_candidate(
    best_candidate
)


# ==========================================================
# FINAL GRAPH PARAMETERS
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL GRAPH PARAMETERS"
)

print(
    "=" * 70
)

print(
    "Best k / decay constant:",
    model.decay_constant
)

print(
    "Best similarity threshold:",
    model.similarity_threshold
)

if best_score is not None:

    print(
        "Best validation R²:",
        best_score
    )

print(
    "=" * 70
)


# ==========================================================
# FINAL OPTIMIZER
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ==========================================================
# FINAL SCHEDULER
# ==========================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=SCHEDULER_FACTOR,

        patience=SCHEDULER_PATIENCE,

        min_lr=MIN_LEARNING_RATE
    )
)


# ==========================================================
# FINAL TRAINING
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING FINAL TRAINING"
)

print(
    "=" * 70
)

print(
    "Maximum epochs:",
    EPOCHS
)

print(
    "Active k:",
    model.decay_constant
)

print(
    "Active threshold:",
    model.similarity_threshold
)

print(
    "HH is now LOCKED."
)

print(
    "Graph parameters will NOT change "
    "during final training."
)

print(
    "=" * 70
)


best_val_loss = (
    float("inf")
)

best_state = None

best_epoch = 0

epochs_without_improvement = 0

history = []


# ==========================================================
# FINAL EPOCH LOOP
# ==========================================================

for epoch in range(

    1,

    EPOCHS + 1

):


    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    train_loss, gradient_norm = (
        train_one_epoch(

            model,

            optimizer
        )
    )


    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    (
        val_loss,

        metrics,

        _,

        _

    ) = evaluate(

        model,

        val_loader
    )


    # ------------------------------------------------------
    # LR
    # ------------------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------

    improvement = (

        best_val_loss -

        val_loss
    )


    if improvement > (
        EARLY_STOPPING_MIN_DELTA
    ):

        best_val_loss = (
            val_loss
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        best_state = copy.deepcopy(
            model.state_dict()
        )

        improved = True

    else:

        epochs_without_improvement += 1

        improved = False


    # ------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss":
            float(train_loss),

        "val_loss":
            float(val_loss),

        "mae":
            metrics["mae"],

        "rmse":
            metrics["rmse"],

        "r2":
            metrics["r2"],

        "accuracy":
            metrics["accuracy"],

        "accuracy_10":
            metrics["accuracy_10"],

        "accuracy_20":
            metrics["accuracy_20"],

        "smape":
            metrics["smape"],

        "gradient":
            float(gradient_norm),

        "learning_rate":
            float(current_lr),

        # These should be identical for EVERY
        # final epoch.

        "decay_constant":
            float(model.decay_constant),

        "similarity_threshold":
            float(
                model.similarity_threshold
            )
    })


    # ------------------------------------------------------
    # PRINT
    # ------------------------------------------------------

    print(

        f"\nEpoch "
        f"[{epoch:04d}/"
        f"{EPOCHS}]"
    )

    print(
        f"  Train Loss : "
        f"{train_loss:.6f}"
    )

    print(
        f"  Val Loss   : "
        f"{val_loss:.6f}"
    )

    print(
        f"  MAE        : "
        f"{metrics['mae']:.4f} µg/m³"
    )

    print(
        f"  RMSE       : "
        f"{metrics['rmse']:.4f} µg/m³"
    )

    print(
        f"  R²         : "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"  Accuracy   : "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"  Accuracy ±10% : "
        f"{metrics['accuracy_10']:.2f}%"
    )

    print(
        f"  Accuracy ±20% : "
        f"{metrics['accuracy_20']:.2f}%"
    )

    print(
        f"  SMAPE      : "
        f"{metrics['smape']:.2f}%"
    )

    print(
        f"  Grad Norm  : "
        f"{gradient_norm:.4f}"
    )

    print(
        f"  Learning Rate : "
        f"{current_lr:.8f}"
    )

    print(
        f"  Graph k       : "
        f"{model.decay_constant:.6f}"
    )

    print(
        f"  Graph threshold : "
        f"{model.similarity_threshold:.6f}"
    )


    if improved:

        print(
            "  ⭐ New best validation model"
        )

    else:

        print(

            f"  No improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEARLY STOPPING"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        break


# ==========================================================
# RESTORE BEST FINAL MODEL
# ==========================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ==========================================================
# SAVE HISTORY
# ==========================================================

with open(
    HISTORY_FILE,
    "w"
) as file:

    json.dump(

        history,

        file,

        indent=4
    )


print(
    "\nSaved:",
    HISTORY_FILE
)


# ==========================================================
# FINAL TEST
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL TEST EVALUATION"
)

print(
    "=" * 70
)


(
    test_loss,

    test_metrics,

    predictions,

    targets

) = evaluate(

    model,

    test_loader
)


print(
    "Test Loss:",
    test_loss
)

print(
    "Test MAE:",
    test_metrics["mae"]
)

print(
    "Test RMSE:",
    test_metrics["rmse"]
)

print(
    "Test R²:",
    test_metrics["r2"]
)

print(
    "Test Accuracy:",
    test_metrics["accuracy"],
    "%"
)

print(
    "Test Accuracy ±10%:",
    test_metrics["accuracy_10"],
    "%"
)

print(
    "Test Accuracy ±20%:",
    test_metrics["accuracy_20"],
    "%"
)

print(
    "Test SMAPE:",
    test_metrics["smape"],
    "%"
)


print(
    "\n" + "=" * 70
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 70
)
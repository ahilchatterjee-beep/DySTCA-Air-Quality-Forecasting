# ==========================================================
# config.py
# ==========================================================
# Central configuration for the Air Quality Prediction Model
#
# IMPORTANT:
# This file is the SINGLE SOURCE OF TRUTH for model
# dimensions and major training/model settings.
#
# Changing EMBEDDING_DIM here should dynamically propagate
# through the TCN, GAT, Fusion/Cross-Attention and Prediction
# layers, provided those modules use the configuration values.
# ==========================================================


# ==========================================================
# DATA CONFIGURATION
# ==========================================================

# Number of input features per station.
#
# Features:
#   1. pm2_5
#   2. pm10
#   3. co
#   4. no2
#   5. temp_c
#   6. humidity
#   7. pressure_mb
#   8. windspeed_kph
#   9. hour_sin
#  10. hour_cos
#  11. day_sin
#  12. day_cos
#  13. month_sin
#  14. month_cos
INPUT_DIM = 14


# Number of historical hourly observations used as input.
SEQUENCE_LENGTH = 24


# Number of future hourly observations predicted.
#
# Current experiment:
#   24-hour forecasting
PREDICTION_HORIZON = 8


# Number of monitoring stations.
NUM_STATIONS = 31


# ==========================================================
# MODEL CAPACITY
# ==========================================================

# ----------------------------------------------------------
# Main representation dimension
# ----------------------------------------------------------
#
# COMPACT REGULARIZED ARCHITECTURE:
#   64-dimensional representation (~262K total parameters).
#   Prevents memorization of 5,998 training windows.
#
# This dimension is shared across the major representation
# stages:
#
#   TCN
#     ↓
#   Temporal Embedding [64]
#
#   GAT
#     ↓
#   Spatial Embedding [64]
#
#   Fusion / Cross-Attention
#     ↓
#   Fused Embedding [64]
#
#   Prediction Head
#     ↓
#   24-hour prediction
#
EMBEDDING_DIM = 64


# ==========================================================
# TEMPORAL ENCODER / TCN
# ==========================================================

# Number of residual TCN blocks.
NUM_TCN_BLOCKS = 6


# Dilation pattern used by the existing TCN.
TCN_DILATIONS = [1, 2, 4, 8, 16, 32]


# Kernel size used by the TCN.
TCN_KERNEL_SIZE = 3


# ==========================================================
# SPATIAL ENCODER / GAT
# ==========================================================

# Number of attention heads used by the spatial GAT.
GAT_HEADS = 4


# ==========================================================
# FUSION / CROSS-ATTENTION
# ==========================================================

# Number of attention heads in Cross-Attention.
# 64 / 4 = 16 dimensions per head.
NUM_ATTENTION_HEADS = 4


# Number of fusion tokens.
NUM_FUSION_TOKENS = 4


# ==========================================================
# SPATIAL GRAPH CONFIGURATION
# ==========================================================

# Graph distance decay constant.
GRAPH_DECAY_CONSTANT = 0.5


# Similarity threshold used to determine graph connections.
GRAPH_SIMILARITY_THRESHOLD = 0.2


# ==========================================================
# TARGET CONFIGURATION
# ==========================================================

# Prediction target.
TARGET_COLUMN = "pm2_5"


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

# Batch size.
BATCH_SIZE = 4


# Learning rate.
# 0.0003 provides rapid, stable convergence for compact 64-dim models.
LEARNING_RATE = 0.0003


# Maximum number of epochs.
MAX_EPOCHS = 1


# ==========================================================
# OPTIMIZATION
# ==========================================================

# AdamW weight decay.
# Set to 0.01 for strong L2 weight regularization to stop parameter drift.
WEIGHT_DECAY = 0.01

# Gradient clipping threshold.
GRADIENT_CLIP_VALUE = 1.0


# ==========================================================
# EARLY STOPPING / LR SCHEDULER
# ==========================================================

# Number of epochs allowed without validation improvement
# before training stops.
EARLY_STOPPING_PATIENCE = 25

# Minimum validation R² improvement required to reset
# early-stopping patience.
EARLY_STOPPING_MIN_DELTA = 0.0

# Reduce LR when validation R² stops improving.
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# Purge gap between chronological splits.
# For a 24h input + 24h forecast, neighboring windows
# share temporal observations unless this boundary is removed.
PURGE_GAP = SEQUENCE_LENGTH + PREDICTION_HORIZON - 1


# ==========================================================
# RANDOMNESS / REPRODUCIBILITY
# ==========================================================

RANDOM_SEED = 42


# ==========================================================
# DEVICE
# ==========================================================

# Automatically use CUDA when available.
#
# This does NOT force GPU usage.
# If CUDA is unavailable, CPU is used automatically.
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


# ==========================================================
# CONFIGURATION VALIDATION
# ==========================================================

def validate_config():
    """
    Validate the central model configuration before training.
    """

    # ------------------------------------------------------
    # Basic dimension checks
    # ------------------------------------------------------

    if INPUT_DIM <= 0:
        raise ValueError("INPUT_DIM must be greater than 0.")

    if EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM must be greater than 0.")

    if SEQUENCE_LENGTH <= 0:
        raise ValueError(
            "SEQUENCE_LENGTH must be greater than 0."
        )

    if PREDICTION_HORIZON <= 0:
        raise ValueError(
            "PREDICTION_HORIZON must be greater than 0."
        )

    if NUM_STATIONS <= 0:
        raise ValueError(
            "NUM_STATIONS must be greater than 0."
        )

    # ------------------------------------------------------
    # Attention-head compatibility
    # ------------------------------------------------------

    if EMBEDDING_DIM % NUM_ATTENTION_HEADS != 0:
        raise ValueError(
            f"EMBEDDING_DIM ({EMBEDDING_DIM}) must be "
            f"divisible by NUM_ATTENTION_HEADS "
            f"({NUM_ATTENTION_HEADS})."
        )

    # ------------------------------------------------------
    # Fusion-token compatibility
    # ------------------------------------------------------

    if EMBEDDING_DIM % NUM_FUSION_TOKENS != 0:
        raise ValueError(
            f"EMBEDDING_DIM ({EMBEDDING_DIM}) must be "
            f"divisible by NUM_FUSION_TOKENS "
            f"({NUM_FUSION_TOKENS})."
        )

    # ------------------------------------------------------
    # TCN configuration
    # ------------------------------------------------------

    if NUM_TCN_BLOCKS != len(TCN_DILATIONS):
        raise ValueError(
            "NUM_TCN_BLOCKS must match the number of "
            "TCN_DILATIONS."
        )

    if TCN_KERNEL_SIZE <= 0:
        raise ValueError(
            "TCN_KERNEL_SIZE must be greater than 0."
        )

    # ------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be greater than 0."
        )

    if LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE must be greater than 0."
        )

    if MAX_EPOCHS <= 0:
        raise ValueError(
            "MAX_EPOCHS must be greater than 0."
        )

    if EARLY_STOPPING_PATIENCE < 0:
        raise ValueError(
            "EARLY_STOPPING_PATIENCE cannot be negative."
        )

    print("Configuration validation: PASS")


# ==========================================================
# DISPLAY CONFIGURATION
# ==========================================================

def print_config():
    """
    Print the important configuration values.
    """

    print()
    print("=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)

    print(f"Input features          : {INPUT_DIM}")
    print(f"Input sequence length   : {SEQUENCE_LENGTH}")
    print(f"Prediction horizon      : {PREDICTION_HORIZON}")
    print(f"Number of stations      : {NUM_STATIONS}")

    print()
    print("MODEL CAPACITY")
    print("-" * 70)
    print(f"Embedding dimension     : {EMBEDDING_DIM}")

    print()
    print("TCN")
    print("-" * 70)
    print(f"TCN residual blocks     : {NUM_TCN_BLOCKS}")
    print(f"TCN dilations           : {TCN_DILATIONS}")
    print(f"TCN kernel size         : {TCN_KERNEL_SIZE}")

    print()
    print("ATTENTION")
    print("-" * 70)
    print(f"GAT heads               : {GAT_HEADS}")
    print(f"Cross-attention heads   : {NUM_ATTENTION_HEADS}")
    print(f"Fusion tokens           : {NUM_FUSION_TOKENS}")

    print()
    print("ATTENTION DIMENSIONS")
    print("-" * 70)
    print(
        f"Cross-attention head dimension : "
        f"{EMBEDDING_DIM // NUM_ATTENTION_HEADS}"
    )
    print(
        f"Fusion token dimension         : "
        f"{EMBEDDING_DIM // NUM_FUSION_TOKENS}"
    )

    print()
    print("GRAPH")
    print("-" * 70)
    print(
        f"Decay constant          : "
        f"{GRAPH_DECAY_CONSTANT}"
    )
    print(
        f"Similarity threshold    : "
        f"{GRAPH_SIMILARITY_THRESHOLD}"
    )

    print()
    print("TRAINING")
    print("-" * 70)
    print(f"Batch size              : {BATCH_SIZE}")
    print(f"Learning rate           : {LEARNING_RATE}")
    print(f"Maximum epochs          : {MAX_EPOCHS}")
    print(
        f"Early stopping patience : "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(f"Device                  : {DEVICE}")

    print("=" * 70)
    print()


# ==========================================================
# OPTIONAL AUTOMATIC VALIDATION
# ==========================================================

validate_config()

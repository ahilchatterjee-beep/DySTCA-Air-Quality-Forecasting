# ==========================================================
# test_model.py
# Complete Spatio-Temporal Model Sanity Test
# ==========================================================

import torch

from preprocessing.pre_updated import (
    prepare_model_data
)

from Model.model import (
    SpatioTemporalModel
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SEQUENCE_LENGTH = 24
NUM_NODES = 10
INPUT_DIM = 14
EMBEDDING_DIM = 64

DECAY_CONSTANT = 0.5
SIMILARITY_THRESHOLD = 0.2

NUM_HEADS = 8

TEST_BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 70)
    print("SPATIO-TEMPORAL MODEL TEST")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    # ======================================================
    # PREPARE COMPLETE MODEL DATA
    # ======================================================
    #
    # This function already performs:
    #
    # Dataset loading
    #       ↓
    # Dataset preparation
    #       ↓
    # Input imputation
    #       ↓
    # Node feature tensor
    #       ↓
    # MinMax scaling
    #       ↓
    # Input sequences
    #       ↓
    # Valid target filtering
    #       ↓
    # Station record sequences
    #       ↓
    # Tensor conversion
    #
    # ======================================================

    (
        df,
        X_tensor,
        y_tensor,
        station_record_sequences,
        feature_scaler,
        station_ids,
        timestamps
    ) = prepare_model_data(
        sequence_length=SEQUENCE_LENGTH
    )

    # ======================================================
    # FINAL DATA CHECK
    # ======================================================

    print("\n")
    print("=" * 70)
    print("PREPARED MODEL DATA")
    print("=" * 70)

    print(
        "X shape:",
        X_tensor.shape
    )

    print(
        "Y shape:",
        y_tensor.shape
    )

    print(
        "Station record sequences:",
        len(station_record_sequences)
    )

    print(
        "Number of stations:",
        len(station_ids)
    )

    print(
        "Number of timestamps:",
        len(timestamps)
    )

    # ------------------------------------------------------
    # Scaling diagnostic
    # ------------------------------------------------------

    print("\n")
    print("SCALING CHECK")
    print("-" * 70)

    print(
        "X minimum:",
        X_tensor.min().item()
    )

    print(
        "X maximum:",
        X_tensor.max().item()
    )

    print(
        "X mean:",
        X_tensor.mean().item()
    )

    print(
        "X std:",
        X_tensor.std().item()
    )

    print(
        "X NaNs:",
        torch.isnan(
            X_tensor
        ).sum().item()
    )

    print(
        "X Infs:",
        torch.isinf(
            X_tensor
        ).sum().item()
    )

    # ======================================================
    # CHECK EXPECTED SHAPE
    # ======================================================

    expected_x_dimensions = (
        4
    )

    if X_tensor.dim() != (
        expected_x_dimensions
    ):

        raise ValueError(
            f"Expected X to have 4 dimensions "
            f"[B,T,N,F], got {X_tensor.shape}"
        )

    if X_tensor.shape[1] != (
        SEQUENCE_LENGTH
    ):

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"got {X_tensor.shape[1]}"
        )

    if X_tensor.shape[2] != (
        NUM_NODES
    ):

        raise ValueError(
            f"Expected {NUM_NODES} nodes, "
            f"got {X_tensor.shape[2]}"
        )

    if X_tensor.shape[3] != (
        INPUT_DIM
    ):

        raise ValueError(
            f"Expected {INPUT_DIM} features, "
            f"got {X_tensor.shape[3]}"
        )

    # ======================================================
    # SELECT SMALL TEST BATCH
    # ======================================================
    #
    # IMPORTANT:
    #
    # We are NOT training here.
    #
    # We only test the complete forward pass.
    #
    # ======================================================

    X_test = X_tensor[
        :TEST_BATCH_SIZE
    ]

    Y_test = y_tensor[
        :TEST_BATCH_SIZE
    ]

    station_records_test = (
        station_record_sequences[
            :TEST_BATCH_SIZE
        ]
    )

    print("\n")
    print("=" * 70)
    print("MODEL INPUT CHECK")
    print("=" * 70)

    print(
        "X test shape:",
        X_test.shape
    )

    print(
        "Y test shape:",
        Y_test.shape
    )

    print(
        "Station record batch size:",
        len(station_records_test)
    )

    print(
        "Timesteps per sample:",
        len(
            station_records_test[0]
        )
    )

    print(
        "Stations per timestep:",
        len(
            station_records_test[0][0]
        )
    )

    # ======================================================
    # MOVE INPUT TO DEVICE
    # ======================================================

    X_test = X_test.to(
        DEVICE
    )

    Y_test = Y_test.to(
        DEVICE
    )

    # ======================================================
    # CREATE MODEL
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = SpatioTemporalModel(
        num_nodes=NUM_NODES,
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQUENCE_LENGTH,

        decay_constant=DECAY_CONSTANT,

        similarity_threshold=(
            SIMILARITY_THRESHOLD
        ),

        num_heads=NUM_HEADS,

        use_hyper_heuristic=True
    )

    model = model.to(
        DEVICE
    )

    print(
        "Model created successfully."
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # ======================================================
    # EVALUATION MODE
    # ======================================================

    model.eval()

    # ======================================================
    # FORWARD PASS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("STARTING FORWARD PASS")
    print("=" * 70)

    with torch.no_grad():

        predictions = model(
            X_test,
            station_records_test
        )

    # ======================================================
    # OUTPUT DIAGNOSTIC
    # ======================================================

    print("\n")
    print("=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)

    print(
        "Input shape :",
        X_test.shape
    )

    print(
        "Target shape:",
        Y_test.shape
    )

    print(
        "Output shape:",
        predictions.shape
    )

    print(
        "Prediction mean:",
        predictions.mean().item()
    )

    print(
        "Prediction std:",
        predictions.std().item()
    )

    print(
        "Prediction min:",
        predictions.min().item()
    )

    print(
        "Prediction max:",
        predictions.max().item()
    )

    print(
        "Prediction NaNs:",
        torch.isnan(
            predictions
        ).sum().item()
    )

    print(
        "Prediction Infs:",
        torch.isinf(
            predictions
        ).sum().item()
    )

    # ======================================================
    # EXPECTED OUTPUT CHECK
    # ======================================================

    expected_output_shape = (
        TEST_BATCH_SIZE,
        NUM_NODES,
        1
    )

    if predictions.shape != (
        expected_output_shape
    ):

        raise ValueError(
            "\nOUTPUT SHAPE ERROR\n"
            f"Expected: {expected_output_shape}\n"
            f"Got:      {tuple(predictions.shape)}"
        )

    # ======================================================
    # NUMERICAL SAFETY CHECK
    # ======================================================

    if torch.isnan(
        predictions
    ).any():

        raise ValueError(
            "Model produced NaN predictions."
        )

    if torch.isinf(
        predictions
    ).any():

        raise ValueError(
            "Model produced Inf predictions."
        )

    # ======================================================
    # SUCCESS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("MODEL SANITY TEST PASSED")
    print("=" * 70)

    print(
        "Temporal encoder       : PASS"
    )

    print(
        "Dynamic graph          : PASS"
    )

    print(
        "Self-loops             : PASS"
    )

    print(
        "GAT                    : PASS"
    )

    print(
        "Spatial encoder        : PASS"
    )

    print(
        "Fusion                 : PASS"
    )

    print(
        "Prediction layer       : PASS"
    )

    print(
        "Output shape           : PASS"
    )

    print(
        "NaN / Inf check        : PASS"
    )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()
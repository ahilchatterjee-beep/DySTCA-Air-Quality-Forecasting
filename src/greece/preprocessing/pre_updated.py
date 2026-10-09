# ==========================================================
# pre_updated.py
#
# QUALITY-CONTROLLED PREPROCESSING
#
# Pipeline:
#
# quality_controlled_sequences.pt
#             ↓
# chronological split
#             ↓
# StandardScaler(X)  ← fitted on TRAIN only
# StandardScaler(Y)  ← fitted on TRAIN only
#             ↓
# scaled train / validation / test
#             ↓
# train.py
#
# ==========================================================

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

QUALITY_DATASET_PATH = (
    "data/quality_controlled_sequences_fixed.pt"
)

SEQUENCE_LENGTH = 24

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================================
# FEATURE DEFINITIONS
# ==========================================================

FEATURE_COLUMNS = [

    "PM2.5 (µg/m³)",

    "PM10 (µg/m³)",

    "NO (µg/m³)",

    "NO2 (µg/m³)",

    "SO2 (µg/m³)",

    "CO (mg/m³)",

    "Ozone (µg/m³)",

    "Temperature",

    "Dew_Point",

    "Wind_U",

    "Wind_V",

    "Surface_Pressure",

    "Boundary_Layer_Height",

    "Rainfall"
]


# ==========================================================
# FIRST 7 FEATURES ARE POLLUTANTS
# ==========================================================

POLLUTANT_FEATURE_COUNT = 7

# ----------------------------------------------------------
# INVALID POLLUTANT INPUT HANDLING
# ----------------------------------------------------------
# The .pt dataset already contains a validity mask for the
# first 7 pollutant features.
#
# We keep the original .pt file unchanged.
# After StandardScaler, invalid pollutant inputs are replaced
# with 0.0, which is the neutral/mean value in standardized
# space. Meteorological features are left unchanged.
#
# IMPORTANT:
# This affects INPUT X only. Y/targets are intentionally not
# modified because the existing mask describes the 24 input
# timesteps, not the future target directly.
# ----------------------------------------------------------

MASK_INVALID_POLLUTANTS = True


# ==========================================================
# NORMALIZE STATION RECORD STRUCTURE
# ==========================================================

def normalize_station_record_sequences(
    station_record_sequences,
    num_stations
):
    """
    Ensures station records have exactly:

        [sample][timestep][station]

    Expected:
        sample       -> sequence
        sequence     -> 24 timesteps
        timestep     -> 10 station dictionaries

    Legacy malformed data may contain:

        24 timesteps × 140 records

    where the same station record was repeated once per
    feature (10 stations × 14 features).

    This function removes those duplicate records.
    """

    print("\n" + "=" * 70)
    print("NORMALIZING STATION RECORD STRUCTURE")
    print("=" * 70)

    normalized = []

    for sample_index, sequence in enumerate(
        station_record_sequences
    ):

        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sample {sample_index}: expected "
                f"{SEQUENCE_LENGTH} timesteps, "
                f"got {len(sequence)}."
            )

        normalized_sequence = []

        for timestep_index, timestep_records in enumerate(
            sequence
        ):

            # --------------------------------------------------
            # Already correct
            # --------------------------------------------------

            if len(timestep_records) == num_stations:

                station_map = {}

                for record in timestep_records:

                    station_id = int(
                        record["Station_ID"]
                    )

                    if station_id in station_map:
                        raise ValueError(
                            f"Duplicate station {station_id} "
                            f"found in correctly-sized timestep."
                        )

                    station_map[station_id] = record

                if len(station_map) != num_stations:
                    raise ValueError(
                        f"Sample {sample_index}, "
                        f"timestep {timestep_index}: "
                        f"expected {num_stations} unique stations, "
                        f"got {len(station_map)}."
                    )

                normalized_timestep = list(
                    station_map.values()
                )

            # --------------------------------------------------
            # Legacy malformed structure
            #
            # Example:
            #     140 records
            #
            #     10 stations × 14 features
            # --------------------------------------------------

            else:

                station_map = {}

                for record in timestep_records:

                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Sample {sample_index}, "
                            f"timestep {timestep_index}: "
                            "station record is not a dictionary."
                        )

                    if "Station_ID" not in record:
                        raise ValueError(
                            f"Sample {sample_index}, "
                            f"timestep {timestep_index}: "
                            "Station_ID missing."
                        )

                    station_id = int(
                        record["Station_ID"]
                    )

                    # Keep the first complete record for
                    # each station and discard duplicates.
                    if station_id not in station_map:
                        station_map[station_id] = record

                if len(station_map) != num_stations:
                    raise ValueError(
                        f"Sample {sample_index}, "
                        f"timestep {timestep_index}: "
                        f"expected {num_stations} unique stations "
                        f"after normalization, "
                        f"got {len(station_map)}."
                    )

                normalized_timestep = list(
                    station_map.values()
                )

            # --------------------------------------------------
            # Final station count check
            # --------------------------------------------------

            if len(normalized_timestep) != num_stations:
                raise ValueError(
                    f"Sample {sample_index}, "
                    f"timestep {timestep_index}: "
                    f"expected {num_stations} stations, "
                    f"got {len(normalized_timestep)}."
                )

            normalized_sequence.append(
                normalized_timestep
            )

        normalized.append(
            normalized_sequence
        )

    # ==========================================================
    # FINAL VALIDATION
    # ==========================================================

    print(
        "Original samples:",
        len(station_record_sequences)
    )

    print(
        "Normalized samples:",
        len(normalized)
    )

    if len(normalized) > 0:

        print(
            "Timesteps in sample 0:",
            len(normalized[0])
        )

        print(
            "Stations in timestep 0:",
            len(normalized[0][0])
        )

        print(
            "First station:",
            normalized[0][0][0]["Station_ID"]
        )

    print("=" * 70)

    return normalized


# ==========================================================
# LOAD QUALITY-CONTROLLED DATA
# ==========================================================

def load_quality_controlled_data(
    path=QUALITY_DATASET_PATH
):

    print("\n" + "=" * 70)
    print("LOADING QUALITY-CONTROLLED DATASET")
    print("=" * 70)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Expected quality-controlled dataset to be a dictionary."
        )

    print("Available keys:", list(data.keys()))

    # X
    if "X" in data:
        X = data["X"]
    elif "x" in data:
        X = data["x"]
    else:
        raise KeyError("X was not found in dataset.")

    # Y
    if "Y" in data:
        Y = data["Y"]
    elif "y" in data:
        Y = data["y"]
    else:
        raise KeyError("Y was not found in dataset.")

    # Validity mask
    if "validity_mask" in data:
        validity_mask = data["validity_mask"]
    elif "mask" in data:
        validity_mask = data["mask"]
    else:
        raise KeyError("Validity mask was not found.")

    # Sequence indices
    if "sequence_indices" in data:
        sequence_indices = data["sequence_indices"]
    elif "indices" in data:
        sequence_indices = data["indices"]
    else:
        sequence_indices = torch.arange(X.shape[0])

    # ------------------------------------------------------
    # STATION RECORD SEQUENCES
    #
    # Required by SpatioTemporalModel. These records contain
    # sequence-aligned station metadata / wind information.
    #
    # They are NOT scaled.
    # ------------------------------------------------------
    if "station_record_sequences" in data:
        station_record_sequences = data["station_record_sequences"]
    elif "station_records" in data:
        station_record_sequences = data["station_records"]
    else:
        raise KeyError(
            "\n\nstation_record_sequences is missing from the "
            "quality-controlled dataset.\n\n"
            "The SpatioTemporalModel requires station records "
            "because the spatial graph uses station metadata "
            "and wind information.\n\n"
            "Rebuild the quality-controlled dataset with "
            "station_record_sequences before training."
        )

    # Numeric tensors
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    validity_mask = torch.as_tensor(validity_mask, dtype=torch.bool)
    sequence_indices = torch.as_tensor(sequence_indices, dtype=torch.long)

    # ------------------------------------------------------
    # NORMALIZE STATION RECORD STRUCTURE
    # ------------------------------------------------------
    #
    # The quality-controlled file may contain legacy station
    # records with duplicate entries (for example 140 records
    # for 10 stations × 14 features).
    #
    # The spatial model requires:
    #
    #     [sample][timestep][station]
    #
    # with exactly one complete dictionary per station.
    # ------------------------------------------------------

    station_record_sequences = normalize_station_record_sequences(
        station_record_sequences,
        num_stations=X.shape[2]
    )

    # Station records must be sample-aligned with X
    try:
        station_record_count = len(station_record_sequences)
    except TypeError:
        raise ValueError(
            "station_record_sequences must be sequence-aligned with X."
        )

    if station_record_count != X.shape[0]:
        raise ValueError(
            "station_record_sequences length does not match X. "
            f"X sequences={X.shape[0]}, "
            f"station records={station_record_count}."
        )

    # ------------------------------------------------------
    # VERIFY [samples][timesteps][stations]
    # ------------------------------------------------------

    if station_record_count > 0:

        if len(station_record_sequences[0]) != X.shape[1]:
            raise ValueError(
                "Station-record timestep count does not match X. "
                f"Expected {X.shape[1]}, "
                f"got {len(station_record_sequences[0])}."
            )

        if len(station_record_sequences[0][0]) != X.shape[2]:
            raise ValueError(
                "Station-record station count does not match X. "
                f"Expected {X.shape[2]}, "
                f"got {len(station_record_sequences[0][0])}."
            )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Validity mask:", validity_mask.shape)
    print("Sequence indices:", sequence_indices.shape)
    print("Station record sequences:", station_record_count)

    return (
        X,
        Y,
        validity_mask,
        sequence_indices,
        station_record_sequences
    )


# ==========================================================
# DATASET VALIDATION
# ==========================================================

def validate_dataset(
    X,
    Y,
    validity_mask
):

    print("\n" + "=" * 70)

    print(
        "DATASET VALIDATION"
    )

    print("=" * 70)


    # ------------------------------------------------------
    # X
    # ------------------------------------------------------

    if X.ndim != 4:

        raise ValueError(
            "X must have shape "
            "[samples, timesteps, stations, features]."
        )


    # ------------------------------------------------------
    # Y
    # ------------------------------------------------------

    if Y.ndim != 3:

        raise ValueError(
            "Y must have shape "
            "[samples, stations, 1]."
        )


    # ------------------------------------------------------
    # MASK
    # ------------------------------------------------------

    if validity_mask.ndim != 4:

        raise ValueError(
            "Validity mask must have shape "
            "[samples, timesteps, stations, pollutants]."
        )


    samples = X.shape[0]

    timesteps = X.shape[1]

    stations = X.shape[2]

    features = X.shape[3]


    if timesteps != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected sequence length "
            f"{SEQUENCE_LENGTH}, "
            f"received {timesteps}."
        )


    if features != len(
        FEATURE_COLUMNS
    ):

        raise ValueError(
            f"Expected "
            f"{len(FEATURE_COLUMNS)} "
            f"features, received "
            f"{features}."
        )


    if Y.shape[0] != samples:

        raise ValueError(
            "X and Y sample counts differ."
        )


    if validity_mask.shape[0] != samples:

        raise ValueError(
            "X and validity mask "
            "sample counts differ."
        )


    if validity_mask.shape[1] != timesteps:

        raise ValueError(
            "Validity mask timestep count "
            "does not match X."
        )


    if validity_mask.shape[2] != stations:

        raise ValueError(
            "Validity mask station count "
            "does not match X."
        )


    if validity_mask.shape[3] != (
        POLLUTANT_FEATURE_COUNT
    ):

        raise ValueError(
            "Validity mask must contain "
            f"{POLLUTANT_FEATURE_COUNT} "
            "pollutant features."
        )


    print(
        "Samples:",
        samples
    )

    print(
        "Timesteps:",
        timesteps
    )

    print(
        "Stations:",
        stations
    )

    print(
        "Features:",
        features
    )

    print(
        "Validation: PASS"
    )


# ==========================================================
# CHRONOLOGICAL SPLIT
# ==========================================================

def chronological_split(
    X,
    Y,
    validity_mask,
    sequence_indices
):

    print("\n" + "=" * 70)

    print(
        "CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT"
    )

    print("=" * 70)


    total_samples = X.shape[0]


    train_end = int(
        total_samples
        *
        TRAIN_RATIO
    )


    val_end = (
        train_end
        +
        int(
            total_samples
            *
            VAL_RATIO
        )
    )


    # ======================================================
    # X
    # ======================================================

    X_train = X[
        :train_end
    ]

    X_val = X[
        train_end:val_end
    ]

    X_test = X[
        val_end:
    ]


    # ======================================================
    # Y
    # ======================================================

    Y_train = Y[
        :train_end
    ]

    Y_val = Y[
        train_end:val_end
    ]

    Y_test = Y[
        val_end:
    ]


    # ======================================================
    # MASK
    # ======================================================

    mask_train = validity_mask[
        :train_end
    ]

    mask_val = validity_mask[
        train_end:val_end
    ]

    mask_test = validity_mask[
        val_end:
    ]


    # ======================================================
    # INDICES
    # ======================================================

    indices_train = sequence_indices[
        :train_end
    ]

    indices_val = sequence_indices[
        train_end:val_end
    ]

    indices_test = sequence_indices[
        val_end:
    ]


    print(
        "Total:",
        total_samples
    )

    print(
        "Training:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_val)
    )

    print(
        "Test:",
        len(X_test)
    )


    # ======================================================
    # CHRONOLOGY CHECK
    # ======================================================

    if (
        len(indices_train) > 0
        and
        len(indices_val) > 0
    ):

        if (
            indices_train[-1]
            >=
            indices_val[0]
        ):

            raise ValueError(
                "Train/validation chronological "
                "ordering is invalid."
            )


    if (
        len(indices_val) > 0
        and
        len(indices_test) > 0
    ):

        if (
            indices_val[-1]
            >=
            indices_test[0]
        ):

            raise ValueError(
                "Validation/test chronological "
                "ordering is invalid."
            )


    print(
        "Chronological ordering: PASS"
    )


    return (

        X_train,
        X_val,
        X_test,

        Y_train,
        Y_val,
        Y_test,

        mask_train,
        mask_val,
        mask_test,

        indices_train,
        indices_val,
        indices_test
    )


# ==========================================================
# FIT X STANDARD SCALER
# ==========================================================
#
# IMPORTANT:
#
# Only TRAIN is used.
#
# For pollutant features:
# only genuine observations are used.
#
# For meteorological features:
# all training observations are used.
#
# ==========================================================

def fit_x_scaler(
    X_train,
    mask_train
):

    print("\n" + "=" * 70)

    print(
        "FITTING X STANDARD SCALER"
    )

    print("=" * 70)


    (
        samples,
        timesteps,
        stations,
        features
    ) = X_train.shape


    X_flat = (
        X_train
        .numpy()
        .reshape(
            -1,
            features
        )
    )


    mask_flat = (
        mask_train
        .numpy()
        .reshape(
            -1,
            POLLUTANT_FEATURE_COUNT
        )
    )


    means = np.zeros(
        features,
        dtype=np.float64
    )


    stds = np.ones(
        features,
        dtype=np.float64
    )


    for feature_index in range(
        features
    ):

        values = X_flat[
            :,
            feature_index
        ]


        # --------------------------------------------------
        # Pollutants
        # --------------------------------------------------

        if (
            feature_index
            <
            POLLUTANT_FEATURE_COUNT
        ):

            genuine = mask_flat[
                :,
                feature_index
            ]


            values = values[
                genuine
            ]


            if len(values) == 0:

                raise ValueError(
                    "No genuine observations "
                    f"for feature "
                    f"{FEATURE_COLUMNS[feature_index]}."
                )


        # --------------------------------------------------
        # Meteorological features
        # --------------------------------------------------

        means[
            feature_index
        ] = np.mean(
            values
        )


        stds[
            feature_index
        ] = np.std(
            values
        )


        if (
            stds[
                feature_index
            ]
            <
            1e-12
        ):

            stds[
                feature_index
            ] = 1.0


        print(
            f"{FEATURE_COLUMNS[feature_index]:30s}"
            f" mean={means[feature_index]:12.5f}"
            f" std={stds[feature_index]:12.5f}"
        )


    # ======================================================
    # Construct sklearn StandardScaler
    # ======================================================

    scaler = StandardScaler()


    scaler.mean_ = means

    scaler.scale_ = stds

    scaler.var_ = (
        stds ** 2
    )

    scaler.n_features_in_ = features


    print(
        "\nX scaler fitted using TRAINING DATA ONLY."
    )


    return scaler


# ==========================================================
# APPLY X SCALER
# ==========================================================

def apply_x_scaler(
    X,
    scaler,
    validity_mask=None
):
    """
    Apply the TRAIN-fitted X scaler.

    For the first 7 pollutant features, the quality-controlled
    dataset provides a validity mask.

    Invalid pollutant values are NOT deleted and the .pt file is
    NOT rebuilt. Instead, after standardization they are replaced
    with 0.0, which corresponds to the training mean in
    standardized space.

    Meteorological features are left unchanged.

    Parameters
    ----------
    X : torch.Tensor
        Shape [samples, timesteps, stations, features].

    scaler : StandardScaler
        Fitted using TRAIN data only.

    validity_mask : torch.Tensor or None
        Shape [samples, timesteps, stations, 7].
        True = genuine/valid pollutant observation.
        False = invalid/questionable pollutant observation.
    """

    original_shape = X.shape

    values = (
        X.numpy()
        .reshape(
            -1,
            original_shape[-1]
        )
    )

    scaled = scaler.transform(values)

    scaled = (
        scaled
        .reshape(original_shape)
        .astype(np.float32)
    )

    # ------------------------------------------------------
    # Mask invalid pollutant inputs AFTER scaling
    # ------------------------------------------------------
    #
    # We deliberately do this after scaling.
    #
    # StandardScaler makes the training mean approximately 0,
    # so 0.0 acts as a neutral value rather than pretending that
    # an invalid observation is a real low/high pollutant value.
    #
    # Only the first 7 pollutant features are affected.
    # ------------------------------------------------------

    if (
        MASK_INVALID_POLLUTANTS
        and validity_mask is not None
    ):

        if validity_mask.shape != (
            original_shape[0],
            original_shape[1],
            original_shape[2],
            POLLUTANT_FEATURE_COUNT
        ):
            raise ValueError(
                "Validity mask shape does not match X pollutant "
                "dimensions. "
                f"X={tuple(original_shape)}, "
                f"mask={tuple(validity_mask.shape)}."
            )

        mask_numpy = (
            validity_mask
            .numpy()
            .astype(bool)
        )

        pollutant_scaled = (
            scaled[..., :POLLUTANT_FEATURE_COUNT]
        )

        pollutant_scaled[
            ~mask_numpy
        ] = 0.0

        scaled[
            ...,
            :POLLUTANT_FEATURE_COUNT
        ] = pollutant_scaled

        invalid_count = int(
            (~mask_numpy).sum()
        )

        total_pollutant_count = int(
            mask_numpy.size
        )

        valid_count = (
            total_pollutant_count
            - invalid_count
        )

        validity_percent = (
            valid_count
            / total_pollutant_count
            * 100.0
        )

        print(
            "\nInvalid pollutant inputs neutralized:"
        )

        print(
            f"  Invalid: {invalid_count:,} / "
            f"{total_pollutant_count:,}"
        )

        print(
            f"  Valid:   {valid_count:,} / "
            f"{total_pollutant_count:,}"
        )

        print(
            f"  Validity: {validity_percent:.2f}%"
        )

        print(
            "  Replacement value after scaling: 0.0"
        )

    elif MASK_INVALID_POLLUTANTS:
        raise ValueError(
            "MASK_INVALID_POLLUTANTS=True but no "
            "validity_mask was supplied."
        )

    return torch.tensor(
        scaled,
        dtype=torch.float32
    )


# ==========================================================
# FIT Y STANDARD SCALER
# ==========================================================
#
# IMPORTANT:
#
# Y is scaled because the model trains against the
# standardized target.
#
# The scaler is fitted ONLY on Y_train.
#
# ==========================================================

def fit_y_scaler(
    Y_train
):

    print("\n" + "=" * 70)

    print(
        "FITTING Y STANDARD SCALER"
    )

    print("=" * 70)


    values = (
        Y_train
        .numpy()
        .reshape(
            -1,
            1
        )
    )


    scaler = StandardScaler()


    scaler.fit(
        values
    )


    print(
        "Target mean:",
        scaler.mean_[0]
    )

    print(
        "Target std:",
        scaler.scale_[0]
    )


    print(
        "\nY scaler fitted using TRAINING DATA ONLY."
    )


    return scaler


# ==========================================================
# APPLY Y SCALER
# ==========================================================

def apply_y_scaler(
    Y,
    scaler
):

    original_shape = Y.shape


    values = (
        Y.numpy()
        .reshape(
            -1,
            1
        )
    )


    scaled = scaler.transform(
        values
    )


    scaled = (
        scaled
        .reshape(
            original_shape
        )
        .astype(
            np.float32
        )
    )


    return torch.tensor(
        scaled,
        dtype=torch.float32
    )


# ==========================================================
# TARGET SCALING CHECK
# ==========================================================

def target_scaling_diagnostics(
    Y_train_original,
    Y_train_scaled,
    Y_val_scaled,
    Y_test_scaled
):

    print("\n" + "=" * 70)

    print(
        "TARGET SCALING DIAGNOSTICS"
    )

    print("=" * 70)


    print(
        "Original Y train mean:",
        Y_train_original.mean().item()
    )

    print(
        "Original Y train std:",
        Y_train_original.std().item()
    )

    print(
        "Original Y train min:",
        Y_train_original.min().item()
    )

    print(
        "Original Y train max:",
        Y_train_original.max().item()
    )


    print(
        "\nScaled Y train mean:",
        Y_train_scaled.mean().item()
    )

    print(
        "Scaled Y train std:",
        Y_train_scaled.std().item()
    )

    print(
        "Scaled Y train min:",
        Y_train_scaled.min().item()
    )

    print(
        "Scaled Y train max:",
        Y_train_scaled.max().item()
    )


    print(
        "\nScaled Y validation mean:",
        Y_val_scaled.mean().item()
    )

    print(
        "Scaled Y validation std:",
        Y_val_scaled.std().item()
    )


    print(
        "\nScaled Y test mean:",
        Y_test_scaled.mean().item()
    )

    print(
        "Scaled Y test std:",
        Y_test_scaled.std().item()
    )


# ==========================================================
# X SCALING DIAGNOSTICS
# ==========================================================

def x_scaling_diagnostics(
    X_train,
    X_val,
    X_test
):

    print("\n" + "=" * 70)

    print(
        "X SCALING DIAGNOSTICS"
    )

    print("=" * 70)


    datasets = [

        (
            "TRAIN",
            X_train
        ),

        (
            "VALIDATION",
            X_val
        ),

        (
            "TEST",
            X_test
        )
    ]


    for name, X in datasets:

        print(
            f"\n{name}"
        )

        print(
            "-" * 70
        )

        print(
            "Shape:",
            X.shape
        )

        print(
            "Mean:",
            X.mean().item()
        )

        print(
            "Std:",
            X.std().item()
        )

        print(
            "Min:",
            X.min().item()
        )

        print(
            "Max:",
            X.max().item()
        )

        print(
            "NaNs:",
            torch.isnan(X).sum().item()
        )

        print(
            "Infs:",
            torch.isinf(X).sum().item()
        )


        if torch.isnan(
            X
        ).any():

            raise ValueError(
                f"{name} contains NaNs."
            )


        if torch.isinf(
            X
        ).any():

            raise ValueError(
                f"{name} contains Infs."
            )


# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation(
    X_train,
    X_val,
    X_test,

    Y_train,
    Y_val,
    Y_test
):

    print("\n" + "=" * 70)

    print(
        "FINAL PREPROCESSING VALIDATION"
    )

    print("=" * 70)


    tensors = [

        (
            "X_train",
            X_train
        ),

        (
            "X_val",
            X_val
        ),

        (
            "X_test",
            X_test
        ),

        (
            "Y_train",
            Y_train
        ),

        (
            "Y_val",
            Y_val
        ),

        (
            "Y_test",
            Y_test
        )
    ]


    for name, tensor in tensors:

        if torch.isnan(
            tensor
        ).any():

            raise ValueError(
                f"{name} contains NaNs."
            )


        if torch.isinf(
            tensor
        ).any():

            raise ValueError(
                f"{name} contains Infs."
            )


        print(
            f"{name:10s}: "
            f"{tuple(tensor.shape)} "
            f"PASS"
        )


# ==========================================================
# COMPLETE PIPELINE
# ==========================================================

def prepare_model_data(
    path=QUALITY_DATASET_PATH
):

    # ======================================================
    # LOAD
    # ======================================================

    (
        X,
        Y,
        validity_mask,
        sequence_indices,
        station_record_sequences

    ) = load_quality_controlled_data(path)


    # ======================================================
    # VALIDATE
    # ======================================================

    validate_dataset(
        X,
        Y,
        validity_mask
    )


    # ======================================================
    # CHRONOLOGICAL SPLIT
    # ======================================================

    (
        X_train_raw,
        X_val_raw,
        X_test_raw,

        Y_train_raw,
        Y_val_raw,
        Y_test_raw,

        mask_train,
        mask_val,
        mask_test,

        indices_train,
        indices_val,
        indices_test

    ) = chronological_split(
        X,
        Y,
        validity_mask,
        sequence_indices
    )


    # ======================================================
    # SPLIT STATION RECORD SEQUENCES
    #
    # Keep these aligned with X/Y. Do not scale them.
    # ======================================================

    total_samples = X.shape[0]

    train_end = int(
        total_samples * TRAIN_RATIO
    )

    val_end = (
        train_end
        +
        int(
            total_samples * VAL_RATIO
        )
    )

    try:
        station_records_train = station_record_sequences[:train_end]
        station_records_val = station_record_sequences[train_end:val_end]
        station_records_test = station_record_sequences[val_end:]
    except (TypeError, KeyError, IndexError):
        raise ValueError(
            "station_record_sequences must support sample-wise slicing."
        )

    if len(station_records_train) != len(X_train_raw):
        raise ValueError(
            "Training station records are not aligned with X_train."
        )

    if len(station_records_val) != len(X_val_raw):
        raise ValueError(
            "Validation station records are not aligned with X_val."
        )

    if len(station_records_test) != len(X_test_raw):
        raise ValueError(
            "Test station records are not aligned with X_test."
        )

    print("\n" + "=" * 70)
    print("STATION RECORD SEQUENCES")
    print("=" * 70)
    print("Total:", len(station_record_sequences))
    print("Training:", len(station_records_train))
    print("Validation:", len(station_records_val))
    print("Test:", len(station_records_test))
    print("Alignment: PASS")


    # ======================================================
    # X SCALER
    # ======================================================

    x_scaler = fit_x_scaler(
        X_train_raw,
        mask_train
    )


    # ======================================================
    # SCALE X
    # ======================================================

    X_train = apply_x_scaler(
        X_train_raw,
        x_scaler,
        mask_train
    )

    X_val = apply_x_scaler(
        X_val_raw,
        x_scaler,
        mask_val
    )

    X_test = apply_x_scaler(
        X_test_raw,
        x_scaler,
        mask_test
    )


    # ======================================================
    # Y SCALER
    # ======================================================

    y_scaler = fit_y_scaler(
        Y_train_raw
    )


    # ======================================================
    # SCALE Y
    # ======================================================

    Y_train = apply_y_scaler(
        Y_train_raw,
        y_scaler
    )

    Y_val = apply_y_scaler(
        Y_val_raw,
        y_scaler
    )

    Y_test = apply_y_scaler(
        Y_test_raw,
        y_scaler
    )


    # ======================================================
    # DIAGNOSTICS
    # ======================================================

    x_scaling_diagnostics(
        X_train,
        X_val,
        X_test
    )

    target_scaling_diagnostics(
        Y_train_raw,
        Y_train,
        Y_val,
        Y_test
    )


    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    final_validation(
        X_train,
        X_val,
        X_test,

        Y_train,
        Y_val,
        Y_test
    )


    # ======================================================
    # FINAL REPORT
    # ======================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape)
    print("Y_val:", Y_val.shape)
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)
    print("Station records:", len(station_record_sequences))

    print("\nX scaler:", "StandardScaler")
    print("Y scaler:", "StandardScaler")
    print("Scaler fitting:", "TRAINING DATA ONLY")
    print(
        "Invalid pollutant inputs:",
        "replaced with 0.0 after scaling"
        if MASK_INVALID_POLLUTANTS
        else "kept unchanged"
    )

    print("\nTarget used by model:", "STANDARDIZED Y")
    print("Final metrics:", "must inverse-transform Y first")

    print("=" * 70)


    # ======================================================
    # RETURN
    # ======================================================

    return {

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test,

        "Y_train_original": Y_train_raw,
        "Y_val_original": Y_val_raw,
        "Y_test_original": Y_test_raw,

        "mask_train": mask_train,
        "mask_val": mask_val,
        "mask_test": mask_test,

        "indices_train": indices_train,
        "indices_val": indices_val,
        "indices_test": indices_test,

        # FULL sequence-aligned records.
        # This is the key train.py was missing.
        "station_record_sequences": station_record_sequences,

        # Split records are also provided for convenience.
        "station_records_train": station_records_train,
        "station_records_val": station_records_val,
        "station_records_test": station_records_test,

        # Compatibility keys required by train.py
        "station_record_sequences_train": station_records_train,
        "station_record_sequences_val": station_records_val,
        "station_record_sequences_test": station_records_test,

        "x_scaler": x_scaler,
        "y_scaler": y_scaler,

        "feature_columns": FEATURE_COLUMNS
    }


# ==========================================================
# DIRECT TEST
# ==========================================================

if __name__ == "__main__":

    data = prepare_model_data()


    print("\n" + "=" * 70)

    print(
        "PREPROCESSING TEST PASSED"
    )

    print("=" * 70)


    print(
        "\nReady for train.py."
    )


    print(
        "X_train:",
        data["X_train"].shape
    )

    print(
        "Y_train:",
        data["Y_train"].shape
    )

    print(
        "X_val:",
        data["X_val"].shape
    )

    print(
        "Y_val:",
        data["Y_val"].shape
    )

    print(
        "X_test:",
        data["X_test"].shape
    )

    print(
        "Y_test:",
        data["Y_test"].shape
    )

    print(
        "Station record sequences:",
        len(data["station_record_sequences"])
    )

    print(
        "Station record alignment: PASS"
    )



    print(
        "\nBoth X and Y are standardized."
    )

    print(
        "Y scaler is available for "
        "inverse transformation."
    )

    print("=" * 70)

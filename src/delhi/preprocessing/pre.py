# ==========================================================
# pre.py
#
# NEW DELHI 2025 DATASET PREPROCESSING
#
# MULTI-HORIZON VERSION
#
# Input:
#     Past 24 hours
#
# Target:
#     Next 24 hours
#
# Output:
#     X : [samples, 24, 6, 8]
#     Y : [samples, 6, 24]
#
# The 24 future timestamps are predicted in one sample.
#
# For evaluation:
#     12-hour results = Y[:, :, :12]
#     24-hour results = Y[:, :, :24]
#
# Graph metadata is kept separately from scaled neural
# network features.
#
# Wind direction is NOT required because the new dataset
# contains wind speed but no wind direction.
#
# Graph formulation:
#
#     Sij = Vi * exp(-k * dij)
#
# where:
#     Vi  = source-station wind speed
#     dij = Haversine distance
#     k   = global decay constant
# ==========================================================


import os
import numpy as np
import pandas as pd
import torch
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON, TARGET_COLUMN

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "delhi-weather-aqi-2025.csv")
DATASET_PATH = DEFAULT_DATASET_PATH if os.path.exists(DEFAULT_DATASET_PATH) else "data/delhi-weather-aqi-2025.csv"


# ----------------------------------------------------------
# INPUT HISTORY
# ----------------------------------------------------------

SEQUENCE_LENGTH = SEQUENCE_LENGTH

# ----------------------------------------------------------
# MULTI-HORIZON FORECAST
#
# The model will be trained to predict the next 24 hours.
# ----------------------------------------------------------

PREDICTION_HORIZON = PREDICTION_HORIZON

# ----------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# ----------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Purge overlapping windows at split boundaries.
# For H historical hours and F forecast hours, adjacent
# windows can share observations across the boundary.
PURGE_GAP = SEQUENCE_LENGTH + PREDICTION_HORIZON - 1

# ----------------------------------------------------------
# TARGET
# ----------------------------------------------------------

TARGET_COLUMN = TARGET_COLUMN

# ----------------------------------------------------------
# STATION / METADATA COLUMNS
# ----------------------------------------------------------

STATION_COLUMN = "location"

LATITUDE_COLUMN = "lat"
LONGITUDE_COLUMN = "lon"

DATE_COLUMN = "date_ist"
TIME_COLUMN = "time_ist"


# ==========================================================
# FEATURE DEFINITIONS
# ==========================================================

# This order MUST remain identical to the order used by
# NodeFeatures.py / model.py.

FEATURE_COLUMNS = [

    "pm2_5",
    "pm10",
    "co",
    "no2",
    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
]


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [

    DATE_COLUMN,
    TIME_COLUMN,
    STATION_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,

    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",

    "aqi_index",

    TARGET_COLUMN,
    "pm10",
    "co",
    "no2"
]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    print("\n" + "=" * 70)
    print("LOADING NEW DELHI DATASET")
    print("=" * 70)

    actual_path = DATASET_PATH
    if not os.path.exists(actual_path):
        import glob
        candidates = [
            os.path.join(BASE_DIR, "data", "delhi-weather-aqi-2025.csv"),
            os.path.join("data", "delhi-weather-aqi-2025.csv"),
            "delhi-weather-aqi-2025.csv"
        ]
        candidates.extend(glob.glob("**/delhi-weather-aqi-2025.csv", recursive=True))
        if os.path.exists("/kaggle"):
            candidates.extend(glob.glob("/kaggle/**/delhi-weather-aqi-2025.csv", recursive=True))
        if os.path.exists("/content"):
            candidates.extend(glob.glob("/content/**/delhi-weather-aqi-2025.csv", recursive=True))
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c) and os.path.getsize(c) > 100000:
                actual_path = c
                print(f"Auto-located dataset at: {actual_path}")
                break

    if not os.path.exists(actual_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Checked alternate locations but delhi-weather-aqi-2025.csv was not found."
        )

    df = pd.read_csv(actual_path)

    # ------------------------------------------------------
    # IMPORTANT:
    # Remove accidental whitespace from CSV headers.
    #
    # Example:
    #     " pm2_5 " -> "pm2_5"
    # ------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    print("Dataset shape:", df.shape)

    print("\nColumns:")

    for column in df.columns:

        print(f"  - {column}")

    return df


# ==========================================================
# COLUMN VALIDATION
# ==========================================================

def validate_columns(df):

    missing = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns

    ]

    if missing:

        raise ValueError(

            "Missing required columns:\n"

            + "\n".join(

                f"  - {column}"

                for column in missing

            )

        )

    print("\nRequired-column check: PASS")


# ==========================================================
# TIMESTAMP
# ==========================================================

def create_timestamp(df):

    df = df.copy()

    df["Timestamp"] = pd.to_datetime(

        df[DATE_COLUMN].astype(str)

        + " "

        + df[TIME_COLUMN].astype(str),

        dayfirst=True,

        errors="coerce"

    )

    invalid = df["Timestamp"].isna().sum()

    if invalid > 0:

        raise ValueError(
            f"Found {invalid} invalid timestamps."
        )

    # ------------------------------------------------------
    # CYCLICAL TEMPORAL & CALENDAR ENCODINGS
    # ------------------------------------------------------
    # Atmospheric boundary layer dynamics, traffic peaks,
    # solar radiation, and seasonal inversions are cyclical functions of
    # the hour (24h), day of week (7d), and month (12m).
    # Sin/Cos transformations preserve continuous distance (e.g. 23:00 -> 00:00).
    hours = df["Timestamp"].dt.hour.to_numpy()
    days = df["Timestamp"].dt.dayofweek.to_numpy()
    months = df["Timestamp"].dt.month.to_numpy()

    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["day_sin"] = np.sin(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["day_cos"] = np.cos(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["month_sin"] = np.sin(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)

    df = df.sort_values(

        ["Timestamp", STATION_COLUMN]

    ).reset_index(drop=True)

    return df



# ==========================================================
# NUMERIC VALIDATION
# ==========================================================

def validate_numeric_columns(df):

    numeric_columns = (

        FEATURE_COLUMNS

        + [

            LATITUDE_COLUMN,
            LONGITUDE_COLUMN

        ]

    )

    df = df.copy()

    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"

        )

    invalid = (

        df[numeric_columns]

        .isna()

        .sum()

    )

    invalid = invalid[invalid > 0]

    if len(invalid) > 0:

        raise ValueError(

            "Invalid numeric values found:\n"

            + invalid.to_string()

        )

    values = (

        df[numeric_columns]

        .to_numpy(dtype=np.float64)

    )

    if not np.isfinite(values).all():

        raise ValueError(
            "NaN or infinite numeric values detected."
        )

    print("Numeric-value validation: PASS")

    return df


# ==========================================================
# WIND VALIDATION
# ==========================================================

def validate_wind_information(df):

    """
    The new dataset contains windspeed_kph but no wind
    direction.

    We intentionally DO NOT fabricate Wind_U / Wind_V.

    The current graph therefore uses:

        Sij = Vi * exp(-k * dij)
    """

    if "windspeed_kph" not in df.columns:

        raise ValueError(
            "The dataset must contain 'windspeed_kph'."
        )

    if df["windspeed_kph"].isna().any():

        raise ValueError(
            "windspeed_kph contains missing values."
        )

    if (df["windspeed_kph"] < 0).any():

        raise ValueError(
            "windspeed_kph contains negative values."
        )

    print("\nWind information check: PASS")
    print("Using wind speed: windspeed_kph")
    print("Wind direction: not available")

    print(
        "Graph formulation: "
        "Sij = Vi * exp(-k * dij)"
    )

    return "windspeed_kph"


# ==========================================================
# DATASET INSPECTION
# ==========================================================

def inspect_dataset(df):

    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print("Rows:", len(df))

    stations = sorted(

        df[STATION_COLUMN]

        .dropna()

        .unique()

        .tolist()

    )

    print("Number of stations:", len(stations))

    print("\nStations:")

    for index, station in enumerate(stations):

        print(
            f"  {index}: {station}"
        )

    print("\nRows per station:")

    print(
        df.groupby(
            STATION_COLUMN
        ).size()
    )

    print("\nMissing values:")

    missing = (

        df[REQUIRED_COLUMNS]

        .isna()

        .sum()

    )

    print(
        missing[missing > 0]
    )

    coordinate_counts = (

        df.groupby(STATION_COLUMN)[

            [
                LATITUDE_COLUMN,
                LONGITUDE_COLUMN
            ]

        ]

        .nunique()

    )

    if (coordinate_counts > 1).any().any():

        raise ValueError(
            "Station coordinates are not fixed."
        )

    print(
        "\nStation-coordinate consistency: PASS"
    )

    duplicates = df.duplicated(

        subset=[

            "Timestamp",
            STATION_COLUMN

        ]

    ).sum()

    print(
        "Duplicate station/timestamp rows:",
        duplicates
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate station/timestamp rows detected."
        )

    return stations


# ==========================================================
# STATION MAPPING
# ==========================================================

def create_station_mapping(df):

    stations = sorted(

        df[STATION_COLUMN]

        .unique()

        .tolist()

    )

    station_to_id = {

        station: index

        for index, station in enumerate(
            stations
        )

    }

    df = df.copy()

    df["Station_ID"] = (

        df[STATION_COLUMN]

        .map(station_to_id)

        .astype(int)

    )

    print("\n" + "=" * 70)
    print("STATION MAPPING")
    print("=" * 70)

    for station, station_id in station_to_id.items():

        row = df[
            df["Station_ID"] == station_id
        ].iloc[0]

        print(

            f"{station_id:02d} | "

            f"{station} | "

            f"lat={row[LATITUDE_COLUMN]:.6f} | "

            f"lon={row[LONGITUDE_COLUMN]:.6f}"

        )

    return df, stations, station_to_id


# ==========================================================
# HOURLY STRUCTURE
# ==========================================================

def verify_hourly_structure(df, stations):

    print("\n" + "=" * 70)
    print("VERIFYING HOURLY MULTI-STATION STRUCTURE")
    print("=" * 70)

    timestamps = (

        df["Timestamp"]

        .drop_duplicates()

        .sort_values()

        .reset_index(drop=True)

    )

    expected_station_count = len(stations)

    station_counts = (

        df.groupby("Timestamp")[STATION_COLUMN]

        .nunique()

    )

    incomplete = (

        station_counts != expected_station_count

    )

    print(
        "Unique timestamps:",
        len(timestamps)
    )

    print(
        "Expected stations per hour:",
        expected_station_count
    )

    print(
        "Incomplete timestamps:",
        incomplete.sum()
    )

    if incomplete.any():

        print(
            station_counts[incomplete].head(10)
        )

        raise ValueError(
            "Some timestamps do not contain all stations."
        )

    differences = (

        timestamps.diff()

        .dropna()

    )

    hourly_gaps = (

        differences
        != pd.Timedelta(hours=1)

    )

    print(
        "Hourly gaps:",
        hourly_gaps.sum()
    )

    if hourly_gaps.any():

        raise ValueError(
            "The timestamp series contains hourly gaps."
        )

    print("Hourly structure: PASS")

    return timestamps


# ==========================================================
# STATION RECORD
# ==========================================================

def create_station_record(row):

    """
    Metadata used by GraphBuilder / SpatialEncoder.

    This is intentionally separate from the scaled neural
    network input.

    Wind_Speed comes directly from windspeed_kph.
    """

    return {

        "Station_ID":
            int(row["Station_ID"]),

        "Station":
            str(row[STATION_COLUMN]),

        "Latitude":
            float(row[LATITUDE_COLUMN]),

        "Longitude":
            float(row[LONGITUDE_COLUMN]),

        "Wind_Speed":
            float(row["windspeed_kph"]),

        "Temperature":
            float(row["temp_c"]),

        "Humidity":
            float(row["humidity"]),

        "Pressure":
            float(row["pressure_mb"]),

        "PM2.5":
            float(row["pm2_5"]),

        "PM10":
            float(row["pm10"]),

        "CO":
            float(row["co"]),

        "NO2":
            float(row["no2"])

    }


# ==========================================================
# BUILD STATION ARRAYS
# ==========================================================

def construct_station_arrays(
    df,
    stations
):

    print("\n" + "=" * 70)
    print("CONSTRUCTING STATION ARRAYS")
    print("=" * 70)

    timestamps = (

        df["Timestamp"]

        .drop_duplicates()

        .sort_values()

        .tolist()

    )

    num_times = len(timestamps)

    num_stations = len(stations)

    num_features = len(FEATURE_COLUMNS)

    timestamp_to_index = {

        timestamp: index

        for index, timestamp in enumerate(
            timestamps
        )

    }

    station_to_index = {

        station: index

        for index, station in enumerate(
            stations
        )

    }

    # ------------------------------------------------------
    # X
    #
    # [time, station, feature]
    # ------------------------------------------------------

    X = np.zeros(

        (
            num_times,
            num_stations,
            num_features
        ),

        dtype=np.float32

    )

    # ------------------------------------------------------
    # Y
    #
    # Raw target remains one value per station per time.
    #
    # [time, station, 1]
    # ------------------------------------------------------

    Y = np.zeros(

        (
            num_times,
            num_stations,
            1
        ),

        dtype=np.float32

    )

    for row in df.itertuples(index=False):

        row_dict = row._asdict()

        timestamp = row_dict["Timestamp"]

        station = row_dict[STATION_COLUMN]

        time_index = (
            timestamp_to_index[timestamp]
        )

        station_index = (
            station_to_index[station]
        )

        for feature_index, column in enumerate(
            FEATURE_COLUMNS
        ):

            X[
                time_index,
                station_index,
                feature_index
            ] = float(
                row_dict[column]
            )

        Y[
            time_index,
            station_index,
            0
        ] = float(
            row_dict[TARGET_COLUMN]
        )

    print(
        "Raw X shape:",
        X.shape
    )

    print(
        "Raw Y shape:",
        Y.shape
    )

    return X, Y, timestamps


# ==========================================================
# STATION RECORD LOOKUP
# ==========================================================

def build_station_record_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        row_dict = row._asdict()

        timestamp = row_dict["Timestamp"]

        station_id = int(
            row_dict["Station_ID"]
        )

        if timestamp not in lookup:

            lookup[timestamp] = {}

        lookup[timestamp][station_id] = (

            create_station_record(
                row_dict
            )

        )

    return lookup


# ==========================================================
# CREATE MULTI-HORIZON SEQUENCES
# ==========================================================

def create_sequences(
    X,
    Y,
    timestamps,
    df
):

    print("\n" + "=" * 70)
    print("CREATING MULTI-HORIZON TEMPORAL SEQUENCES")
    print("=" * 70)

    num_times = X.shape[0]

    num_stations = X.shape[1]

    station_lookup = (
        build_station_record_lookup(df)
    )

    X_sequences = []

    Y_sequences = []

    station_record_sequences = []

    sequence_timestamps = []

    # ------------------------------------------------------
    # Number of valid windows
    #
    # Example:
    #
    # input:
    #   t ... t+23
    #
    # target:
    #   t+24 ... t+47
    #
    # ------------------------------------------------------

    max_start = (

        num_times
        - SEQUENCE_LENGTH
        - PREDICTION_HORIZON
        + 1

    )

    if max_start <= 0:

        raise ValueError(

            "Not enough timestamps to create "
            "the requested input/forecast windows.\n"

            f"Available timestamps: {num_times}\n"

            f"Input sequence length: "
            f"{SEQUENCE_LENGTH}\n"

            f"Prediction horizon: "
            f"{PREDICTION_HORIZON}"

        )

    for start in range(max_start):

        # --------------------------------------------------
        # INPUT WINDOW
        #
        # [24, stations, features]
        # --------------------------------------------------

        input_end = (
            start + SEQUENCE_LENGTH
        )

        X_window = X[
            start:input_end
        ]

        # --------------------------------------------------
        # TARGET WINDOW
        #
        # Immediately after the input.
        #
        # [24, stations, 1]
        # --------------------------------------------------

        target_start = input_end

        target_end = (

            target_start
            + PREDICTION_HORIZON

        )

        Y_window = Y[
            target_start:target_end
        ]

        # --------------------------------------------------
        # Reorganize target:
        #
        # Current:
        #   [24, stations, 1]
        #
        # Desired:
        #   [stations, 24]
        #
        # This makes it directly compatible with:
        #
        #   prediction output [B, N, 24]
        # --------------------------------------------------

        Y_target = (

            Y_window[:, :, 0]

            .T

        )

        # --------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------

        if Y_target.shape != (
            num_stations,
            PREDICTION_HORIZON
        ):

            raise ValueError(

                "Unexpected multi-horizon target shape: "

                f"{Y_target.shape}. "

                "Expected: "

                f"({num_stations}, "
                f"{PREDICTION_HORIZON})"

            )

        # --------------------------------------------------
        # STATION RECORDS
        #
        # These remain ONLY for the 24-hour INPUT WINDOW.
        #
        # Graph construction operates on the historical
        # station records that are actually available to
        # the model.
        # --------------------------------------------------

        records_for_sample = []

        for time_index in range(

            start,
            input_end

        ):

            timestamp = timestamps[
                time_index
            ]

            if timestamp not in station_lookup:

                raise ValueError(

                    f"Missing station records "
                    f"for {timestamp}"

                )

            current_records = []

            for station_id in range(
                num_stations
            ):

                if (

                    station_id
                    not in
                    station_lookup[timestamp]

                ):

                    raise ValueError(

                        f"Missing station "
                        f"{station_id} "
                        f"at {timestamp}"

                    )

                current_records.append(

                    station_lookup[
                        timestamp
                    ][station_id]

                )

            records_for_sample.append(
                current_records
            )

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        X_sequences.append(
            X_window
        )

        Y_sequences.append(
            Y_target
        )

        station_record_sequences.append(
            records_for_sample
        )

        # --------------------------------------------------
        # TIMESTAMP INFORMATION
        # --------------------------------------------------

        sequence_timestamps.append({

            "input_start":
                timestamps[start],

            "input_end":
                timestamps[input_end - 1],

            "target_start":
                timestamps[target_start],

            "target_end":
                timestamps[target_end - 1],

            # Keep the complete forecast timestamps
            # available for later evaluation/plotting.
            "target":
                timestamps[
                    target_start:target_end
                ]

        })

    X_sequences = np.asarray(

        X_sequences,

        dtype=np.float32

    )

    Y_sequences = np.asarray(

        Y_sequences,

        dtype=np.float32

    )

    print(
        "X sequences:",
        X_sequences.shape
    )

    print(
        "Y sequences:",
        Y_sequences.shape
    )

    print(
        "Station-record sequences:",
        len(station_record_sequences)
    )

    # ------------------------------------------------------
    # Explicit shape report
    # ------------------------------------------------------

    print("\nMulti-horizon target structure:")

    print(
        "  Input X :",
        f"[samples, {SEQUENCE_LENGTH}, "
        f"{num_stations}, {X.shape[-1]}]"
    )

    print(
        "  Target Y:",
        f"[samples, {num_stations}, "
        f"{PREDICTION_HORIZON}]"
    )

    print(
        f"  Forecast hours: 1 -> "
        f"{PREDICTION_HORIZON}"
    )

    return (

        X_sequences,

        Y_sequences,

        station_record_sequences,

        sequence_timestamps

    )


# ==========================================================
# CHRONOLOGICAL SPLIT
# ==========================================================

def split_sequences(
    X,
    Y,
    station_records,
    sequence_timestamps
):

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / VAL / TEST SPLIT")
    print("=" * 70)

    total = len(X)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_boundary = (

        train_end

        + int(total * VAL_RATIO)

    )

    # Purge a boundary gap so the last train window and the
    # first validation window cannot share any timestamps.
    val_start = train_end + PURGE_GAP
    test_start = val_boundary + PURGE_GAP

    if val_start >= val_boundary:
        raise ValueError(
            "PURGE_GAP is too large for the validation split."
        )

    if test_start > total:
        raise ValueError(
            "PURGE_GAP is too large for the test split."
        )

    X_train = X[:train_end]
    Y_train = Y[:train_end]
    records_train = station_records[:train_end]
    timestamps_train = sequence_timestamps[:train_end]

    X_val = X[val_start:val_boundary]
    Y_val = Y[val_start:val_boundary]
    records_val = station_records[val_start:val_boundary]
    timestamps_val = sequence_timestamps[val_start:val_boundary]

    X_test = X[test_start:]
    Y_test = Y[test_start:]
    records_test = station_records[test_start:]
    timestamps_test = sequence_timestamps[test_start:]

    print(
        "Train samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print(
        "Purge gap (windows):",
        PURGE_GAP
    )

    # ------------------------------------------------------
    # NO CROSS-SPLIT TEMPORAL OVERLAP CHECK
    # ------------------------------------------------------

    if timestamps_train and timestamps_val:

        if (
            timestamps_train[-1]["target_end"]
            >=
            timestamps_val[0]["input_start"]
        ):
            raise ValueError(
                "Train/validation windows still overlap in time."
            )

    if timestamps_val and timestamps_test:

        if (
            timestamps_val[-1]["target_end"]
            >=
            timestamps_test[0]["input_start"]
        ):
            raise ValueError(
                "Validation/test windows still overlap in time."
            )

    print(
        "Cross-split temporal overlap: PASS"
    )

    return (

        X_train,
        Y_train,
        records_train,
        timestamps_train,

        X_val,
        Y_val,
        records_val,
        timestamps_val,

        X_test,
        Y_test,
        records_test,
        timestamps_test

    )


# ==========================================================
# FIT SCALERS ON TRAIN ONLY
# ==========================================================

def fit_scalers(
    X_train,
    Y_train
):

    print("\n" + "=" * 70)
    print("FITTING SCALERS ON TRAINING DATA ONLY")
    print("=" * 70)

    num_features = X_train.shape[-1]

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()

    # ------------------------------------------------------
    # Feature scaler
    # ------------------------------------------------------

    feature_scaler.fit(

        X_train.reshape(
            -1,
            num_features
        )

    )

    # ------------------------------------------------------
    # Target scaler
    #
    # Y_train is:
    #
    # [samples, stations, 24]
    #
    # Every PM2.5 target value is scaled using
    # the training target distribution.
    # ------------------------------------------------------

    target_scaler.fit(

        Y_train.reshape(
            -1,
            1
        )

    )

    print(
        "Feature scaler: fitted"
    )

    print(
        "Target scaler: fitted"
    )

    return (
        feature_scaler,
        target_scaler
    )


# ==========================================================
# SCALE X
# ==========================================================

def scale_X(
    X,
    scaler
):

    original_shape = X.shape

    num_features = X.shape[-1]

    scaled = scaler.transform(

        X.reshape(
            -1,
            num_features
        )

    )

    return scaled.reshape(
        original_shape
    ).astype(np.float32)


# ==========================================================
# SCALE Y
# ==========================================================

def scale_Y(
    Y,
    scaler
):

    original_shape = Y.shape

    scaled = scaler.transform(

        Y.reshape(
            -1,
            1
        )

    )

    return scaled.reshape(
        original_shape
    ).astype(np.float32)


# ==========================================================
# CONVERT TO TENSORS
# ==========================================================

def convert_to_tensors(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test
):

    return (

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            Y_train,
            dtype=torch.float32
        ),

        torch.tensor(
            X_val,
            dtype=torch.float32
        ),

        torch.tensor(
            Y_val,
            dtype=torch.float32
        ),

        torch.tensor(
            X_test,
            dtype=torch.float32
        ),

        torch.tensor(
            Y_test,
            dtype=torch.float32
        )

    )


# ==========================================================
# FINAL VALIDATION
# ==========================================================

def validate_final_data(
    X_train,
    Y_train,
    records_train,

    X_val,
    Y_val,
    records_val,

    X_test,
    Y_test,
    records_test
):

    print("\n" + "=" * 70)
    print("FINAL PREPROCESSING VALIDATION")
    print("=" * 70)

    splits = [

        (
            "TRAIN",
            X_train,
            Y_train,
            records_train
        ),

        (
            "VALIDATION",
            X_val,
            Y_val,
            records_val
        ),

        (
            "TEST",
            X_test,
            Y_test,
            records_test
        )

    ]

    for (

        name,
        X,
        Y,
        records

    ) in splits:

        # --------------------------------------------------
        # X
        # --------------------------------------------------

        if X.ndim != 4:

            raise ValueError(

                f"{name}: X must be "
                "[samples,time,stations,features]. "

                f"Got {X.shape}"

            )

        # --------------------------------------------------
        # Y
        #
        # Multi-horizon:
        #
        # [samples, stations, horizon]
        # --------------------------------------------------

        if Y.ndim != 3:

            raise ValueError(

                f"{name}: Y must be "
                "[samples,stations,horizon]. "

                f"Got {Y.shape}"

            )

        # --------------------------------------------------
        # Sample count
        # --------------------------------------------------

        if X.shape[0] != Y.shape[0]:

            raise ValueError(
                f"{name}: X/Y sample mismatch."
            )

        # --------------------------------------------------
        # Station count
        # --------------------------------------------------

        if X.shape[2] != Y.shape[1]:

            raise ValueError(
                f"{name}: station mismatch."
            )

        # --------------------------------------------------
        # Input sequence length
        # --------------------------------------------------

        if X.shape[1] != SEQUENCE_LENGTH:

            raise ValueError(

                f"{name}: expected "
                f"{SEQUENCE_LENGTH} input timesteps."

            )

        # --------------------------------------------------
        # Number of input features
        # --------------------------------------------------

        if X.shape[-1] != len(
            FEATURE_COLUMNS
        ):

            raise ValueError(

                f"{name}: expected "
                f"{len(FEATURE_COLUMNS)} features."

            )

        # --------------------------------------------------
        # Forecast horizon
        # --------------------------------------------------

        if Y.shape[-1] != PREDICTION_HORIZON:

            raise ValueError(

                f"{name}: expected "
                f"target horizon "
                f"{PREDICTION_HORIZON}. "

                f"Got {Y.shape[-1]}."

            )

        # --------------------------------------------------
        # Station records
        # --------------------------------------------------

        if len(records) != X.shape[0]:

            raise ValueError(

                f"{name}: "
                "station-record mismatch."

            )

        print(

            f"{name}: PASS | "

            f"X={tuple(X.shape)} | "

            f"Y={tuple(Y.shape)} | "

            f"records={len(records)}"

        )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def prepare_model_data():

    print("\n" + "=" * 70)
    print("PREPARING MODEL DATA")
    print("=" * 70)

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------

    df = load_dataset()

    # ------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------

    validate_columns(df)

    df = create_timestamp(df)

    df = validate_numeric_columns(df)

    stations = inspect_dataset(df)

    validate_wind_information(df)

    # ------------------------------------------------------
    # STATION MAPPING
    # ------------------------------------------------------

    (
        df,
        stations,
        station_to_id
    ) = create_station_mapping(df)

    # ------------------------------------------------------
    # HOURLY STRUCTURE
    # ------------------------------------------------------

    timestamps = verify_hourly_structure(
        df,
        stations
    )

    # ------------------------------------------------------
    # RAW ARRAYS
    # ------------------------------------------------------

    (
        X_raw,
        Y_raw,
        all_timestamps
    ) = construct_station_arrays(
        df,
        stations
    )

    # ------------------------------------------------------
    # MULTI-HORIZON SEQUENCES
    # ------------------------------------------------------

    (
        X_sequences,
        Y_sequences,
        station_record_sequences,
        sequence_timestamps
    ) = create_sequences(

        X_raw,
        Y_raw,
        all_timestamps,
        df

    )

    # ------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ------------------------------------------------------

    (
        X_train,
        Y_train,
        records_train,
        timestamps_train,

        X_val,
        Y_val,
        records_val,
        timestamps_val,

        X_test,
        Y_test,
        records_test,
        timestamps_test

    ) = split_sequences(

        X_sequences,
        Y_sequences,
        station_record_sequences,
        sequence_timestamps

    )

    # ------------------------------------------------------
    # FIT SCALERS ON TRAIN ONLY
    # ------------------------------------------------------

    (
        feature_scaler,
        target_scaler

    ) = fit_scalers(

        X_train,
        Y_train

    )

    # ------------------------------------------------------
    # SCALE X
    # ------------------------------------------------------

    X_train_scaled = scale_X(
        X_train,
        feature_scaler
    )

    X_val_scaled = scale_X(
        X_val,
        feature_scaler
    )

    X_test_scaled = scale_X(
        X_test,
        feature_scaler
    )

    # ------------------------------------------------------
    # SCALE Y
    # ------------------------------------------------------

    Y_train_scaled = scale_Y(
        Y_train,
        target_scaler
    )

    Y_val_scaled = scale_Y(
        Y_val,
        target_scaler
    )

    Y_test_scaled = scale_Y(
        Y_test,
        target_scaler
    )

    # ------------------------------------------------------
    # CONVERT TO TENSORS
    # ------------------------------------------------------

    (
        X_train_tensor,
        Y_train_tensor,

        X_val_tensor,
        Y_val_tensor,

        X_test_tensor,
        Y_test_tensor

    ) = convert_to_tensors(

        X_train_scaled,
        Y_train_scaled,

        X_val_scaled,
        Y_val_scaled,

        X_test_scaled,
        Y_test_scaled

    )

    # ------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------

    validate_final_data(

        X_train_tensor,
        Y_train_tensor,
        records_train,

        X_val_tensor,
        Y_val_tensor,
        records_val,

        X_test_tensor,
        Y_test_tensor,
        records_test

    )

    # ======================================================
    # FINAL INFORMATION
    # ======================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        "Stations:",
        len(stations)
    )

    print(
        "Features:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Feature columns:",
        FEATURE_COLUMNS
    )

    print(
        "Input sequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Prediction horizon:",
        PREDICTION_HORIZON
    )

    print(
        "Target:",
        TARGET_COLUMN
    )

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    print("\nTRAIN:")

    print(
        "  X:",
        X_train_tensor.shape
    )

    print(
        "  Y:",
        Y_train_tensor.shape
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    print("\nVALIDATION:")

    print(
        "  X:",
        X_val_tensor.shape
    )

    print(
        "  Y:",
        Y_val_tensor.shape
    )

    # ------------------------------------------------------
    # TEST
    # ------------------------------------------------------

    print("\nTEST:")

    print(
        "  X:",
        X_test_tensor.shape
    )

    print(
        "  Y:",
        Y_test_tensor.shape
    )

    # ------------------------------------------------------
    # Explicit forecast structure
    # ------------------------------------------------------

    print("\nFORECAST STRUCTURE:")

    print(
        "  Input:",
        "previous 24 hourly observations"
    )

    print(
        "  Output:",
        "next 24 hourly PM2.5 values"
    )

    print(
        "  12-hour evaluation:",
        "forecast hours 1-12"
    )

    print(
        "  24-hour evaluation:",
        "forecast hours 1-24"
    )

    print(
        "\nFeature scaler fitted on TRAIN only."
    )

    print(
        "Target scaler fitted on TRAIN only."
    )

    # ======================================================
    # RETURN
    # ======================================================

    return {

        "X_train":
            X_train_tensor,

        "Y_train":
            Y_train_tensor,

        "X_val":
            X_val_tensor,

        "Y_val":
            Y_val_tensor,

        "X_test":
            X_test_tensor,

        "Y_test":
            Y_test_tensor,

        "station_records_train":
            records_train,

        "station_records_val":
            records_val,

        "station_records_test":
            records_test,

        "x_scaler":
            feature_scaler,

        "y_scaler":
            target_scaler,

        "feature_scaler":
            feature_scaler,

        "target_scaler":
            target_scaler,

        "feature_columns":
            FEATURE_COLUMNS,

        "target_column":
            TARGET_COLUMN,

        "stations":
            stations,

        "station_to_id":
            station_to_id,

        "sequence_timestamps_train":
            timestamps_train,

        "sequence_timestamps_val":
            timestamps_val,

        "sequence_timestamps_test":
            timestamps_test

    }


# ==========================================================
# DIRECT EXECUTION
# ==========================================================

if __name__ == "__main__":

    data = prepare_model_data()

    print(
        "\nPreprocessing finished successfully."
    )
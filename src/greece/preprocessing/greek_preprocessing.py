import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import SEQUENCE_LENGTH, PREDICTION_HORIZON

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "greek_urban_air_quality_2020_2024.csv")
DATASET_PATH = DEFAULT_DATASET_PATH if os.path.exists(DEFAULT_DATASET_PATH) else "greek_urban_air_quality_2020_2024.csv"

# Greek CSV Mapping -> DySTCA Feature
FEATURE_MAPPING = {
    "PM2_5_ugm3": "pm2_5",
    "PM10_ugm3": "pm10",
    "CO_mgm3": "co",
    "NO2_ugm3": "no2",
    "Temperature_C": "temp_c",
    "Humidity_pct": "humidity",
    "Pressure_hPa": "pressure_mb",
    "Wind_Speed_ms": "windspeed_kph"
}

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "co", "no2", "temp_c", "humidity", "pressure_mb", "windspeed_kph",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"
]

def load_and_split_dataset():
    print("\n" + "=" * 70)
    print("LOADING AND SPLITTING GREEK DATASET")
    print("=" * 70)
    
    df = pd.read_csv(DATASET_PATH)
    df['Timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['Timestamp', 'station']).reset_index(drop=True)
    
    unique_timestamps = df['Timestamp'].drop_duplicates().sort_values().reset_index(drop=True)
    total_len = len(unique_timestamps)
    
    train_end = int(total_len * 0.70)
    val_end = train_end + int(total_len * 0.15)
    
    train_ts = unique_timestamps.iloc[:train_end]
    val_ts = unique_timestamps.iloc[train_end:val_end]
    test_ts = unique_timestamps.iloc[val_end:]
    
    print(f"Original 3-hourly timestamps: {total_len}")
    print(f"Train split: {len(train_ts)} timestamps (ends {train_ts.iloc[-1]})")
    print(f"Val split:   {len(val_ts)} timestamps (starts {val_ts.iloc[0]}, ends {val_ts.iloc[-1]})")
    print(f"Test split:  {len(test_ts)} timestamps (starts {test_ts.iloc[0]}, ends {test_ts.iloc[-1]})")
    
    train_df = df[df['Timestamp'].isin(train_ts)].copy()
    val_df = df[df['Timestamp'].isin(val_ts)].copy()
    test_df = df[df['Timestamp'].isin(test_ts)].copy()
    
    return train_df, val_df, test_df

def process_split(df, split_name):
    print(f"\nProcessing {split_name} split...")
    
    # Map column names
    df = df.rename(columns=FEATURE_MAPPING)
    
    stations = sorted(df['station'].unique().tolist())
    station_to_id = {st: idx for idx, st in enumerate(stations)}
    
    # Resample to 1H per station and interpolate
    resampled_dfs = []
    for station in stations:
        st_df = df[df['station'] == station].copy()
        st_df = st_df.set_index('Timestamp')
        
        # Resample numeric columns and interpolate
        numeric_cols = list(FEATURE_MAPPING.values())
        numeric_df = st_df[numeric_cols].resample('1h').interpolate(method='time')
        
        # Merge back with non-numeric, using ffill then bfill for categorical/metadata
        st_df = st_df.resample('1h').ffill().bfill()
        
        for col in numeric_cols:
            st_df[col] = numeric_df[col]
            
        st_df['station'] = station
        st_df['Station_ID'] = station_to_id[station]
        
        resampled_dfs.append(st_df)
        
    df = pd.concat(resampled_dfs).reset_index()
    df = df.sort_values(['Timestamp', 'Station_ID']).reset_index(drop=True)
    
    # Add cyclic features
    hours = df["Timestamp"].dt.hour.to_numpy()
    days = df["Timestamp"].dt.dayofweek.to_numpy()
    months = df["Timestamp"].dt.month.to_numpy()
    
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["day_sin"] = np.sin(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["day_cos"] = np.cos(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["month_sin"] = np.sin(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    
    # Extract station records for EVERY timestep
    print("Building station records lookup...")
    
    # We will build a list of length num_times, where each element is a list of 31 station records
    # To do this efficiently, we can use group_by or just iterate since it's already sorted by Timestamp then Station_ID
    # But it's easier to reshape a list of all dicts.
    
    # Pre-calculate a list of all rows as dicts to avoid iterrows
    df_records = df.to_dict('records')
    
    # df is sorted by Timestamp, then Station_ID. 
    # So every num_stations (31) rows correspond to one timestamp.
    timestamps = df["Timestamp"].drop_duplicates().sort_values().tolist()
    num_stations = len(stations)
    num_times = len(timestamps)
    
    station_records_over_time = []
    
    for t_idx in range(num_times):
        start_idx = t_idx * num_stations
        end_idx = start_idx + num_stations
        time_rows = df_records[start_idx:end_idx]
        
        t_records = []
        for row in time_rows:
            record = {
                "Station_ID": int(row["Station_ID"]),
                "Station": str(row["station"]),
                "Latitude": float(row["latitude"]),
                "Longitude": float(row["longitude"]),
                "Wind_Speed": float(row["windspeed_kph"]),
                "Temperature": float(row["temp_c"]),
                "Humidity": float(row["humidity"]),
                "Pressure": float(row["pressure_mb"]),
                "PM2.5": float(row["pm2_5"]),
                "PM10": float(row["pm10"]),
                "CO": float(row["co"]),
                "NO2": float(row["no2"])
            }
            t_records.append(record)
        
        # Sort by Station_ID just to be safe
        t_records.sort(key=lambda x: x["Station_ID"])
        station_records_over_time.append(t_records)
        
    print("Station records lookup built.")
    
    num_features = len(FEATURE_COLUMNS)
    X = np.zeros((num_times, num_stations, num_features), dtype=np.float32)
    Y = np.zeros((num_times, num_stations, 1), dtype=np.float32)
    
    timestamp_to_index = {ts: idx for idx, ts in enumerate(timestamps)}
    
    # Vectorized X and Y array assignment
    t_indices = df['Timestamp'].map(timestamp_to_index).values
    s_indices = df['Station_ID'].values.astype(int)
    
    for f_idx, feature in enumerate(FEATURE_COLUMNS):
        X[t_indices, s_indices, f_idx] = df[feature].values.astype(np.float32)
        
    Y[t_indices, s_indices, 0] = df["pm2_5"].values.astype(np.float32)
        
    return X, Y, station_records_over_time



def prepare_model_data():
    train_df, val_df, test_df = load_and_split_dataset()
    
    X_raw_train, Y_raw_train, records_train = process_split(train_df, "TRAIN")
    X_raw_val, Y_raw_val, records_val = process_split(val_df, "VALIDATION")
    X_raw_test, Y_raw_test, records_test = process_split(test_df, "TEST")
    
    print("\n" + "=" * 70)
    print("FITTING SCALERS ON TRAINING DATA ONLY")
    print("=" * 70)
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feature_scaler.fit(X_raw_train.reshape(-1, X_raw_train.shape[-1]))
    target_scaler.fit(Y_raw_train.reshape(-1, 1))
    
    def scale_array(arr, scaler):
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
        
    X_train_scaled = scale_array(X_raw_train, feature_scaler)
    X_val_scaled = scale_array(X_raw_val, feature_scaler)
    X_test_scaled = scale_array(X_raw_test, feature_scaler)
    
    Y_train_scaled = scale_array(Y_raw_train, target_scaler)
    Y_val_scaled = scale_array(Y_raw_val, target_scaler)
    Y_test_scaled = scale_array(Y_raw_test, target_scaler)
    
    print("Scalers applied successfully.")
    
    import torch
    
    return {
        "X_train": torch.from_numpy(X_train_scaled),
        "Y_train": torch.from_numpy(Y_train_scaled),
        "station_records_train": records_train,
        
        "X_val": torch.from_numpy(X_val_scaled),
        "Y_val": torch.from_numpy(Y_val_scaled),
        "station_records_val": records_val,
        
        "X_test": torch.from_numpy(X_test_scaled),
        "Y_test": torch.from_numpy(Y_test_scaled),
        "station_records_test": records_test,
        
        "x_scaler": feature_scaler,
        "y_scaler": target_scaler,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "pm2_5"
    }

if __name__ == "__main__":
    data = prepare_model_data()
    print("Train X shape:", data["X_train"].shape)
    print("Train Y shape:", data["Y_train"].shape)
    print("Val X shape:", data["X_val"].shape)
    print("Val Y shape:", data["Y_val"].shape)
    print("Test X shape:", data["X_test"].shape)
    print("Test Y shape:", data["Y_test"].shape)

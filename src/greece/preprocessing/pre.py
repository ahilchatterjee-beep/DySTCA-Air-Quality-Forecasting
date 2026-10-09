import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "greek_urban_air_quality_2020_2024.csv")

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "greece", "greek_urban_air_quality_2020_2024.csv")

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

def load_and_validate_dataset():
    print("Loading Greek Dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    stations = df['station'].unique()
    num_stations = len(stations)
    
    assert num_stations == 31, f"Expected 31 stations, found {num_stations}"
    
    df['Timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['Timestamp', 'station']).reset_index(drop=True)
    
    unique_timestamps = df['Timestamp'].drop_duplicates().sort_values().reset_index(drop=True)
    diffs = unique_timestamps.diff().dropna()
    mode_diff_hours = diffs.mode()[0].total_seconds() / 3600
    assert mode_diff_hours == 3.0, f"Expected native 3-hour interval, found {mode_diff_hours}h"
    
    return df, stations

def create_sliding_windows(df, stations, lookback_steps=24, horizon_steps=8):
    df = df.rename(columns=FEATURE_MAPPING)
    
    station_to_id = {st: idx for idx, st in enumerate(stations)}
    df['Station_ID'] = df['station'].map(station_to_id)
    
    hours = df["Timestamp"].dt.hour.to_numpy()
    days = df["Timestamp"].dt.dayofweek.to_numpy()
    months = df["Timestamp"].dt.month.to_numpy()
    
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["day_sin"] = np.sin(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["day_cos"] = np.cos(2.0 * np.pi * days / 7.0).astype(np.float32)
    df["month_sin"] = np.sin(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    
    timestamps = df['Timestamp'].drop_duplicates().sort_values().to_numpy()
    num_times = len(timestamps)
    num_stations = len(stations)
    num_features = len(FEATURE_COLUMNS)
    
    df_pivot = df.pivot(index='Timestamp', columns='Station_ID')
    df_pivot = df_pivot.ffill().bfill()
    
    X_matrix = np.zeros((num_times, num_stations, num_features), dtype=np.float32)
    for f_idx, feature in enumerate(FEATURE_COLUMNS):
        if feature in df_pivot.columns.levels[0]:
            X_matrix[:, :, f_idx] = df_pivot[feature].to_numpy()
    
    Y_matrix = df_pivot["pm2_5"].to_numpy().astype(np.float32)
    
    station_meta = df.groupby('Station_ID').first()[['station', 'latitude', 'longitude']].reset_index()
    
    X_windows = []
    Y_windows = []
    records_windows = []
    
    for i in range(num_times - lookback_steps - horizon_steps + 1):
        X_w = X_matrix[i:i+lookback_steps, :, :]
        Y_w = Y_matrix[i+lookback_steps:i+lookback_steps+horizon_steps, :]
        Y_w = Y_w.T # [stations, 8]
        
        t_records = []
        for t in range(lookback_steps):
            step_records = []
            for st_id in range(num_stations):
                meta = station_meta.iloc[st_id]
                step_records.append({
                    "Station_ID": st_id,
                    "Station": str(meta['station']),
                    "Latitude": float(meta['latitude']),
                    "Longitude": float(meta['longitude']),
                    "Wind_U": 0.0,
                    "Wind_V": 0.0
                })
            t_records.append(step_records)
            
        X_windows.append(X_w)
        Y_windows.append(Y_w)
        records_windows.append(t_records)
        
    X_windows = np.array(X_windows, dtype=np.float32)
    Y_windows = np.array(Y_windows, dtype=np.float32)
    
    return X_windows, Y_windows, records_windows, timestamps

def prepare_model_data(lookback_steps=24, horizon_steps=8):
    df, stations = load_and_validate_dataset()
    X, Y, records, timestamps = create_sliding_windows(df, stations, lookback_steps, horizon_steps)
    
    total_len = len(X)
    train_end = int(total_len * 0.70)
    val_end = train_end + int(total_len * 0.15)
    
    X_train, Y_train, rec_train = X[:train_end], Y[:train_end], records[:train_end]
    X_val, Y_val, rec_val = X[train_end:val_end], Y[train_end:val_end], records[train_end:val_end]
    X_test, Y_test, rec_test = X[val_end:], Y[val_end:], records[val_end:]
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    target_scaler.fit(Y_train.reshape(-1, 1))
    
    def scale_array(arr, scaler):
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
        
    def scale_y_array(arr, scaler):
        orig_shape = arr.shape
        flat = arr.reshape(-1, 1)
        scaled = scaler.transform(flat)
        return scaled.reshape(orig_shape)
        
    X_train_scaled = scale_array(X_train, feature_scaler)
    X_val_scaled = scale_array(X_val, feature_scaler)
    X_test_scaled = scale_array(X_test, feature_scaler)
    
    Y_train_scaled = scale_y_array(Y_train, target_scaler)
    Y_val_scaled = scale_y_array(Y_val, target_scaler)
    Y_test_scaled = scale_y_array(Y_test, target_scaler)
    
    return {
        "X_train": torch.from_numpy(X_train_scaled),
        "Y_train": torch.from_numpy(Y_train_scaled),
        "station_records_train": rec_train,
        
        "X_val": torch.from_numpy(X_val_scaled),
        "Y_val": torch.from_numpy(Y_val_scaled),
        "station_records_val": rec_val,
        
        "X_test": torch.from_numpy(X_test_scaled),
        "Y_test": torch.from_numpy(Y_test_scaled),
        "station_records_test": rec_test,
        
        "y_scaler": target_scaler,
        "timestamps": timestamps
    }

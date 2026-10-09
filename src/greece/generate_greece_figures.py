import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.makedirs('Greece_Paper_Results', exist_ok=True)

# 1. Load data
npz = np.load('greece_predictions.npz')
y_true = npz['targets_original']  # [samples, stations, horizon]
y_pred = npz['predictions_original']

samples, stations, horizon = y_true.shape
print(f"Loaded predictions: {samples} samples, {stations} stations, {horizon} horizon")

# Helper metric functions
def calc_mae(t, p): return np.mean(np.abs(t - p))
def calc_rmse(t, p): return np.sqrt(np.mean((t - p)**2))
def calc_r2(t, p):
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - np.mean(t))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# ---------------------------------------------------------
# STEP 2: OVERALL METRICS
# ---------------------------------------------------------
overall_mae = calc_mae(y_true, y_pred)
overall_rmse = calc_rmse(y_true, y_pred)
overall_r2 = calc_r2(y_true, y_pred)
smape = np.mean(200 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

print(f"Overall MAE: {overall_mae:.2f}")
print(f"Overall RMSE: {overall_rmse:.2f}")
print(f"Overall R2: {overall_r2:.4f}")
print(f"Overall sMAPE: {smape:.2f}%")

# ---------------------------------------------------------
# HORIZON METRICS
# ---------------------------------------------------------
h_mae = []
h_rmse = []
h_r2 = []

for h in range(horizon):
    t = y_true[:, :, h]
    p = y_pred[:, :, h]
    h_mae.append(calc_mae(t, p))
    h_rmse.append(calc_rmse(t, p))
    h_r2.append(calc_r2(t, p))

# ---------------------------------------------------------
# FIGURE B: COMPACT SUMMARY (3-panel)
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

x = np.arange(1, horizon + 1)
axs[0].plot(x, h_r2, marker='o', color='tab:blue')
axs[0].set_title('R² vs Forecast Horizon')
axs[0].set_xlabel('Forecast horizon (hours ahead)')
axs[0].set_ylabel('R²')
axs[0].grid(True)

axs[1].plot(x, h_mae, marker='o', color='tab:orange')
axs[1].set_title('MAE vs Forecast Horizon')
axs[1].set_xlabel('Forecast horizon (hours ahead)')
axs[1].set_ylabel('MAE (µg/m³)')
axs[1].grid(True)

axs[2].plot(x, h_rmse, marker='o', color='tab:red')
axs[2].set_title('RMSE vs Forecast Horizon')
axs[2].set_xlabel('Forecast horizon (hours ahead)')
axs[2].set_ylabel('RMSE (µg/m³)')
axs[2].grid(True)

plt.tight_layout()
plt.savefig('Greece_Paper_Results/02_greece_horizon_summary.png', dpi=300)
plt.savefig('Greece_Paper_Results/02_greece_horizon_summary.pdf')
plt.close()

# ---------------------------------------------------------
# TABLES (Step 7)
# ---------------------------------------------------------
table_data = {
    'Metric': ['R²', 'MAE', 'RMSE'],
    'h+1': [h_r2[0], h_mae[0], h_rmse[0]],
    'h+6': [h_r2[5], h_mae[5], h_rmse[5]],
    'h+12': [h_r2[11], h_mae[11], h_rmse[11]],
    'h+24': [h_r2[23], h_mae[23], h_rmse[23]],
    '24-h Pooled': [overall_r2, overall_mae, overall_rmse]
}
df_table = pd.DataFrame(table_data)

# Round values for display
for col in ['h+1', 'h+6', 'h+12', 'h+24', '24-h Pooled']:
    if df_table['Metric'][0] == 'R²':  # R2
        df_table.loc[0, col] = round(df_table.loc[0, col], 3)
    df_table.loc[1, col] = round(df_table.loc[1, col], 2)
    df_table.loc[2, col] = round(df_table.loc[2, col], 2)

df_table.to_csv('Greece_Paper_Results/greece_results_table.csv', index=False)
df_table.to_markdown('Greece_Paper_Results/greece_results_table.md', index=False)
df_table.style.to_latex('Greece_Paper_Results/greece_results_table.tex')

# ---------------------------------------------------------
# STATION METRICS & HEATMAP (Step 4 & 8)
# ---------------------------------------------------------
station_names = [f"Station_{i+1}" for i in range(stations)]
s_mae = []
s_rmse = []
s_r2 = []

heatmap_r2 = np.zeros((stations, horizon))

for s in range(stations):
    t = y_true[:, s, :]
    p = y_pred[:, s, :]
    s_mae.append(calc_mae(t, p))
    s_rmse.append(calc_rmse(t, p))
    s_r2.append(calc_r2(t, p))
    
    for h in range(horizon):
        heatmap_r2[s, h] = calc_r2(t[:, h], p[:, h])

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_r2, cmap='viridis', xticklabels=np.arange(1, horizon+1), yticklabels=station_names, 
            cbar_kws={'label': 'R²'})
plt.title('Station × Horizon Performance Heatmap (R²)')
plt.xlabel('Forecast Horizon (h)')
plt.ylabel('Monitoring Station')
plt.tight_layout()
plt.savefig('Greece_Paper_Results/03_greece_station_horizon_heatmap.png', dpi=300)
plt.savefig('Greece_Paper_Results/03_greece_station_horizon_heatmap.pdf')
plt.close()

# Station CSV
df_station = pd.DataFrame({
    'Station': station_names,
    'MAE': s_mae,
    'RMSE': s_rmse,
    'R²': s_r2,
    'Number_of_test_samples': [samples] * stations
})
df_station.to_csv('Greece_Paper_Results/greece_station_metrics.csv', index=False)

# Summary stats
summary = {
    'Station': ['Mean', 'Median', 'Std', 'Min', 'Max'], 
    'MAE': [np.mean(s_mae), np.median(s_mae), np.std(s_mae), np.min(s_mae), np.max(s_mae)], 
    'RMSE': [np.mean(s_rmse), np.median(s_rmse), np.std(s_rmse), np.min(s_rmse), np.max(s_rmse)], 
    'R²': [np.mean(s_r2), np.median(s_r2), np.std(s_r2), np.min(s_r2), np.max(s_r2)], 
    'Number_of_test_samples': [samples]*5
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv('Greece_Paper_Results/greece_station_metrics.csv', mode='a', header=False, index=False)


# Summary stats
summary = {
    'Station': ['Mean', 'Median', 'Std', 'Min', 'Max'], 
    'MAE': [np.mean(s_mae), np.median(s_mae), np.std(s_mae), np.min(s_mae), np.max(s_mae)], 
    'RMSE': [np.mean(s_rmse), np.median(s_rmse), np.std(s_rmse), np.min(s_rmse), np.max(s_rmse)], 
    'R²': [np.mean(s_r2), np.median(s_r2), np.std(s_r2), np.min(s_r2), np.max(s_r2)], 
    'Number_of_test_samples': [samples]*5
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv('Greece_Paper_Results/greece_station_metrics.csv', mode='a', header=False, index=False)


# Summary stats
summary = {
    'Station': ['Mean', 'Median', 'Std', 'Min', 'Max'], 
    'MAE': [np.mean(s_mae), np.median(s_mae), np.std(s_mae), np.min(s_mae), np.max(s_mae)], 
    'RMSE': [np.mean(s_rmse), np.median(s_rmse), np.std(s_rmse), np.min(s_rmse), np.max(s_rmse)], 
    'R²': [np.mean(s_r2), np.median(s_r2), np.std(s_r2), np.min(s_r2), np.max(s_r2)], 
    'Number_of_test_samples': [samples]*5
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv('Greece_Paper_Results/greece_station_metrics.csv', mode='a', header=False, index=False)


# Summary stats
summary = {
    'Station': ['Mean', 'Median', 'Std', 'Min', 'Max'], 
    'MAE': [np.mean(s_mae), np.median(s_mae), np.std(s_mae), np.min(s_mae), np.max(s_mae)], 
    'RMSE': [np.mean(s_rmse), np.median(s_rmse), np.std(s_rmse), np.min(s_rmse), np.max(s_rmse)], 
    'R²': [np.mean(s_r2), np.median(s_r2), np.std(s_r2), np.min(s_r2), np.max(s_r2)], 
    'Number_of_test_samples': [samples]*5
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv('Greece_Paper_Results/greece_station_metrics.csv', mode='a', header=False, index=False)


# ---------------------------------------------------------
# REPRESENTATIVE FORECAST (Step 5)
# ---------------------------------------------------------
median_mae = np.median(s_mae)
diffs = np.abs(np.array(s_mae) - median_mae)
repr_station_idx = np.argmin(diffs)

# Select a specific test sample: e.g., sample with mean MAE over horizon closest to the station's median MAE
t_repr = y_true[:, repr_station_idx, :]
p_repr = y_pred[:, repr_station_idx, :]
sample_maes = np.mean(np.abs(t_repr - p_repr), axis=1)
median_sample_mae = np.median(sample_maes)
repr_sample_idx = np.argmin(np.abs(sample_maes - median_sample_mae))

plt.figure(figsize=(10, 4))
x_repr = np.arange(1, horizon + 1)
plt.plot(x_repr, t_repr[repr_sample_idx], label='Actual PM2.5', marker='o')
plt.plot(x_repr, p_repr[repr_sample_idx], label='Predicted', marker='x', linestyle='--')
plt.title(f'Representative Forecast (Station {repr_station_idx+1}, Sample {repr_sample_idx})')
plt.xlabel('Forecast Horizon (h)')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Greece_Paper_Results/representative_greece_forecast.png', dpi=300)
plt.savefig('Greece_Paper_Results/representative_greece_forecast.pdf')
plt.close()


# ---------------------------------------------------------
# ERROR ANALYSIS (Step 9)
# ---------------------------------------------------------
residuals = y_true - y_pred
mean_res = np.mean(residuals)
median_res = np.median(residuals)
std_res = np.std(residuals)
q25 = np.percentile(residuals, 25)
q75 = np.percentile(residuals, 75)

horizon_bias = np.mean(residuals, axis=(0, 1)) # [horizon]

error_report = f"""ERROR ANALYSIS
==============
Mean Residual: {mean_res:.3f} µg/m³
Median Residual: {median_res:.3f} µg/m³
Residual Std Dev: {std_res:.3f} µg/m³
MAE: {overall_mae:.3f} µg/m³
RMSE: {overall_rmse:.3f} µg/m³
Residual 25th Percentile: {q25:.3f} µg/m³
Residual 75th Percentile: {q75:.3f} µg/m³

Horizon-wise Residual Bias (h+1 to h+24):
"""
for h, bias in enumerate(horizon_bias):
    error_report += f"h+{h+1}: {bias:.3f} µg/m³\n"

with open('Greece_Paper_Results/greece_error_analysis.txt', 'w') as f:
    f.write(error_report)


# ---------------------------------------------------------
# FULL HORIZON TABLE (Step 7 supplementary)
# ---------------------------------------------------------
df_full_horizon = pd.DataFrame({
    'Horizon': [f"h+{h+1}" for h in range(horizon)],
    'R²': h_r2,
    'MAE': h_mae,
    'RMSE': h_rmse
})
df_full_horizon.to_csv('Greece_Paper_Results/greece_full_horizon_table.csv', index=False)

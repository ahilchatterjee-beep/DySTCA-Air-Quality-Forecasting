with open('generate_greece_figures.py', encoding='utf-8') as f:
    content = f.read()

replacement = """df_station.to_csv('Greece_Paper_Results/greece_station_metrics.csv', index=False)

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
"""

content = content.replace("df_station.to_csv('Greece_Paper_Results/greece_station_metrics.csv', index=False)", replacement)

with open('generate_greece_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)

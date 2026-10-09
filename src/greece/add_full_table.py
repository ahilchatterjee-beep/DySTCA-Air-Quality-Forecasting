with open('generate_greece_figures.py', encoding='utf-8') as f:
    content = f.read()

full_table_code = """
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
"""

if "greece_full_horizon_table.csv" not in content:
    content += "\n" + full_table_code

with open('generate_greece_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)

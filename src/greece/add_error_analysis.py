with open('generate_greece_figures.py', encoding='utf-8') as f:
    content = f.read()

error_analysis_code = """
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

error_report = f\"\"\"ERROR ANALYSIS
==============
Mean Residual: {mean_res:.3f} µg/m³
Median Residual: {median_res:.3f} µg/m³
Residual Std Dev: {std_res:.3f} µg/m³
MAE: {overall_mae:.3f} µg/m³
RMSE: {overall_rmse:.3f} µg/m³
Residual 25th Percentile: {q25:.3f} µg/m³
Residual 75th Percentile: {q75:.3f} µg/m³

Horizon-wise Residual Bias (h+1 to h+24):
\"\"\"
for h, bias in enumerate(horizon_bias):
    error_report += f"h+{h+1}: {bias:.3f} µg/m³\\n"

with open('Greece_Paper_Results/greece_error_analysis.txt', 'w') as f:
    f.write(error_report)
"""

if "ERROR ANALYSIS" not in content:
    content += "\n" + error_analysis_code

with open('generate_greece_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)

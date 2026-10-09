import os

audit_text = """EXISTING FIGURE AUDIT
=====================

1. 01_training_validation_loss.png
PURPOSE: Shows learning curves over epochs.
REDUNDANT? No.
USEFUL FOR PAPER? Yes, as a standard supplementary plot.
RECOMMENDED LOCATION: SUPPLEMENTARY

2. 02_validation_mae_rmse.png
PURPOSE: Shows validation MAE/RMSE over epochs.
REDUNDANT? Yes (training loss curve suffices for convergence tracking).
USEFUL FOR PAPER? No.
RECOMMENDED LOCATION: NOT NEEDED

3. 03_validation_r2.png
PURPOSE: Shows R2 progression over epochs.
REDUNDANT? Yes.
USEFUL FOR PAPER? No.
RECOMMENDED LOCATION: NOT NEEDED

4. 04_mean_future_prediction.png
PURPOSE: Shows prediction vs actual for the mean of all stations across a single test horizon window.
REDUNDANT? Yes (too simplified).
USEFUL FOR PAPER? No.
RECOMMENDED LOCATION: NOT NEEDED

5. 05_actual_vs_predicted_24h.png
PURPOSE: Shows 24-h sequence prediction vs actual for Station 0.
REDUNDANT? Yes. We are replacing this with a statistically-selected representative station (Figure C).
USEFUL FOR PAPER? No.
RECOMMENDED LOCATION: NOT NEEDED

6. 06_actual_vs_predicted_scatter.png
PURPOSE: Shows scatter density of predicted vs actual across all samples.
REDUNDANT? No.
USEFUL FOR PAPER? Yes.
RECOMMENDED LOCATION: SUPPLEMENTARY

7. 07_residual_distribution.png
PURPOSE: Histogram of errors.
REDUNDANT? No.
USEFUL FOR PAPER? Yes (shows error distribution properties).
RECOMMENDED LOCATION: SUPPLEMENTARY

8. 08_horizon_mae_rmse.png
9. 09_horizon_r2.png
10. 10_horizon_actual_vs_predicted.png
PURPOSE: Individual horizon metric charts.
REDUNDANT? Yes. These are being replaced by the single Compact 3-Panel Summary (Figure B) which shows all horizon metrics simultaneously.
USEFUL FOR PAPER? No.
RECOMMENDED LOCATION: NOT NEEDED

---
FLAGGED ISSUES FROM PREVIOUS RUNS (Step 11 Audit):
* The folder `all_generated_graphs` contains Delhi-specific baseline comparison vectors (e.g., `01_models_comparison_r2.png`, `17_ablation_tcn_gat.png`) from the original experiment. These do NOT apply to the Greece dataset because the baselines were not re-trained. They are flagged and must NOT be used to represent Greece experiment benchmarks.
* The original Delhi experiment methodology flowchart (`16_methodology_implementation_flowchart.png`) is a conceptual figure and is independent of Greece.

RECOMMENDATION: We generated exactly two new highly informative figures specifically for Greece: The Station x Horizon Heatmap, and the Compact Horizon Summary, plus one rigorous representative forecast.
"""

with open('Greece_Paper_Results/01_existing_figures_audit.txt', 'w') as f:
    f.write(audit_text)

readme_text = """# Greece Dataset External Evaluation

1. **What the Greece experiment demonstrates:** The geographic robustness, horizon-wise stability, and multi-station generalization capabilities of the frozen DySTCA architecture on an unseen European multi-station graph.
2. **What it does NOT demonstrate:** State-of-the-art superiority against other baselines on the Greece dataset (since baselines were not re-trained on Greece), nor universal zero-shot generalizability (the model was fine-tuned/re-trained specifically on the Greek training split).
3. **Exact preprocessing used:** Standard chronological split (70/15/15), temporal interpolation to 1h frequency purely within the historical boundaries (no future leakage), and extraction of 14 features (8 physical, 6 calendar). 
4. **Exact test protocol:** Evaluated on the held-out chronological test split (6386 sequences, 24-h prediction horizon).
5. **Exact metrics:** R², MAE (µg/m³), RMSE (µg/m³) evaluated both globally and explicitly across all 24 horizons on the original unscaled PM2.5 scale.
6. **Main Paper Recommended Figures:** 
    - Figure B (02_greece_horizon_summary): The Compact 3-Panel Horizon plot perfectly captures temporal degradation.
    - Figure A (03_greece_station_horizon_heatmap): Shows the spatial consistency of the R2 score across all 31 stations.
7. **Supplementary Recommended Figures:** 
    - `01_training_validation_loss.png`
    - `06_actual_vs_predicted_scatter.png`
    - `07_residual_distribution.png`
8. **Numerical Results for Paper:** 
    - Use `greece_results_table.md` / `.tex` for the compact main text. 
    - Use `greece_station_metrics.csv` for the supplementary data files.
9. **Methodological Limitation:** The model was given access to future meteorology as covariates (a standard practice, but implies the system relies on accurate meteorological forecasts in a real-world deployment). 
10. **Do NOT Use:** Any multi-model comparison charts present in `all_generated_graphs/` (e.g., ablation studies, baseline comparisons), as these reflect the Delhi experiment.

## Files
- `01_existing_figures_audit.txt`: Review of original default graphs.
- `02_greece_horizon_summary.pdf/.png`: 3-panel horizon degradation.
- `03_greece_station_horizon_heatmap.pdf/.png`: Station-wise spatial R² performance.
- `representative_greece_forecast.pdf/.png`: Methodologically selected 24h sample.
- `greece_results_table.*`: Numerical performance summary.
- `greece_station_metrics.csv`: Per-station breakdown.
- `greece_experiment_metadata.json`: Audit log of exact configuration for reproducibility.
"""

with open('Greece_Paper_Results/README_GREECE_RESULTS.md', 'w', encoding='utf-8') as f:
    f.write(readme_text)

print("Created 01_existing_figures_audit.txt and README_GREECE_RESULTS.md")

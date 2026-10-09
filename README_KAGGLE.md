# DySTCA: Dynamic Spatio-Temporal Cross-Attention

This repository contains the complete experimental suite for the DySTCA architecture, a multi-horizon air quality forecasting model featuring a dynamic hyper-heuristic graph.

## Project Structure
- `run_kaggle.py` - Single entry point for all experiments
- `src/delhi/` - Source code for the 24-hour PM2.5 forecasting experiment
- `src/greece/` - Source code for the native 3-hour Greek dataset evaluation
- `data/` - Datasets (auto-discovered on Kaggle)
- `tests/` - Automated test suite

## Usage

To run all experiments automatically:
```bash
python run_kaggle.py --experiment all
```

To run individual experiments:
```bash
python run_kaggle.py --experiment baseline
python run_kaggle.py --experiment multiseed
python run_kaggle.py --experiment ablation
python run_kaggle.py --experiment greece
python run_kaggle.py --experiment benchmark
```

## Datasets
1. **Delhi (delhi-weather-aqi-2025.csv)**
   - 6 Stations, hourly sampling
   - Lookback: 168 hours (7 days)
   - Horizon: 24 hours
2. **Greece (greek_urban_air_quality_2020_2024.csv)**
   - 10 Stations used in evaluation
   - Native 3-hour sampling (NOT interpolated to hourly)
   - Lookback: 72 hours (24 native steps)
   - Horizon: 3 hours (1 native step)

## Hardware
The pipeline automatically detects CUDA if available. If not, experiments will gracefully fallback to CPU (though training may take a long time). 

For the autoregressive evaluation, batch size is explicitly reduced to fit within Kaggle 16GB GPU memory constraints.

# DySTCA: Dynamic Spatio-Temporal Cross-Attention for Air-Quality Forecasting

Research implementation and experiment utilities for the DySTCA air-quality forecasting project.

> **Research integrity note:** The Python source, model implementation, configuration files, tests, and existing lightweight experiment summaries are preserved from the supplied project archive. This GitHub-ready package excludes raw datasets, serialized dataset caches, model checkpoints, and prediction-array files to keep the repository manageable. No model architecture or scientific source logic was intentionally edited as part of this packaging step.

## Repository contents

- `src/delhi/` — Delhi experiment source code, including temporal, spatial, fusion, prediction, preprocessing, and training modules.
- `src/greece/` — Greek-dataset experiment source code.
- `config/` — project configuration files, if present in the original package.
- `tests/` — automated tests.
- `results/` — selected lightweight result summaries, logs, and experiment configurations where present.
- `EXPERIMENT_MANIFEST.json` — experiment manifest from the supplied package.
- `DATA_VALIDATION_REPORT.md` and `FINAL_AUDIT_REPORT.md` — reports from the supplied package.
- `requirements.txt` — Python dependencies.

## Data availability

Large datasets and generated binary artifacts are intentionally not stored in this repository. Obtain each dataset from its original source and confirm that its license permits your intended use and redistribution.

Place data files in the paths expected by the relevant scripts. The original package references files such as:

- `data/delhi-weather-aqi-2025.csv`
- `data/greek_urban_air_quality_2020_2024.csv`
- `data/delhi/quality_controlled_sequences_fixed.pt`

Some source files also reference dataset paths under `src/delhi/data/` or `src/greece/data/`. Check the relevant configuration/preprocessing scripts before running an experiment and update only local data-path settings as needed; do not change model logic to accommodate the repository layout.

## Environment setup

Use Python in a virtual environment. From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency versions in `requirements.txt` are the ones supplied with the project. For strict reproducibility, record your Python version, OS, CUDA version, GPU model, and exact installed package versions alongside results.

## Running experiments

The supplied project README documents these entry points:

```bash
python run_kaggle.py --experiment baseline
python run_kaggle.py --experiment multiseed
python run_kaggle.py --experiment ablation
python run_kaggle.py --experiment greece
python run_kaggle.py --experiment benchmark
```

Run commands only after placing the required dataset files in the expected paths and checking the configuration for the selected experiment. Hardware availability and runtime can affect whether an experiment completes.

## Tests

Run the available test suite from the repository root:

```bash
python -m pytest
```

If `pytest` is not installed in your environment, install it separately before running the tests. Test results should be reported only after you execute them in your own environment.

## Results and reporting

The included lightweight result files are preserved artifacts from the supplied archive; they are not independently revalidated by this packaging step. When reporting results in a paper, ensure the reported metrics correspond to the correct dataset split, forecast horizon, seed aggregation method, and exact experiment configuration.

## Citation

If you use this implementation in a publication, cite the associated DySTCA paper once its final bibliographic details or DOI are available. Add the final citation here before making the repository public.

## License

No license is asserted by this packaging step. Add a license only after confirming that you own or have permission to license the code and that third-party code/data terms are compatible. Without a license, standard copyright restrictions apply.

# Algorithmic Trading & AI Research Framework

A reproducible research framework to evaluate classic technical strategies and ML models for predicting price direction. This repo is built for research, not live trading.

---

## What This Project Does

- Downloads OHLCV data with `yfinance` and stores it in `data/raw/`.
- Runs rule-based backtests across multiple instruments, intervals, and parameter grids.
- Trains ML models with strict leave-one-dataset-out evaluation.
- Writes all metrics and artifacts to structured, queryable files for analysis.

---

## Quickstart

1. Create a virtual environment (optional but recommended).

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies. Note the filename is `requirments.txt` in this repo.

```bash
pip install -r requirments.txt
```

3. Configure instruments, intervals, and parameters in `config/config.yaml`.

4. Download price data into `data/raw/`.

```bash
python utils/data_loader.py
```

Note: the loader clears the entire `data/` folder before downloading, so it will remove any cached ML features.

5. Run the classic strategy backtests.

```bash
python run_all.py
```

6. Run the ML pipeline (leave-one-dataset-out).

```bash
python run_ml.py prepare
python run_ml.py run --test-dataset AAPL_1d
```

Run all datasets:

```bash
python run_ml.py run-all
```

7. Analyze results in the notebook:

`analysis/results_analyser.ipynb`

---

## Configuration

All core settings live in `config/config.yaml`:

- Instruments: `stocks`, `forex`, `indices`, `crypto`
- Time intervals: `intervals` and `periods`
- Strategy parameter grids: `params`
- Risk sizing settings: `risk`
- ML settings: `ml`

---

## Classic Backtesting Pipeline

**Inputs**

- Raw price CSVs in `data/raw/*.csv` (created by `utils/data_loader.py`).
- Strategy definitions in `strategies/`.
- Parameter grids from `config/config.yaml`.

**Run**

```bash
python run_all.py
```

**Outputs**

- `database/results.parquet` with one row per strategy/asset/interval/parameter set.

---

## Risk Optimization (Optional)

Tune the risk sizing config using a coordinate search over a predefined grid.

```bash
python optimize_risk_params.py --symbols AAPL,EURUSD=X,USO --intervals 1d,1h,15m --passes 1
```

Write the best settings back into `config/config.yaml`:

```bash
python optimize_risk_params.py --apply-best
```

Output log:

- `analysis/results/risk_optimization.csv`

---

## ML Pipeline (Leave-One-Dataset-Out)

**Prepare features**

```bash
python run_ml.py prepare
```

Use `--force` to rebuild the cache:

```bash
python run_ml.py prepare --force
```

**Run a single test dataset**

```bash
python run_ml.py run --test-dataset AAPL_1d
```

**Run all datasets**

```bash
python run_ml.py run-all
```

**Model selection**

```bash
python run_ml.py run --test-dataset AAPL_1d --models logistic,random_forest
```

**Notes**

- The ML feature cache keeps the most recent `ml.test_candles` rows per dataset.
- Datasets that do not have enough rows are skipped and recorded in `data/features/manifest.json`.
- Training requires at least 2 datasets after filtering (GroupKFold).

---

## Outputs And Artifacts

| Path | Description |
| --- | --- |
| `data/raw/*.csv` | Raw OHLCV data downloaded from `yfinance`. |
| `database/results.parquet` | Classic backtest results. |
| `analysis/results/*.csv` | Exported analysis results from experiments. |
| `data/features/*.parquet` | Feature cache for ML. |
| `data/features/manifest.json` | Feature metadata and dataset inventory. |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/model.joblib` | Trained ML model artifact. |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/metadata.json` | Run metadata and hyperparameters. |
| `analysis/results/ml/{run_id}/{test_dataset}_{model_name}.csv` | Per-candle ML predictions and signals. |
| `analysis/results/ml/{run_id}/summary.csv` | ML run summary. |
| `database/ml_results.parquet` | Aggregated ML metrics. |

---

## Project Layout

| Path | Purpose |
| --- | --- |
| `utils/data_loader.py` | Downloads data from `yfinance` into `data/raw/`. |
| `strategies/` | Rule-based trading strategies. |
| `run_all.py` | Runs the parameter grid backtests. |
| `run_ml.py` | ML training and evaluation CLI. |
| `ml/` | Feature engineering, models, evaluation logic, persistence. |
| `analysis/results_analyser.ipynb` | Notebook for exploring results. |

---

## Notes

- This project is for research only. It is not a live trading system.
- Results depend on the data window defined in `config/config.yaml`.
- `utils/data_loader.py` truncates each dataset to the most recent 5000 rows so all datasets have equal length.

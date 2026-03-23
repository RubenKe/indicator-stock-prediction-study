# Algorithmic Trading & AI Research Framework

A reproducible research framework to evaluate classic technical strategies and ML models for predicting price direction. This repo is built for research, not live trading.

---

## What This Project Does

- Downloads OHLCV data with `yfinance` and stores it in `data/raw/`.
- Runs rule-based backtests across multiple instruments, intervals, and parameter grids.
- Trains ML models with strict leave-one-dataset-out evaluation.
- Writes all metrics and artifacts to structured, queryable files for analysis.

---

## At A Glance

**Classic (rule-based)**

- Deterministic strategy rules.
- No training phase.
- Results: `database/results.parquet`.
- Graded on trading metrics (return, sharpe, drawdown, win rate, trades).

**ML (AI)**

- Trains models to predict next-candle direction.
- Leave-one-dataset-out evaluation.
- Results: a summary table and per-candle predictions.
- Graded on prediction metrics (ROC AUC, accuracy) and trading metrics.

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

2. Install dependencies. This sets up all required Python packages.

```bash
pip install -r requirements.txt
```

3. Configure instruments, intervals, and parameters in `config/config.yaml`. This controls what markets and timeframes are tested.

4. Download price data into `data/raw/`. This fetches the latest OHLCV data for your config.

```bash
python utils/data_loader.py
```

This clears only `data/raw` (downloaded market data), not cached ML features.

5. Run the classic strategy backtests. This evaluates every rule-based strategy across your config grid.

```bash
python run_all.py
```

You should see `database/results.parquet` updated or created.

6. Run the ML pipeline (leave-one-dataset-out). This trains ML models on all datasets except one, then tests on the held-out dataset.

```bash
python run_ml.py prepare
python run_ml.py run --test-dataset AAPL_1d
```

Run all datasets:

```bash
python run_ml.py run-all
```

You should see `data/features/manifest.json`, `database/ml_results.parquet`, and `analysis/results/ml/`.

7. Analyze results in the notebook:

`analysis/results_analyser.ipynb`

---

## Configuration

All core settings live in `config/config.yaml`:

- Instruments: `stocks`, `forex`, `indices`, `crypto`
- Time intervals: `intervals` and `periods`
- Strategy parameter grids: `params`
- Execution costs: `commission`, `slippage`
- Risk sizing settings: `risk`
- ML settings: `ml`

---

## Classic Backtesting Pipeline

**Inputs**

- Raw price CSVs in `data/raw/*.csv` (created by `utils/data_loader.py`).
- Strategy definitions in `strategies/`.
- Parameter grids from `config/config.yaml`.
- Execution costs from `commission` and `slippage`.

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

**Models used**

- `logistic`: logistic regression baseline (linear, fast, interpretable).
- `random_forest`: ensemble of decision trees (non-linear, robust).
- `gradient_boosting`: boosted trees (strong accuracy, more sensitive to tuning).

**How the AI works**

1. Features are engineered from OHLCV data (`ml/features.py`).
2. The target label is whether the next candle return is positive (`y`).
3. Models train on all datasets except one and validate with GroupKFold.
4. The held-out dataset is used for final testing.
5. Model probabilities are turned into signals using `threshold_long` / `threshold_short`.
6. The signal stream is backtested with `commission` and `slippage` costs.

**Notes**

- The ML feature cache keeps the most recent `ml.test_candles` rows per dataset.
- Datasets that do not have enough rows are skipped and recorded in `data/features/manifest.json`.
- Training requires at least 2 datasets after filtering (GroupKFold).

---

## When Results Are Produced

**Classic results**

- Created by `python run_all.py`.
- Written to `database/results.parquet`.

**ML results**

- Feature cache created by `python run_ml.py prepare`.
- Model runs created by `python run_ml.py run` or `python run_ml.py run-all`.
- Written to `database/ml_results.parquet` and `analysis/results/ml/`.

## Outputs And Artifacts

| Path | Description |
| --- | --- |
| `data/raw/*.csv` | Raw OHLCV data downloaded from `yfinance`. |
| `database/results.parquet` | Classic backtest results (created by `python run_all.py`). |
| `analysis/results/*.csv` | Exported analysis results from experiments. |
| `data/features/*.parquet` | Feature cache for ML (created by `python run_ml.py prepare`). |
| `data/features/manifest.json` | Feature metadata and dataset inventory (created by `python run_ml.py prepare`). |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/model.joblib` | Trained ML model artifact. |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/metadata.json` | Run metadata and hyperparameters. |
| `analysis/results/ml/{run_id}/{test_dataset}_{model_name}.csv` | Per-candle ML predictions and signals (created by `python run_ml.py run` / `run-all`). |
| `analysis/results/ml/{run_id}/summary.csv` | ML run summary (created by `python run_ml.py run` / `run-all`). |
| `database/ml_results.parquet` | Aggregated ML metrics (created by `python run_ml.py run` / `run-all`). |

---

## Project Layout

**Directories**

- `analysis/`: notebooks and exported analysis results.
- `config/`: configuration files (symbols, intervals, params, ML settings).
- `data/`: local datasets.
- `data/raw/`: downloaded OHLCV CSVs from `yfinance`.
- `data/features/`: ML feature cache and `manifest.json`.
- `database/`: parquet result stores (classic and ML).
- `ml/`: feature engineering, model definitions, training, evaluation, persistence.
- `models/`: saved ML model artifacts.
- `strategies/`: rule-based strategy implementations.
- `tests/`: tests (if present).
- `utils/`: data loading, logging, and helper scripts.

**Key files**

- `utils/data_loader.py`: downloads data into `data/raw/`.
- `run_all.py`: runs classic strategy backtests over the parameter grid.
- `run_ml.py`: ML training and evaluation CLI.
- `analysis/results_analyser.ipynb`: notebook for exploring results.

---

## Notes

- This project is for research only. It is not a live trading system.
- Results depend on the data window defined in `config/config.yaml`.
- `utils/data_loader.py` truncates each dataset to the most recent 5000 rows so all datasets have equal length.

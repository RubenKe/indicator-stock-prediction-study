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
- Currently testing 13 rule-based strategies (see `config/config.yaml` `params`).
- Results: `database/results.parquet`.
- Graded on trading metrics (return, sharpe, drawdown, win rate, trades).

**ML (AI)**

- Trains models to predict next-candle direction.
- Leave-one-dataset-out evaluation.
- Results: a summary table and per-candle predictions.
- Graded on prediction metrics (ROC AUC, accuracy) and trading metrics.

---

## Quickstart

### Setup (One-Time)

1. **Create a virtual environment** (recommended to isolate dependencies).

   ```bash
   python -m venv .venv
   ```

   - **Windows**: `.venv\Scripts\activate`
   - **macOS/Linux**: `source .venv/bin/activate`

2. **Install dependencies**.

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the project** (optional, but customize if needed).
   - Edit `config/config.yaml` to set instruments (stocks, forex, crypto), time intervals, strategy parameters, etc.

4. **Download price data**.
   - Fetches OHLCV data for configured instruments and saves to `data/raw/` in Parquet format.

   ```bash
   python utils/data_loader.py
   ```

   - Note: This only clears `data/raw/` (market data), not ML feature caches.

### Run Classic (Rule-Based) Strategies

5. **Run all classic strategy backtests**.
   - Evaluates every rule-based strategy across your parameter grid.
   - Results saved to `database/results.parquet`.

   ```bash
   python run_all.py
   ```

### Train All AI/ML Models

6. **Prepare ML features** (one-time after data download).
   - Processes raw data into ML-ready features and caches them.

   ```bash
   python run_ml.py prepare
   ```

7. **Train ML models on all datasets**.
   - Uses leave-one-dataset-out evaluation: trains on all datasets except one, tests on the held-out one.
   - Default models: logistic regression, random forest, gradient boosting.
   - Models and predictions saved to `models/ml_registry/` and `analysis/results/ml/`.

   ```bash
   python run_ml.py run-all
   ```

   - For a single dataset test: `python run_ml.py run --test-dataset AAPL_1d`
   - For a quick test: `python run_ml.py quick`
   - Customize models: `--models "logistic,random_forest"`
   - Use parallel processing: `--n-jobs -1`

### Retrieve and Analyze Results

8. **View ML results**.
   - **Summary metrics**: `database/ml_results.csv` (ROC AUC, accuracy, trading returns, Sharpe, etc.)
   - **Detailed predictions**: `analysis/results/ml/<run_folder>/<dataset>_<model>.csv` (per-candle predictions)
   - **Model artifacts**: `models/ml_registry/<run_folder>/<dataset>/<model>/` (saved models and metadata)

9. **Analyze in the notebook**.
   - Open `analysis/ml_results_analysis.ipynb` for visualizations, comparisons, and deeper analysis.
   - Loads results from the above files and generates plots/metrics.

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

- Raw price Parquet files in `data/raw/*.parquet` (created by `utils/data_loader.py`).
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

## Run A Single Strategy With Full Results

If you want the **full per-bar equity curve** (not just summary stats), use the helper:

```python
from utils.single_strategy_runner import run_single_strategy

out = run_single_strategy(
    strategy_name="DMAC",
    symbol="AAPL",
    interval="1d",
    param_dict={"pfast": 10, "pslow": 200, "adx_period": 14, "adx_threshold": 20},
)
```

**What you get back**

- `out["summary"]`: dict with the same summary metrics stored in `database/results.parquet`.
- `out["equity_curve"]`: `DataFrame` with `equity` and `returns` indexed by datetime.
- `out["price_df"]`: the price data used for the run.
- `out["result"]`: the raw Backtrader strategy instance.

**Plot the full equity curve**

```python
out["equity_curve"]["equity"].plot(figsize=(12, 4), title="Equity Curve")
```

Notes:
- This uses the same config values from `config/config.yaml` by default (`commission`, `slippage`, `sizer`, `risk`, `benchmark_symbol`).
- You can override them via the optional arguments in `run_single_strategy(...)`.

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

**Quick test (fast, 1 dataset, 1 model)**

```bash
python run_ml.py quick
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
- Crypto symbols are excluded from ML by default (to avoid short history issues).
- Training requires at least 2 datasets after filtering (GroupKFold).
- Default ML execution includes a 0.1% commission per trade and no extra slippage.

---

## When Results Are Produced

**Classic results**

- Created by `python run_all.py`.
- Written to `database/results.parquet`.

**ML results**

- Feature cache created by `python run_ml.py prepare`.
- Model runs created by `python run_ml.py run` or `python run_ml.py run-all`.
- Written to `database/ml_results.csv` and `analysis/results/ml/`.

## Outputs And Artifacts

| Path | Description |
| --- | --- |
| `data/raw/*.parquet` | Raw OHLCV data downloaded from `yfinance`. |
| `database/results.parquet` | Classic backtest results (created by `python run_all.py`). |
| `analysis/results/*.csv` | Exported analysis results from experiments. |
| `data/features/*.parquet` | Feature cache for ML (created by `python run_ml.py prepare`). |
| `data/features/manifest.json` | Feature metadata and dataset inventory (created by `python run_ml.py prepare`). |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/model.joblib` | Trained ML model artifact. |
| `models/ml_registry/{run_id}/{test_dataset}/{model_name}/metadata.json` | Run metadata and hyperparameters. |
| `analysis/results/ml/{run_id}/{test_dataset}_{model_name}.csv` | Per-candle ML predictions and signals (created by `python run_ml.py run` / `run-all`). |
| `analysis/results/ml/{run_id}/summary.csv` | ML run summary (created by `python run_ml.py run` / `run-all`). |
| `database/ml_results.csv` | Aggregated ML metrics (created by `python run_ml.py run` / `run-all`). |

---

## Project Layout

**Directories**

- `analysis/`: notebooks and exported analysis results.
- `config/`: configuration files (symbols, intervals, params, ML settings).
- `data/`: local datasets.
- `data/raw/`: downloaded OHLCV Parquet files from `yfinance`.
- `data/features/`: ML feature cache and `manifest.json`.
- `database/`: result stores (classic Parquet, ML CSV).
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

## ML Run Folder Names

Each ML run is saved under `models/ml_registry/{run_id}/` and `analysis/results/ml/{run_id}/`.
The `run_id` is human-readable and includes a timestamp, model list, seed, and a short hash.

Example:

```
ml_20260323T112233Z_models-logistic-random_forest_seed-42_a1b2c3d4
```

---

## Notes

- This project is for research only. It is not a live trading system.
- Results depend on the data window defined in `config/config.yaml`.
- `utils/data_loader.py` truncates each dataset to the most recent 5000 rows so all datasets have equal length.

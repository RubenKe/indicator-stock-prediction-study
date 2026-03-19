# Algorithmic Trading & AI Research Framework

## Project description

The goal of this project is to systematically evaluate whether:
- traditional algorithmic trading strategies
- and artificial intelligence models  

can contribute to predicting stock price movements and achieving effective or profitable trading strategies.

The project is designed as a **reproducible research framework**, not as a live trading system.

---

## Research goals

- Evaluate classic technical trading strategies (non-AI)
- Compare their performance with AI-based models
- Test strategies across:
  - multiple markets (stocks, crypto, forex, indices)
  - multiple time intervals (e.g. 15m, 1h, 1d)
  - multiple parameter combinations
- Store all results in a structured format for later analysis

---

## Setup

Install dependencies:

```bash
pip install -r requirments.txt
```

---

## ML Pipeline (Leave-One-Dataset-Out)

This repository now includes a full machine learning workflow with:

- deterministic feature engineering from `data/raw/*.csv`
- strict leave-one-dataset-out evaluation
- three models: logistic regression, random forest, gradient boosting
- grouped hyperparameter tuning (GroupKFold on training datasets only)
- probability-to-signal conversion with fixed thresholds
- saved model artifacts + metadata + per-candle predictions

### 1) Prepare feature cache

```bash
python run_ml.py prepare
```

Use `--force` to rebuild from scratch:

```bash
python run_ml.py prepare --force
```

### 2) Run one test dataset

```bash
python run_ml.py run --test-dataset AAPL_15m
```

Optional model subset:

```bash
python run_ml.py run --test-dataset AAPL_15m --models logistic,random_forest
```

### 3) Run all datasets (LOO)

```bash
python run_ml.py run-all
```

Limit number of test datasets for quick experiments:

```bash
python run_ml.py run-all --max-tests 3
```

---

## ML Artifacts

- Feature cache:
  - `data/features/*.parquet`
  - `data/features/manifest.json`
- Model registry:
  - `models/ml_registry/{run_id}/{test_dataset}/{model_name}/model.joblib`
  - `models/ml_registry/{run_id}/{test_dataset}/{model_name}/metadata.json`
- Prediction exports:
  - `analysis/results/ml/{run_id}/{test_dataset}_{model_name}.csv`
- Run summary:
  - `analysis/results/ml/{run_id}/summary.csv`
- Aggregated ML results:
  - `database/ml_results.parquet`

---

## Load A Saved Model

```python
import joblib
model = joblib.load("models/ml_registry/<run_id>/<test_dataset>/<model_name>/model.joblib")
```

The paired `metadata.json` contains best parameters, train/test datasets, thresholds, commission, and feature schema.

---

## ML Metrics in Output

Each ML run stores:

- `total_return_pct`
- `buy_hold_return_pct`
- `excess_return_pct`
- `sharpe`
- `max_drawdown_pct`
- `annualized_return_pct`
- `num_trades`
- `signal_coverage`
- classification diagnostics (`test_accuracy`, `test_roc_auc`)

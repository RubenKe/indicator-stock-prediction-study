from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


RESULT_COLUMNS = [
    "run_id",
    "created_at_utc",
    "experiment_key",
    "experiment_version",
    "test_dataset",
    "symbol",
    "interval",
    "model_name",
    "profile",
    "seed",
    "n_train_datasets",
    "train_datasets_json",
    "n_train_rows",
    "n_test_rows",
    "cv_splits",
    "cv_best_roc_auc",
    "cv_best_accuracy",
    "test_roc_auc",
    "test_accuracy",
    "best_params_json",
    "threshold_long",
    "threshold_short",
    "commission",
    "total_return_pct",
    "buy_hold_return_pct",
    "excess_return_pct",
    "sharpe",
    "max_drawdown_pct",
    "annualized_return_pct",
    "num_trades",
    "signal_coverage",
    "feature_columns_json",
    "model_path",
    "metadata_path",
    "candle_path",
    "summary_path",
]


RESULT_DTYPES = {
    "run_id": "string",
    "created_at_utc": "string",
    "experiment_key": "string",
    "experiment_version": "string",
    "test_dataset": "string",
    "symbol": "string",
    "interval": "string",
    "model_name": "string",
    "profile": "string",
    "seed": "Int64",
    "n_train_datasets": "Int64",
    "train_datasets_json": "string",
    "n_train_rows": "Int64",
    "n_test_rows": "Int64",
    "cv_splits": "Int64",
    "cv_best_roc_auc": "float64",
    "cv_best_accuracy": "float64",
    "test_roc_auc": "float64",
    "test_accuracy": "float64",
    "best_params_json": "string",
    "threshold_long": "float64",
    "threshold_short": "float64",
    "commission": "float64",
    "total_return_pct": "float64",
    "buy_hold_return_pct": "float64",
    "excess_return_pct": "float64",
    "sharpe": "float64",
    "max_drawdown_pct": "float64",
    "annualized_return_pct": "float64",
    "num_trades": "Int64",
    "signal_coverage": "float64",
    "feature_columns_json": "string",
    "model_path": "string",
    "metadata_path": "string",
    "candle_path": "string",
    "summary_path": "string",
}


def _empty_results_frame() -> pd.DataFrame:
    data = {col: pd.Series(dtype=RESULT_DTYPES[col]) for col in RESULT_COLUMNS}
    return pd.DataFrame(data)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in RESULT_DTYPES.items():
        if col not in out.columns:
            out[col] = pd.NA
        if dtype == "float64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif dtype == "Int64":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        else:
            out[col] = out[col].astype("string")
    return out[RESULT_COLUMNS]


def bootstrap_results_file(results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        _empty_results_frame().to_parquet(results_path, index=False)


def load_existing_experiment_keys(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    df = pd.read_parquet(results_path, columns=["experiment_key"])
    return set(df["experiment_key"].dropna().astype(str).tolist())


def append_result_rows(
    results_path: Path,
    rows: list[dict[str, Any]],
    dedupe_on_experiment_key: bool = True,
) -> int:
    if not rows:
        return 0

    bootstrap_results_file(results_path)
    existing = pd.read_parquet(results_path)
    incoming = pd.DataFrame(rows)

    for col in RESULT_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = None
    incoming = _coerce_dtypes(incoming)
    existing = _coerce_dtypes(existing)

    if dedupe_on_experiment_key and not existing.empty:
        existing_keys = set(existing["experiment_key"].dropna().astype(str))
        incoming = incoming[~incoming["experiment_key"].astype(str).isin(existing_keys)]

    if incoming.empty:
        return 0

    if existing.empty:
        out = incoming.copy()
    else:
        out = pd.concat([existing, incoming], ignore_index=True)
        out = _coerce_dtypes(out)
    out.to_parquet(results_path, index=False)
    return int(len(incoming))


def save_model_artifacts(
    run_id: str,
    test_dataset: str,
    model_name: str,
    estimator,
    candle_df: pd.DataFrame,
    metadata: dict[str, Any],
    model_registry_dir: Path,
    analysis_ml_dir: Path,
) -> dict[str, Path]:
    model_dir = model_registry_dir / run_id / test_dataset / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(estimator, model_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    run_analysis_dir = analysis_ml_dir / run_id
    run_analysis_dir.mkdir(parents=True, exist_ok=True)
    candle_path = run_analysis_dir / f"{test_dataset}_{model_name}.csv"
    candle_df.to_csv(candle_path, index=False)

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "candle_path": candle_path,
    }


def save_run_summary(
    run_id: str,
    summary_rows: list[dict[str, Any]],
    analysis_ml_dir: Path,
) -> Path:
    run_analysis_dir = analysis_ml_dir / run_id
    run_analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_analysis_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return summary_path

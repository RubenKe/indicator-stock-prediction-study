from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from .features import FEATURE_COLUMNS
from .models import build_estimator, get_param_grid
from .types import RunConfig
from .utils import annualization_factor


def build_train_test_frames(
    dataset_frames: dict[str, pd.DataFrame],
    test_dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if test_dataset not in dataset_frames:
        raise KeyError(f"Unknown test dataset '{test_dataset}'.")

    train_ids = sorted([k for k in dataset_frames.keys() if k != test_dataset])
    if not train_ids:
        raise RuntimeError("No training datasets available after selecting test dataset.")

    test_df = dataset_frames[test_dataset].copy()
    train_df = pd.concat([dataset_frames[ds] for ds in train_ids], axis=0).copy()
    return train_df, test_df, train_ids


def build_group_kfold_splits(
    groups: pd.Series | np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_arr = np.asarray(groups)
    cv = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups_arr))
    return list(cv.split(idx, groups=groups_arr))


def _refit_strategy(cv_results: dict[str, Any]) -> int:
    roc = np.asarray(cv_results["mean_test_roc_auc"], dtype=float)
    acc = np.asarray(cv_results["mean_test_accuracy"], dtype=float)
    roc = np.nan_to_num(roc, nan=-np.inf)
    acc = np.nan_to_num(acc, nan=-np.inf)

    best_roc = np.max(roc)
    candidates = np.where(roc == best_roc)[0]
    if len(candidates) == 1:
        return int(candidates[0])

    best_acc_idx = candidates[np.argmax(acc[candidates])]
    return int(best_acc_idx)


def tune_model_with_group_kfold(
    model_name: str,
    profile: str,
    train_df: pd.DataFrame,
    run_cfg: RunConfig,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    if "dataset_id" not in train_df.columns:
        raise ValueError("train_df must include 'dataset_id' column for grouped CV.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["y"].astype(int)
    groups = train_df["dataset_id"].astype(str)

    unique_groups = groups.nunique()
    cv_splits = min(run_cfg.cv_splits, unique_groups)
    if cv_splits < 2:
        raise ValueError(
            f"GroupKFold requires at least 2 groups. Found {unique_groups}."
        )

    estimator = build_estimator(model_name, random_state=seed, n_jobs=n_jobs)
    param_grid = get_param_grid(model_name, profile=profile)
    cv = GroupKFold(n_splits=cv_splits)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"},
        refit=_refit_strategy,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    search.fit(X_train, y_train, groups=groups)

    best_idx = int(search.best_index_)
    cv_roc_auc = float(search.cv_results_["mean_test_roc_auc"][best_idx])
    cv_accuracy = float(search.cv_results_["mean_test_accuracy"][best_idx])

    return {
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_best_roc_auc": cv_roc_auc,
        "cv_best_accuracy": cv_accuracy,
        "cv_splits": int(cv_splits),
        "n_train_rows": int(len(train_df)),
    }


def build_signal_frame(
    probabilities: np.ndarray,
    next_returns: pd.Series,
    threshold_long: float,
    threshold_short: float,
    commission: float,
    slippage: float,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    probs = np.asarray(probabilities, dtype=float)
    signals_raw = np.where(probs > threshold_long, 1, np.where(probs < threshold_short, -1, 0))
    signal_series = pd.Series(signals_raw, index=index, dtype=int)

    position = signal_series.shift(1).fillna(0).astype(int)
    returns = pd.Series(next_returns, index=index, dtype=float)
    gross_ret = position * returns
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * float(commission + slippage)
    net_ret = gross_ret - cost
    equity_curve = (1.0 + net_ret).cumprod()

    return pd.DataFrame(
        {
            "p_up": probs,
            "signal_raw": signal_series,
            "position": position,
            "next_return": returns,
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "equity_curve": equity_curve,
        },
        index=index,
    )


def _safe_roc_auc(y_true: pd.Series, p_up: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return 0.0
    return float(roc_auc_score(y_true, p_up))


def _compute_backtest_metrics(
    signal_df: pd.DataFrame,
    y_true: pd.Series,
    interval: str,
) -> dict[str, float]:
    periods_per_year = annualization_factor(interval)
    net_ret = signal_df["net_ret"].astype(float)
    next_ret = signal_df["next_return"].astype(float)
    equity_curve = signal_df["equity_curve"].astype(float)
    buy_hold_curve = (1.0 + next_ret).cumprod()

    total_return = float((equity_curve.iloc[-1] - 1.0) * 100.0)
    buy_hold_return = float((buy_hold_curve.iloc[-1] - 1.0) * 100.0)
    excess_return = total_return - buy_hold_return

    std = float(net_ret.std(ddof=0))
    if std > 0:
        sharpe = float((net_ret.mean() / std) * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown_pct = float(abs(drawdown.min()) * 100.0)

    years = len(signal_df) / periods_per_year
    if years > 0 and equity_curve.iloc[-1] > 0:
        annualized_return = float((equity_curve.iloc[-1] ** (1.0 / years) - 1.0) * 100.0)
    else:
        annualized_return = 0.0

    entries = ((signal_df["position"] != 0) & (signal_df["position"].shift(1).fillna(0) == 0))
    num_trades = int(entries.sum())
    signal_coverage = float((signal_df["position"] != 0).mean())

    preds = (signal_df["p_up"].values >= 0.5).astype(int)
    test_accuracy = float(accuracy_score(y_true, preds))
    test_roc_auc = _safe_roc_auc(y_true, signal_df["p_up"].values)

    return {
        "total_return_pct": total_return,
        "buy_hold_return_pct": buy_hold_return,
        "excess_return_pct": excess_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_return_pct": annualized_return,
        "num_trades": num_trades,
        "signal_coverage": signal_coverage,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
    }


def evaluate_model_on_test(
    estimator,
    test_df: pd.DataFrame,
    run_cfg: RunConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["y"].astype(int)
    probs = estimator.predict_proba(X_test)[:, 1]

    signal_df = build_signal_frame(
        probabilities=probs,
        next_returns=test_df["next_return"],
        threshold_long=run_cfg.threshold_long,
        threshold_short=run_cfg.threshold_short,
        commission=run_cfg.commission,
        slippage=run_cfg.slippage,
        index=test_df.index,
    )
    metrics = _compute_backtest_metrics(
        signal_df=signal_df,
        y_true=y_test,
        interval=str(test_df["interval"].iloc[0]),
    )

    candle_df = pd.DataFrame(
        {
            "timestamp": test_df.index,
            "symbol": test_df["symbol"].values,
            "interval": test_df["interval"].values,
            "p_up": signal_df["p_up"].values,
            "signal_raw": signal_df["signal_raw"].values,
            "position": signal_df["position"].values,
            "next_return": signal_df["next_return"].values,
            "gross_ret": signal_df["gross_ret"].values,
            "cost": signal_df["cost"].values,
            "net_ret": signal_df["net_ret"].values,
            "equity_curve": signal_df["equity_curve"].values,
        }
    )
    return candle_df, metrics

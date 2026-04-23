from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "analysis" / "results" / "ml" / "analyzer_output"

NUMERIC_COLUMNS = [
    "cv_best_roc_auc",
    "cv_best_accuracy",
    "test_roc_auc",
    "test_accuracy",
    "total_return_pct",
    "buy_hold_return_pct",
    "excess_return_pct",
    "sharpe",
    "max_drawdown_pct",
    "annualized_return_pct",
    "signal_coverage",
    "commission",
    "slippage",
    "sp500_buy_hold_return_pct",
    "sp500_excess_return_pct",
]

INT_COLUMNS = [
    "num_trades",
    "seed",
    "n_train_datasets",
    "n_train_rows",
    "n_test_rows",
    "cv_splits",
]

DISPLAY_COLUMNS = [
    "test_dataset",
    "symbol",
    "interval",
    "model_name",
    "test_roc_auc",
    "test_accuracy",
    "total_return_pct",
    "excess_return_pct",
    "sp500_excess_return_pct",
    "sharpe",
    "max_drawdown_pct",
    "num_trades",
    "run_id",
]

PLOT_METRICS = [
    "total_return_pct",
    "excess_return_pct",
    "sp500_excess_return_pct",
    "buy_hold_return_pct",
]

MODEL_SCORE_PLOT_METRICS = [
    "test_roc_auc",
    "test_accuracy",
    "excess_return_pct",
    "sharpe",
]

CORRELATION_METRICS = [
    "test_roc_auc",
    "test_accuracy",
    "cv_best_roc_auc",
    "cv_best_accuracy",
    "total_return_pct",
    "buy_hold_return_pct",
    "excess_return_pct",
    "sp500_buy_hold_return_pct",
    "sp500_excess_return_pct",
    "sharpe",
    "max_drawdown_pct",
    "annualized_return_pct",
    "signal_coverage",
    "num_trades",
]


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ML results file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)

    return add_sp500_outperformance(df)


def add_sp500_outperformance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"run_id", "interval", "symbol", "buy_hold_return_pct", "total_return_pct"}
    if not required.issubset(out.columns):
        if "sp500_buy_hold_return_pct" not in out.columns:
            out["sp500_buy_hold_return_pct"] = np.nan
        if "sp500_excess_return_pct" not in out.columns:
            out["sp500_excess_return_pct"] = np.nan
        return out

    sp500_rows = out[out["symbol"] == "^GSPC"].copy()
    if sp500_rows.empty:
        out["sp500_buy_hold_return_pct"] = np.nan
        out["sp500_excess_return_pct"] = np.nan
        return out

    baseline = (
        sp500_rows.groupby(["run_id", "interval"], as_index=False)["buy_hold_return_pct"]
        .mean()
        .rename(columns={"buy_hold_return_pct": "sp500_buy_hold_return_pct"})
    )
    out = out.merge(baseline, on=["run_id", "interval"], how="left")
    out["sp500_excess_return_pct"] = out["total_return_pct"] - out["sp500_buy_hold_return_pct"]
    return out


def resolve_results_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)

    candidates = [
        PROJECT_ROOT / "database" / "ml_results.csv",
        PROJECT_ROOT / "database" / "ml_results.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def apply_filters(
    df: pd.DataFrame,
    run_id: str | None = None,
    model_names: list[str] | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if run_id:
        out = out[out["run_id"] == run_id]
    if model_names:
        out = out[out["model_name"].isin(model_names)]
    if symbols:
        out = out[out["symbol"].isin(symbols)]
    if intervals:
        out = out[out["interval"].isin(intervals)]
    return out.reset_index(drop=True)


def _ordered_unique(values: pd.Series) -> str:
    clean = [str(v) for v in values.dropna().tolist()]
    return ", ".join(dict.fromkeys(clean))


def build_overview(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "runs": int(df["run_id"].nunique()) if "run_id" in df.columns else 0,
        "datasets": int(df["test_dataset"].nunique()) if "test_dataset" in df.columns else 0,
        "models": _ordered_unique(df["model_name"]) if "model_name" in df.columns else "",
        "symbols": _ordered_unique(df["symbol"]) if "symbol" in df.columns else "",
        "intervals": _ordered_unique(df["interval"]) if "interval" in df.columns else "",
        "created_min": df["created_at_utc"].min() if "created_at_utc" in df.columns else pd.NaT,
        "created_max": df["created_at_utc"].max() if "created_at_utc" in df.columns else pd.NaT,
        "mean_test_roc_auc": df["test_roc_auc"].mean() if "test_roc_auc" in df.columns else np.nan,
        "mean_test_accuracy": df["test_accuracy"].mean() if "test_accuracy" in df.columns else np.nan,
        "mean_total_return_pct": df["total_return_pct"].mean() if "total_return_pct" in df.columns else np.nan,
        "mean_buy_hold_return_pct": df["buy_hold_return_pct"].mean() if "buy_hold_return_pct" in df.columns else np.nan,
        "mean_excess_return_pct": df["excess_return_pct"].mean() if "excess_return_pct" in df.columns else np.nan,
        "mean_sp500_buy_hold_return_pct": df["sp500_buy_hold_return_pct"].mean() if "sp500_buy_hold_return_pct" in df.columns else np.nan,
        "mean_sp500_excess_return_pct": df["sp500_excess_return_pct"].mean() if "sp500_excess_return_pct" in df.columns else np.nan,
        "mean_sharpe": df["sharpe"].mean() if "sharpe" in df.columns else np.nan,
        "mean_max_drawdown_pct": df["max_drawdown_pct"].mean() if "max_drawdown_pct" in df.columns else np.nan,
        "mean_num_trades": df["num_trades"].mean() if "num_trades" in df.columns else np.nan,
        "mean_signal_coverage": df["signal_coverage"].mean() if "signal_coverage" in df.columns else np.nan,
    }


def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model_name", as_index=False)
        .agg(
            experiments=("experiment_key", "count"),
            datasets=("test_dataset", "nunique"),
            mean_test_roc_auc=("test_roc_auc", "mean"),
            std_test_roc_auc=("test_roc_auc", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_total_return_pct=("total_return_pct", "mean"),
            median_total_return_pct=("total_return_pct", "median"),
            mean_excess_return_pct=("excess_return_pct", "mean"),
            median_excess_return_pct=("excess_return_pct", "median"),
            positive_excess_rate=("excess_return_pct", lambda s: float((s > 0).mean())),
            mean_sp500_excess_return_pct=("sp500_excess_return_pct", "mean"),
            median_sp500_excess_return_pct=("sp500_excess_return_pct", "median"),
            positive_sp500_excess_rate=("sp500_excess_return_pct", lambda s: float((s > 0).mean())),
            mean_sharpe=("sharpe", "mean"),
            positive_sharpe_rate=("sharpe", lambda s: float((s > 0).mean())),
            mean_max_drawdown_pct=("max_drawdown_pct", "mean"),
            mean_signal_coverage=("signal_coverage", "mean"),
            mean_num_trades=("num_trades", "mean"),
        )
        .sort_values(
            by=["mean_excess_return_pct", "mean_test_roc_auc", "mean_sharpe"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return summary


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("test_dataset", as_index=False)
        .agg(
            symbol=("symbol", "first"),
            interval=("interval", "first"),
            models_tested=("model_name", "nunique"),
            mean_test_roc_auc=("test_roc_auc", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_total_return_pct=("total_return_pct", "mean"),
            mean_buy_hold_return_pct=("buy_hold_return_pct", "mean"),
            mean_excess_return_pct=("excess_return_pct", "mean"),
            mean_sp500_buy_hold_return_pct=("sp500_buy_hold_return_pct", "mean"),
            mean_sp500_excess_return_pct=("sp500_excess_return_pct", "mean"),
            mean_sharpe=("sharpe", "mean"),
            mean_max_drawdown_pct=("max_drawdown_pct", "mean"),
        )
        .sort_values("mean_sp500_excess_return_pct", ascending=False)
        .reset_index(drop=True)
    )

    best_idx = df.groupby("test_dataset")["excess_return_pct"].idxmax()
    best_rows = (
        df.loc[best_idx, ["test_dataset", "model_name", "excess_return_pct", "test_roc_auc", "total_return_pct"]]
        .rename(
            columns={
                "model_name": "best_model_by_excess_return",
                "excess_return_pct": "best_excess_return_pct",
                "test_roc_auc": "best_model_test_roc_auc",
                "total_return_pct": "best_model_total_return_pct",
            }
        )
        .reset_index(drop=True)
    )
    best_sp500_idx = df.groupby("test_dataset")["sp500_excess_return_pct"].idxmax()
    best_sp500_rows = (
        df.loc[
            best_sp500_idx,
            ["test_dataset", "model_name", "sp500_excess_return_pct", "total_return_pct"],
        ]
        .rename(
            columns={
                "model_name": "best_model_vs_sp500",
                "sp500_excess_return_pct": "best_sp500_excess_return_pct",
                "total_return_pct": "best_model_vs_sp500_total_return_pct",
            }
        )
        .reset_index(drop=True)
    )
    return grouped.merge(best_rows, on="test_dataset", how="left").merge(
        best_sp500_rows,
        on="test_dataset",
        how="left",
    )


def build_dataset_winners(df: pd.DataFrame, sort_by: str, ascending: bool) -> pd.DataFrame:
    ordered = df.sort_values(
        by=[sort_by, "test_roc_auc", "test_accuracy", "sharpe"],
        ascending=[ascending, False, False, False],
        na_position="last",
    )
    winners = ordered.groupby("test_dataset", as_index=False).first()
    columns = [c for c in DISPLAY_COLUMNS if c in winners.columns]
    return winners[columns]


def build_top_experiments(
    df: pd.DataFrame,
    sort_by: str,
    ascending: bool,
    top_n: int,
) -> pd.DataFrame:
    ordered = df.sort_values(
        by=[sort_by, "test_roc_auc", "test_accuracy", "sharpe"],
        ascending=[ascending, False, False, False],
        na_position="last",
    )
    columns = [c for c in DISPLAY_COLUMNS if c in ordered.columns]
    return ordered[columns].head(top_n).reset_index(drop=True)


def build_metric_rankings(df: pd.DataFrame, top_n: int) -> dict[str, pd.DataFrame]:
    ranking_specs = {
        "top_by_test_roc_auc": "test_roc_auc",
        "top_by_sharpe": "sharpe",
        "top_by_total_return_pct": "total_return_pct",
        "top_by_excess_return_pct": "excess_return_pct",
        "top_by_sp500_excess_return_pct": "sp500_excess_return_pct",
    }
    return {
        name: build_top_experiments(df, sort_by=metric, ascending=False, top_n=top_n)
        for name, metric in ranking_specs.items()
    }


def build_metric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in CORRELATION_METRICS if c in df.columns]
    if not columns:
        return pd.DataFrame()
    return df[columns].corr(numeric_only=True)


def resolve_artifact_path(path_str: str | float | None) -> Path | None:
    if path_str is None or pd.isna(path_str):
        return None
    path = Path(str(path_str))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_candle_diagnostics(candle_path: Path) -> dict[str, object]:
    candle_df = pd.read_csv(candle_path)
    if "timestamp" in candle_df.columns:
        candle_df["timestamp"] = pd.to_datetime(candle_df["timestamp"], errors="coerce")

    position = candle_df["position"] if "position" in candle_df.columns else pd.Series(dtype=float)
    net_ret = candle_df["net_ret"] if "net_ret" in candle_df.columns else pd.Series(dtype=float)

    non_flat = position != 0 if not position.empty else pd.Series(dtype=bool)
    trade_days = int(non_flat.sum()) if not position.empty else 0
    turnover_events = int(position.fillna(0).diff().ne(0).sum() - 1) if not position.empty else 0
    turnover_events = max(turnover_events, 0)

    return {
        "detail_start": candle_df["timestamp"].min() if "timestamp" in candle_df.columns else pd.NaT,
        "detail_end": candle_df["timestamp"].max() if "timestamp" in candle_df.columns else pd.NaT,
        "detail_rows": int(len(candle_df)),
        "detail_final_equity": float(candle_df["equity_curve"].iloc[-1]) if "equity_curve" in candle_df.columns and len(candle_df) else np.nan,
        "detail_long_share": float((position > 0).mean()) if not position.empty else np.nan,
        "detail_short_share": float((position < 0).mean()) if not position.empty else np.nan,
        "detail_flat_share": float((position == 0).mean()) if not position.empty else np.nan,
        "detail_active_share": float(non_flat.mean()) if not position.empty else np.nan,
        "detail_trade_days": trade_days,
        "detail_turnover_events": turnover_events,
        "detail_positive_day_rate": float((net_ret > 0).mean()) if not net_ret.empty else np.nan,
        "detail_mean_net_ret": float(net_ret.mean()) if not net_ret.empty else np.nan,
        "detail_std_net_ret": float(net_ret.std()) if not net_ret.empty else np.nan,
        "detail_best_day_ret": float(net_ret.max()) if not net_ret.empty else np.nan,
        "detail_worst_day_ret": float(net_ret.min()) if not net_ret.empty else np.nan,
    }


def build_detailed_experiment_diagnostics(
    df: pd.DataFrame,
    sort_by: str,
    ascending: bool,
    top_n: int,
) -> pd.DataFrame:
    ranked = df.sort_values(
        by=[sort_by, "test_roc_auc", "test_accuracy", "sharpe"],
        ascending=[ascending, False, False, False],
        na_position="last",
    ).head(top_n)

    rows: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        item = row.to_dict()
        candle_path = resolve_artifact_path(item.get("candle_path"))
        detail = {}
        if candle_path and candle_path.exists():
            detail = build_candle_diagnostics(candle_path)
        item["resolved_candle_path"] = str(candle_path) if candle_path else ""
        item.update(detail)
        rows.append(item)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _format_scalar(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC") if value.tzinfo else value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(
    overview: dict[str, object],
    model_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    dataset_winners: pd.DataFrame,
    top_rankings: dict[str, pd.DataFrame],
    sort_by: str,
    top_n: int,
) -> None:
    print("ML Results Analyzer")
    print("===================")
    print(f"Rows: {_format_scalar(overview['rows'])}")
    print(f"Runs: {_format_scalar(overview['runs'])}")
    print(f"Datasets: {_format_scalar(overview['datasets'])}")
    print(f"Models: {_format_scalar(overview['models'])}")
    print(f"Symbols: {_format_scalar(overview['symbols'])}")
    print(f"Intervals: {_format_scalar(overview['intervals'])}")
    print(f"Created range: {_format_scalar(overview['created_min'])} -> {_format_scalar(overview['created_max'])}")
    print(f"Mean test ROC-AUC: {_format_scalar(overview['mean_test_roc_auc'])}")
    print(f"Mean test accuracy: {_format_scalar(overview['mean_test_accuracy'])}")
    print(f"Mean total return pct: {_format_scalar(overview['mean_total_return_pct'])}")
    print(f"Mean tested-market buy-hold pct: {_format_scalar(overview['mean_buy_hold_return_pct'])}")
    print(f"Mean excess return pct: {_format_scalar(overview['mean_excess_return_pct'])}")
    print(f"Mean S&P 500 buy-hold pct: {_format_scalar(overview['mean_sp500_buy_hold_return_pct'])}")
    print(f"Mean S&P 500 outperformance pct: {_format_scalar(overview['mean_sp500_excess_return_pct'])}")
    print(f"Mean Sharpe: {_format_scalar(overview['mean_sharpe'])}")
    print(f"Mean max drawdown pct: {_format_scalar(overview['mean_max_drawdown_pct'])}")

    print("\nModel leaderboard")
    print("-----------------")
    print(model_summary.round(4).to_string(index=False) if not model_summary.empty else "No rows available.")

    print("\nDataset summary")
    print("---------------")
    print(dataset_summary.round(4).to_string(index=False) if not dataset_summary.empty else "No rows available.")

    print(f"\nBest model per dataset by {sort_by}")
    print("--------------------------------")
    print(dataset_winners.round(4).to_string(index=False) if not dataset_winners.empty else "No rows available.")

    for name, ranking in top_rankings.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(ranking.round(4).to_string(index=False) if not ranking.empty else "No rows available.")

    print(f"\nDetailed run focus uses top {top_n} experiments ranked by {sort_by}.")


def export_tables(
    tables_dir: Path,
    model_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    dataset_winners: pd.DataFrame,
    top_rankings: dict[str, pd.DataFrame],
    correlation: pd.DataFrame,
    detailed_diagnostics: pd.DataFrame,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_summary.to_csv(tables_dir / "model_summary.csv", index=False)
    dataset_summary.to_csv(tables_dir / "dataset_summary.csv", index=False)
    dataset_winners.to_csv(tables_dir / "dataset_winners.csv", index=False)
    detailed_diagnostics.to_csv(tables_dir / "detailed_experiment_diagnostics.csv", index=False)
    if not correlation.empty:
        correlation.to_csv(tables_dir / "metric_correlation.csv")
    for name, ranking in top_rankings.items():
        ranking.to_csv(tables_dir / f"{name}.csv", index=False)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def clear_old_plots(plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_path in plots_dir.glob("*.png"):
        if plot_path.is_file():
            plot_path.unlink()


def plot_model_boxplots(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    chart_specs = [
        ("total_return_pct", "Strategy Return % by Model", "Strategy return %", "#8db7d9"),
        ("excess_return_pct", "Excess Return vs Tested Market % by Model", "Excess return %", "#7fc8a9"),
        ("sp500_excess_return_pct", "Outperformance vs S&P 500 % by Model", "S&P 500 outperformance %", "#f49f85"),
        ("buy_hold_return_pct", "Tested Market Buy-Hold Return % by Model", "Buy-hold return %", "#b8bec9"),
    ]
    for ax, (metric, title, y_label, color) in zip(axes, chart_specs):
        _styled_model_boxplot(ax, df, metric, title, y_label, color)
    fig.suptitle("")
    path = plots_dir / "model_boxplots.png"
    _save_figure(fig, path)
    return path


def _styled_model_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    color: str,
) -> None:
    plot_df = df[["model_name", metric]].dropna()
    if plot_df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xlabel("Model")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.2)
        return

    plot_df.boxplot(
        column=metric,
        by="model_name",
        ax=ax,
        grid=False,
        patch_artist=True,
        boxprops={"facecolor": color, "edgecolor": "#274060", "linewidth": 2.2, "alpha": 0.75},
        whiskerprops={"color": "#274060", "linewidth": 2.0},
        capprops={"color": "#274060", "linewidth": 2.0},
        medianprops={"color": "#8b1e3f", "linewidth": 2.8},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#8b1e3f",
            "markeredgecolor": "#8b1e3f",
            "markersize": 4.5,
            "alpha": 0.45,
        },
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)


def plot_model_score_boxplots(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    chart_specs = [
        ("test_roc_auc", "Test ROC-AUC by Model", "ROC-AUC", "#9ecae1"),
        ("test_accuracy", "Test Accuracy by Model", "Accuracy", "#c7d36f"),
        ("excess_return_pct", "Excess Return vs Tested Market % by Model", "Excess return %", "#7fc8a9"),
        ("sharpe", "Sharpe by Model", "Sharpe", "#f2b880"),
    ]
    for ax, (metric, title, y_label, color) in zip(axes, chart_specs):
        _styled_model_boxplot(ax, df, metric, title, y_label, color)
    fig.suptitle("")
    path = plots_dir / "model_score_boxplots.png"
    _save_figure(fig, path)
    return path


def plot_model_mean_bars(model_summary: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    chart_specs = [
        ("mean_total_return_pct", "Mean Strategy Return %", "steelblue"),
        ("mean_excess_return_pct", "Mean Excess Return vs Tested Market %", "seagreen"),
        ("mean_sp500_excess_return_pct", "Mean Outperformance vs S&P 500 %", "coral"),
        ("mean_sharpe", "Mean Sharpe", "goldenrod"),
    ]
    for ax, (col, title, color) in zip(axes.flatten(), chart_specs):
        ax.bar(model_summary["model_name"], model_summary[col], color=color, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    path = plots_dir / "model_mean_metrics.png"
    _save_figure(fig, path)
    return path


def plot_dataset_bars(dataset_summary: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    chart_specs = [
        ("mean_total_return_pct", "Average Strategy Return % by Dataset", "steelblue"),
        ("mean_buy_hold_return_pct", "Average Tested Market Buy-Hold % by Dataset", "slategray"),
        ("mean_excess_return_pct", "Average Excess Return vs Tested Market % by Dataset", "seagreen"),
        ("mean_sp500_excess_return_pct", "Average Outperformance vs S&P 500 % by Dataset", "coral"),
    ]
    for ax, (col, title, color) in zip(axes.flatten(), chart_specs):
        ordered = dataset_summary.sort_values(col, ascending=True)
        ax.barh(ordered["test_dataset"], ordered[col], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.grid(axis="x", alpha=0.25)
    path = plots_dir / "dataset_metric_bars.png"
    _save_figure(fig, path)
    return path


def plot_return_vs_benchmarks_by_dataset(dataset_summary: pd.DataFrame, plots_dir: Path) -> Path:
    ordered = dataset_summary.sort_values("best_sp500_excess_return_pct", ascending=False).reset_index(drop=True)
    labels = ordered["test_dataset"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.7), 7))
    ax.bar(x - width, ordered["mean_total_return_pct"], width=width, label="Strategy return %", color="steelblue")
    ax.bar(x, ordered["mean_buy_hold_return_pct"], width=width, label="Tested market buy-hold %", color="darkgray")
    ax.bar(x + width, ordered["mean_sp500_buy_hold_return_pct"], width=width, label="S&P 500 buy-hold %", color="darkorange")
    ax.set_title("Strategy Return vs Tested Market and S&P 500 by Dataset")
    ax.set_ylabel("Return %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = plots_dir / "return_vs_benchmarks_by_dataset.png"
    _save_figure(fig, path)
    return path


def plot_dataset_outperformance_bars(dataset_summary: pd.DataFrame, plots_dir: Path) -> Path:
    ordered = dataset_summary.sort_values("best_sp500_excess_return_pct", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    specs = [
        ("best_excess_return_pct", "Best Excess Return vs Tested Market by Dataset", "seagreen"),
        ("best_sp500_excess_return_pct", "Best Outperformance vs S&P 500 by Dataset", "coral"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        ax.barh(ordered["test_dataset"], ordered[col], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.grid(axis="x", alpha=0.25)
    path = plots_dir / "dataset_outperformance_bars.png"
    _save_figure(fig, path)
    return path


def plot_return_distributions(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    chart_specs = [
        ("total_return_pct", "Distribution of Strategy Returns %", "steelblue"),
        ("buy_hold_return_pct", "Distribution of Tested Market Buy-Hold %", "gray"),
        ("excess_return_pct", "Distribution of Excess Return vs Tested Market %", "seagreen"),
        ("sp500_excess_return_pct", "Distribution of Outperformance vs S&P 500 %", "coral"),
    ]
    for ax, (col, title, color) in zip(axes.flatten(), chart_specs):
        series = df[col].dropna()
        bins = min(20, max(8, int(math.sqrt(len(series))))) if len(series) else 10
        ax.hist(series, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.grid(axis="y", alpha=0.25)
    path = plots_dir / "return_distributions.png"
    _save_figure(fig, path)
    return path


def _annotated_heatmap(
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    center: float | None,
    path: Path,
    value_format: str,
) -> Path:
    fig_width = max(8, 2 + 1.3 * len(matrix.columns))
    fig_height = max(6, 2 + 0.45 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    values = matrix.to_numpy(dtype=float)
    if center is None:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
    else:
        spread = np.nanmax(np.abs(values - center))
        if not np.isfinite(spread) or spread == 0:
            spread = 1.0
        vmin = center - spread
        vmax = center + spread

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    norm_threshold = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = "n/a"
                color = "black"
            else:
                label = format(value, value_format)
                color = "white" if value < norm_threshold else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)
    return path


def plot_heatmaps(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    paths: list[Path] = []
    roc_pivot = df.pivot_table(values="test_roc_auc", index="test_dataset", columns="model_name", aggfunc="mean")
    total_return_pivot = df.pivot_table(values="total_return_pct", index="test_dataset", columns="model_name", aggfunc="mean")
    excess_pivot = df.pivot_table(values="excess_return_pct", index="test_dataset", columns="model_name", aggfunc="mean")
    sp500_excess_pivot = df.pivot_table(values="sp500_excess_return_pct", index="test_dataset", columns="model_name", aggfunc="mean")
    if not roc_pivot.empty:
        paths.append(
            _annotated_heatmap(
                roc_pivot,
                title="ROC-AUC Heatmap: Models vs Datasets",
                cmap="RdYlGn",
                center=0.5,
                path=plots_dir / "heatmap_test_roc_auc.png",
                value_format=".3f",
            )
        )
    if not total_return_pivot.empty:
        paths.append(
            _annotated_heatmap(
                total_return_pivot,
                title="Strategy Return Heatmap: Models vs Datasets",
                cmap="RdYlGn",
                center=0.0,
                path=plots_dir / "heatmap_total_return_pct.png",
                value_format=".1f",
            )
        )
    if not excess_pivot.empty:
        paths.append(
            _annotated_heatmap(
                excess_pivot,
                title="Excess Return Heatmap: Models vs Datasets",
                cmap="RdYlGn",
                center=0.0,
                path=plots_dir / "heatmap_excess_return_pct.png",
                value_format=".1f",
            )
        )
    if not sp500_excess_pivot.empty:
        paths.append(
            _annotated_heatmap(
                sp500_excess_pivot,
                title="S&P 500 Outperformance Heatmap: Models vs Datasets",
                cmap="RdYlGn",
                center=0.0,
                path=plots_dir / "heatmap_sp500_excess_return_pct.png",
                value_format=".1f",
            )
        )
    return paths


def plot_correlation_heatmap(correlation: pd.DataFrame, plots_dir: Path) -> Path | None:
    if correlation.empty:
        return None
    return _annotated_heatmap(
        correlation,
        title="Correlation Matrix: ML Metrics",
        cmap="coolwarm",
        center=0.0,
        path=plots_dir / "correlation_heatmap.png",
        value_format=".2f",
    )


def _scatter_by_model_with_regression(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    *,
    legend: bool = False,
) -> None:
    plot_df = df[[c for c in ["model_name", x_col, y_col] if c in df.columns]].dropna()
    if plot_df.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        return

    models = list(dict.fromkeys(plot_df["model_name"].dropna().tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        subset = plot_df[plot_df["model_name"] == model]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=55,
            alpha=0.8,
            label=model,
            color=cmap(idx % 10),
        )

    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(plot_df) >= 2 else np.nan

    if len(plot_df) >= 2 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.5)

    ax.axhline(0.0, linestyle=":", linewidth=1, color="gray")
    ax.axvline(0.0, linestyle=":", linewidth=1, color="gray")
    ax.set_title(f"{title}\nPearson r = {_format_scalar(correlation)}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(fontsize=8)


def plot_roc_vs_excess_scatter(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    models = list(dict.fromkeys(df["model_name"].dropna().tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        subset = df[df["model_name"] == model]
        ax.scatter(
            subset["test_roc_auc"],
            subset["excess_return_pct"],
            s=50,
            alpha=0.8,
            label=model,
            color=cmap(idx % 10),
        )
    ax.axvline(0.5, linestyle="--", linewidth=1, color="gray")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.set_title("Predictive Quality vs Excess Return")
    ax.set_xlabel("Test ROC-AUC")
    ax.set_ylabel("Excess Return %")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plots_dir / "roc_vs_excess_scatter.png"
    _save_figure(fig, path)
    return path


def plot_return_vs_market_scatter(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    models = list(dict.fromkeys(df["model_name"].dropna().tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        subset = df[df["model_name"] == model]
        ax.scatter(
            subset["buy_hold_return_pct"],
            subset["total_return_pct"],
            s=55,
            alpha=0.8,
            label=model,
            color=cmap(idx % 10),
        )
    bounds = [
        np.nanmin([df["buy_hold_return_pct"].min(), df["total_return_pct"].min()]),
        np.nanmax([df["buy_hold_return_pct"].max(), df["total_return_pct"].max()]),
    ]
    ax.plot(bounds, bounds, linestyle="--", linewidth=1, color="gray")
    ax.axhline(0.0, linestyle=":", linewidth=1, color="black")
    ax.axvline(0.0, linestyle=":", linewidth=1, color="black")
    ax.set_title("Strategy Return vs Tested Market Buy-Hold")
    ax.set_xlabel("Tested market buy-hold return %")
    ax.set_ylabel("Strategy return %")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plots_dir / "return_vs_tested_market_scatter.png"
    _save_figure(fig, path)
    return path


def plot_excess_vs_sp500_scatter(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    models = list(dict.fromkeys(df["model_name"].dropna().tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        subset = df[df["model_name"] == model]
        ax.scatter(
            subset["excess_return_pct"],
            subset["sp500_excess_return_pct"],
            s=55,
            alpha=0.8,
            label=model,
            color=cmap(idx % 10),
        )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.set_title("Outperformance vs Tested Market and S&P 500")
    ax.set_xlabel("Excess return vs tested market %")
    ax.set_ylabel("Outperformance vs S&P 500 %")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plots_dir / "excess_vs_sp500_scatter.png"
    _save_figure(fig, path)
    return path


def plot_risk_relationships(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    _scatter_by_model_with_regression(
        axes[0],
        df,
        x_col="sharpe",
        y_col="excess_return_pct",
        title="Sharpe vs Excess Return",
        x_label="Sharpe",
        y_label="Excess return vs tested market %",
        legend=True,
    )
    _scatter_by_model_with_regression(
        axes[1],
        df,
        x_col="max_drawdown_pct",
        y_col="excess_return_pct",
        title="Max Drawdown vs Excess Return",
        x_label="Max drawdown %",
        y_label="Excess return vs tested market %",
    )
    path = plots_dir / "risk_relationships.png"
    _save_figure(fig, path)
    return path


def plot_trade_return_regressions(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    _scatter_by_model_with_regression(
        axes[0],
        df,
        x_col="num_trades",
        y_col="total_return_pct",
        title="Trades vs Strategy Return",
        x_label="Number of trades",
        y_label="Strategy return %",
        legend=True,
    )
    _scatter_by_model_with_regression(
        axes[1],
        df,
        x_col="num_trades",
        y_col="excess_return_pct",
        title="Trades vs Excess Return",
        x_label="Number of trades",
        y_label="Excess return vs tested market %",
    )
    path = plots_dir / "trade_return_regressions.png"
    _save_figure(fig, path)
    return path


def plot_best_outperformance_strategies(df: pd.DataFrame, plots_dir: Path, top_n: int = 10) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, top_n * 0.6)))
    chart_specs = [
        (
            "excess_return_pct",
            "Top Strategies by Tested-Market Outperformance",
            "Excess return vs tested market %",
            "seagreen",
        ),
        (
            "sp500_excess_return_pct",
            "Top Strategies by S&P 500 Outperformance",
            "Outperformance vs S&P 500 %",
            "coral",
        ),
    ]

    for ax, (metric, title, x_label, color) in zip(axes, chart_specs):
        cols = [c for c in ["test_dataset", "model_name", "symbol", metric] if c in df.columns]
        ranked = (
            df[cols]
            .dropna(subset=[metric])
            .sort_values(metric, ascending=False)
            .head(top_n)
            .copy()
        )

        if ranked.empty:
            ax.set_title(f"{title}\n(no data)")
            ax.set_xlabel(x_label)
            ax.grid(axis="x", alpha=0.25)
            continue

        ranked = ranked.iloc[::-1].reset_index(drop=True)
        labels = [
            f"{row['model_name']} | {row['test_dataset']}"
            for _, row in ranked.iterrows()
        ]
        ax.barh(labels, ranked[metric], color=color, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)

        values = ranked[metric].tolist()
        offset = max(0.5, max(abs(v) for v in values) * 0.015)
        for idx, (_, row) in enumerate(ranked.iterrows()):
            value = float(row[metric])
            text_x = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(text_x, idx, f"{value:.1f}%", va="center", ha=ha, fontsize=9)

    path = plots_dir / "best_outperformance_strategies.png"
    _save_figure(fig, path)
    return path


def _load_buy_hold_curve(candle_path: Path) -> pd.DataFrame:
    candle_df = pd.read_csv(candle_path)
    if "timestamp" in candle_df.columns:
        candle_df["timestamp"] = pd.to_datetime(candle_df["timestamp"], errors="coerce")
    if "next_return" not in candle_df.columns:
        raise ValueError(f"Missing next_return in candle file: {candle_path}")
    curve = pd.DataFrame(
        {
            "timestamp": candle_df["timestamp"],
            "buy_hold_curve": (1.0 + candle_df["next_return"].astype(float)).cumprod(),
        }
    ).dropna(subset=["timestamp"])
    return curve


def plot_single_best_strategy_vs_benchmarks(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    required = {"candle_path", "sp500_excess_return_pct", "excess_return_pct", "sharpe", "total_return_pct"}
    if not required.issubset(df.columns):
        return None

    ranked = (
        df.dropna(subset=["candle_path", "sp500_excess_return_pct"])
        .sort_values(
            by=["sp500_excess_return_pct", "excess_return_pct", "sharpe", "total_return_pct"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    if ranked.empty:
        return None

    best_row = ranked.iloc[0]
    best_candle_path = resolve_artifact_path(best_row.get("candle_path"))
    if best_candle_path is None or not best_candle_path.exists():
        return None

    best_candle_df = pd.read_csv(best_candle_path)
    if "timestamp" not in best_candle_df.columns or "equity_curve" not in best_candle_df.columns:
        return None
    best_candle_df["timestamp"] = pd.to_datetime(best_candle_df["timestamp"], errors="coerce")
    best_candle_df = best_candle_df.dropna(subset=["timestamp"]).copy()
    if best_candle_df.empty:
        return None

    tested_market_curve = _load_buy_hold_curve(best_candle_path)

    fig, ax = plt.subplots(figsize=(12, 7))
    strategy_label = f"Strategy: {best_row['model_name']} | {best_row['test_dataset']}"
    ax.plot(
        best_candle_df["timestamp"],
        best_candle_df["equity_curve"].astype(float),
        label=strategy_label,
        linewidth=2.4,
        color="steelblue",
    )
    ax.plot(
        tested_market_curve["timestamp"],
        tested_market_curve["buy_hold_curve"],
        label=f"Tested market buy-hold: {best_row['symbol']}",
        linewidth=2.0,
        color="dimgray",
    )

    sp500_rows = df[
        (df["run_id"] == best_row["run_id"])
        & (df["interval"] == best_row["interval"])
        & (df["symbol"] == "^GSPC")
    ].dropna(subset=["candle_path"])
    if not sp500_rows.empty:
        sp500_candle_path = resolve_artifact_path(sp500_rows.iloc[0]["candle_path"])
        if sp500_candle_path is not None and sp500_candle_path.exists():
            sp500_curve = _load_buy_hold_curve(sp500_candle_path)
            if not sp500_curve.empty:
                sp500_curve = (
                    sp500_curve.set_index("timestamp")
                    .reindex(best_candle_df["timestamp"])
                    .ffill()
                    .dropna()
                    .reset_index()
                )
                if not sp500_curve.empty:
                    ax.plot(
                        sp500_curve["timestamp"],
                        sp500_curve["buy_hold_curve"],
                        label="S&P 500 buy-hold",
                        linewidth=2.0,
                        color="darkorange",
                    )

    ax.set_title(
        "Single Best Strategy vs Tested Market and S&P 500\n"
        f"Best by S&P 500 outperformance: {best_row['sp500_excess_return_pct']:.1f}%"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized equity")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plots_dir / "single_best_strategy_vs_benchmarks.png"
    _save_figure(fig, path)
    return path


def plot_top_equity_curves(detailed_diagnostics: pd.DataFrame, plots_dir: Path) -> Path | None:
    if detailed_diagnostics.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = 0
    for _, row in detailed_diagnostics.iterrows():
        candle_path = resolve_artifact_path(row.get("candle_path"))
        if candle_path is None or not candle_path.exists():
            continue
        candle_df = pd.read_csv(candle_path)
        if "timestamp" in candle_df.columns:
            x = pd.to_datetime(candle_df["timestamp"], errors="coerce")
        else:
            x = np.arange(len(candle_df))
        if "equity_curve" not in candle_df.columns:
            continue
        label = f"{row['test_dataset']} | {row['model_name']}"
        ax.plot(x, candle_df["equity_curve"], label=label, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title("Equity Curves for Top Ranked Experiments")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity Curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = plots_dir / "top_equity_curves.png"
    _save_figure(fig, path)
    return path


def generate_plots(
    df: pd.DataFrame,
    model_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    detailed_diagnostics: pd.DataFrame,
    plots_dir: Path,
) -> list[Path]:
    clear_old_plots(plots_dir)
    paths = [
        plot_model_boxplots(df, plots_dir),
        plot_model_score_boxplots(df, plots_dir),
        plot_model_mean_bars(model_summary, plots_dir),
        plot_dataset_bars(dataset_summary, plots_dir),
        plot_return_vs_benchmarks_by_dataset(dataset_summary, plots_dir),
        plot_dataset_outperformance_bars(dataset_summary, plots_dir),
        plot_return_distributions(df, plots_dir),
        plot_roc_vs_excess_scatter(df, plots_dir),
        plot_return_vs_market_scatter(df, plots_dir),
        plot_excess_vs_sp500_scatter(df, plots_dir),
        plot_risk_relationships(df, plots_dir),
        plot_trade_return_regressions(df, plots_dir),
        plot_best_outperformance_strategies(df, plots_dir),
    ]
    paths.extend(plot_heatmaps(df, plots_dir))
    corr_path = plot_correlation_heatmap(correlation, plots_dir)
    if corr_path is not None:
        paths.append(corr_path)
    best_strategy_path = plot_single_best_strategy_vs_benchmarks(df, plots_dir)
    if best_strategy_path is not None:
        paths.append(best_strategy_path)
    equity_path = plot_top_equity_curves(detailed_diagnostics, plots_dir)
    if equity_path is not None:
        paths.append(equity_path)
    return paths


def _markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    sample = df.head(max_rows).copy()
    for col in sample.select_dtypes(include=["float", "float64"]).columns:
        sample[col] = sample[col].map(lambda x: round(x, 4) if pd.notna(x) else x)
    headers = [str(col) for col in sample.columns]
    rows: list[list[str]] = []
    for _, row in sample.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, pd.Timestamp):
                values.append(_format_scalar(value))
            else:
                values.append(str(value))
        rows.append(values)

    align_row = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align_row) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def write_markdown_report(
    report_path: Path,
    overview: dict[str, object],
    sort_by: str,
    model_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    dataset_winners: pd.DataFrame,
    top_rankings: dict[str, pd.DataFrame],
    detailed_diagnostics: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    lines = [
        "# ML Results Analysis Report",
        "",
        "## Overview",
        "",
        f"- Rows: {overview['rows']}",
        f"- Runs: {overview['runs']}",
        f"- Datasets: {overview['datasets']}",
        f"- Models: {overview['models']}",
        f"- Symbols: {overview['symbols']}",
        f"- Intervals: {overview['intervals']}",
        f"- Created range: {_format_scalar(overview['created_min'])} -> {_format_scalar(overview['created_max'])}",
        f"- Mean test ROC-AUC: {_format_scalar(overview['mean_test_roc_auc'])}",
        f"- Mean test accuracy: {_format_scalar(overview['mean_test_accuracy'])}",
        f"- Mean total return pct: {_format_scalar(overview['mean_total_return_pct'])}",
        f"- Mean buy-hold return pct: {_format_scalar(overview['mean_buy_hold_return_pct'])}",
        f"- Mean excess return pct: {_format_scalar(overview['mean_excess_return_pct'])}",
        f"- Mean S&P 500 buy-hold pct: {_format_scalar(overview['mean_sp500_buy_hold_return_pct'])}",
        f"- Mean S&P 500 outperformance pct: {_format_scalar(overview['mean_sp500_excess_return_pct'])}",
        f"- Mean Sharpe: {_format_scalar(overview['mean_sharpe'])}",
        f"- Mean max drawdown pct: {_format_scalar(overview['mean_max_drawdown_pct'])}",
        "",
        "## Model Leaderboard",
        "",
        _markdown_table(model_summary, max_rows=20),
        "",
        "## Dataset Summary",
        "",
        _markdown_table(dataset_summary, max_rows=25),
        "",
        f"## Best Model Per Dataset by `{sort_by}`",
        "",
        _markdown_table(dataset_winners, max_rows=25),
        "",
    ]

    for name, ranking in top_rankings.items():
        lines.extend(
            [
                f"## {name}",
                "",
                _markdown_table(ranking, max_rows=20),
                "",
            ]
        )

    lines.extend(
        [
            "## Detailed Diagnostics for Top Ranked Experiments",
            "",
            _markdown_table(
                detailed_diagnostics[
                    [
                        c
                        for c in [
                            "test_dataset",
                            "model_name",
                            "test_roc_auc",
                            "total_return_pct",
                            "excess_return_pct",
                            "sp500_excess_return_pct",
                            "sharpe",
                            "detail_final_equity",
                            "detail_active_share",
                            "detail_turnover_events",
                            "detail_best_day_ret",
                            "detail_worst_day_ret",
                            "resolved_candle_path",
                        ]
                        if c in detailed_diagnostics.columns
                    ]
                ],
                max_rows=20,
            ),
            "",
            "## Generated Plots",
            "",
        ]
    )
    for path in plot_paths:
        rel = path.relative_to(report_path.parent)
        lines.append(f"- `{rel}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ML experiment results and export a full report.")
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to database/ml_results.csv or database/ml_results.parquet",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Filter to a single run_id.")
    parser.add_argument("--model", nargs="*", default=None, help="Filter to one or more model names.")
    parser.add_argument("--symbol", nargs="*", default=None, help="Filter to one or more symbols.")
    parser.add_argument("--interval", nargs="*", default=None, help="Filter to one or more intervals.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="excess_return_pct",
        help="Metric used to rank experiments and dataset winners.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top experiments to include in rankings.",
    )
    parser.add_argument(
        "--detail-top",
        type=int,
        default=6,
        help="Number of top experiments to inspect at candle level.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing tables, plots, and the markdown report.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory used for exported tables, plots, and the markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = resolve_results_path(args.results_path)
    export_dir = Path(args.export_dir)

    df = load_results(results_path)
    if args.sort_by not in df.columns:
        raise ValueError(f"Unknown sort column '{args.sort_by}'. Available columns: {df.columns.tolist()}")

    filtered = apply_filters(
        df,
        run_id=args.run_id,
        model_names=args.model,
        symbols=args.symbol,
        intervals=args.interval,
    )
    if filtered.empty:
        raise ValueError("No ML results matched the selected filters.")

    overview = build_overview(filtered)
    model_summary = build_model_summary(filtered)
    dataset_summary = build_dataset_summary(filtered)
    dataset_winners = build_dataset_winners(filtered, sort_by=args.sort_by, ascending=args.ascending)
    top_rankings = build_metric_rankings(filtered, top_n=args.top)
    correlation = build_metric_correlation(filtered)
    detailed_diagnostics = build_detailed_experiment_diagnostics(
        filtered,
        sort_by=args.sort_by,
        ascending=args.ascending,
        top_n=args.detail_top,
    )

    print_report(
        overview=overview,
        model_summary=model_summary,
        dataset_summary=dataset_summary,
        dataset_winners=dataset_winners,
        top_rankings=top_rankings,
        sort_by=args.sort_by,
        top_n=args.detail_top,
    )

    if not args.no_export:
        tables_dir = export_dir / "tables"
        plots_dir = export_dir / "plots"
        export_tables(
            tables_dir=tables_dir,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            dataset_winners=dataset_winners,
            top_rankings=top_rankings,
            correlation=correlation,
            detailed_diagnostics=detailed_diagnostics,
        )
        plot_paths = generate_plots(
            df=filtered,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            correlation=correlation,
            detailed_diagnostics=detailed_diagnostics,
            plots_dir=plots_dir,
        )
        write_markdown_report(
            report_path=export_dir / "report.md",
            overview=overview,
            sort_by=args.sort_by,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            dataset_winners=dataset_winners,
            top_rankings=top_rankings,
            detailed_diagnostics=detailed_diagnostics,
            plot_paths=plot_paths,
        )
        print(f"\nExported analysis artifacts to: {export_dir}")


if __name__ == "__main__":
    main()

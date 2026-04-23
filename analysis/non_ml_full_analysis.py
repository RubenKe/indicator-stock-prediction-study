import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_all import STARTING_CASH, build_interval_to_timeframe, load_config, run_single_backtest


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "database" / "results.parquet"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "analysis" / "results" / "non_ml" / "graphs"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "analysis" / "results" / "non_ml" / "csv"
DEFAULT_PREFIX = "non_ml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _load_price_df(
    symbol: str, interval: str, cache: dict[tuple[str, str], pd.DataFrame | None]
) -> pd.DataFrame | None:
    key = (symbol, interval)
    if key in cache:
        return cache[key]
    parquet_path = DATA_RAW_DIR / f"{symbol}_{interval}.parquet"
    csv_path = DATA_RAW_DIR / f"{symbol}_{interval}.csv"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        cache[key] = None
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[frame.index.notna()]
    frame = frame.sort_index()
    cache[key] = frame
    return frame


def _run_single_strategy_from_raw(
    strategy_name: str,
    symbol: str,
    interval: str,
    param_dict: dict,
    commission: float,
    slippage: float,
    sizer: float,
    risk_config: dict | None,
    cache: dict[tuple[str, str], pd.DataFrame | None],
) -> pd.DataFrame:
    config = load_config()
    if risk_config is None:
        risk_config = dict(config.get("risk", {}))
    else:
        risk_config = dict(risk_config)
    risk_config["sizer_pct"] = sizer
    risk_config["max_position_value_pct"] = sizer / 100.0

    interval_to_timeframe = build_interval_to_timeframe(config["INTERVAL_TO_TIMEFRAME"])
    price_df = _load_price_df(symbol, interval, cache)
    if price_df is None:
        raise FileNotFoundError(f"Missing data for {symbol} {interval} in data/raw")

    result = run_single_backtest(
        strategy_name=strategy_name,
        param_dict=param_dict,
        price_df=price_df,
        commission=commission,
        slippage=slippage,
        sizer=sizer,
        interval=interval,
        interval_to_timeframe=interval_to_timeframe,
        risk_config=risk_config,
    )
    returns = pd.Series(result.analyzers.timereturn.get_analysis())
    returns.index = pd.to_datetime(returns.index)
    equity = (1.0 + returns).cumprod() * STARTING_CASH
    return pd.DataFrame({"equity": equity, "returns": returns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extensive non-ML backtest analysis plots and tables."
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--backtest-version", type=str, default="")
    parser.add_argument("--risk-profile", type=str, default="")
    parser.add_argument("--top-strategies", type=int, default=12)
    parser.add_argument("--top-runs", type=int, default=100)
    return parser.parse_args()


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "return",
        "sharpe",
        "trades",
        "win_rate",
        "market_gain",
        "excess_return",
        "benchmark_gain",
        "benchmark_excess_return",
        "annualized_return",
        "max_drawdown_pct",
        "calmar",
        "profit_factor",
    ]
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _scope(df: pd.DataFrame, backtest_version: str, risk_profile: str) -> pd.DataFrame:
    out = df.copy()
    if backtest_version:
        out = out[out["backtest_version"] == backtest_version]
    elif "backtest_version" in out.columns and not out.empty:
        out = out[out["backtest_version"] == out["backtest_version"].mode().iloc[0]]

    if risk_profile:
        out = out[out["risk_profile"] == risk_profile]
    return out


def _score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_rank"] = out["return"].rank(pct=True, ascending=True)
    out["sharpe_rank"] = out["sharpe"].rank(pct=True, ascending=True)
    out["benchmark_excess_rank"] = out["benchmark_excess_return"].rank(
        pct=True, ascending=True
    )
    out["ann_benchmark_excess_rank"] = out["annualized_benchmark_excess"].rank(
        pct=True, ascending=True
    )
    out["drawdown_rank"] = out["max_drawdown_pct"].rank(pct=True, ascending=False)
    out["duration_rank"] = out["duration_days"].rank(pct=True, ascending=True)
    out["composite_score"] = out[
        [
            "return_rank",
            "sharpe_rank",
            "benchmark_excess_rank",
            "ann_benchmark_excess_rank",
            "drawdown_rank",
            "duration_rank",
        ]
    ].mean(axis=1)
    return out


def _save(fig: plt.Figure, plots_dir: Path, prefix: str, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _clean_old_plots(plots_dir: Path, prefix: str) -> int:
    plots_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in plots_dir.glob(f"{prefix}_*.png"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _robust_bounds(series: pd.Series, low_q: float = 0.01, high_q: float = 0.99) -> tuple[float, float] | None:
    vals = series.dropna()
    if vals.empty:
        return None
    lo = float(vals.quantile(low_q))
    hi = float(vals.quantile(high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return lo, hi


def _group_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (
        df.groupby(col, dropna=False)
        .agg(
            runs=("strategy", "size"),
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            mean_drawdown_pct=("max_drawdown_pct", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_trades=("trades", "mean"),
            mean_benchmark_excess=("benchmark_excess_return", "mean"),
            median_benchmark_excess=("benchmark_excess_return", "median"),
            mean_ann_benchmark_excess=("annualized_benchmark_excess", "mean"),
            median_ann_benchmark_excess=("annualized_benchmark_excess", "median"),
            mean_duration_days=("duration_days", "mean"),
        )
        .reset_index()
    )
    g["pct_positive_return"] = (
        df.groupby(col, dropna=False)["return"].apply(lambda s: (s > 0).mean()).values
    )
    g["pct_beat_benchmark"] = (
        df.groupby(col, dropna=False)["benchmark_excess_return"]
        .apply(lambda s: (s > 0).mean())
        .values
    )
    # Duration-weighted annualized benchmark excess to reduce short-window bias.
    weighted_ann = (
        df.assign(_w=df["duration_days"].fillna(0).clip(lower=0))
        .groupby(col, dropna=False)[["annualized_benchmark_excess", "_w"]]
        .apply(
            lambda x: np.average(
                x["annualized_benchmark_excess"].fillna(0.0),
                weights=np.where(x["_w"].to_numpy(dtype=float) > 0, x["_w"], 1.0),
            )
            if len(x) > 0
            else np.nan
        )
    )
    g["duration_weighted_ann_benchmark_excess"] = g[col].map(weighted_ann)
    return g


def _compute_duration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "start_date" in out.columns:
        out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    if "end_date" in out.columns:
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")

    if "start_date" in out.columns and "end_date" in out.columns:
        out["duration_days"] = (out["end_date"] - out["start_date"]).dt.days.clip(lower=1)
    else:
        out["duration_days"] = np.nan
    out["duration_years"] = out["duration_days"] / 365.25

    # Annualize benchmark gain to compare runs with different windows.
    gross_benchmark = 1.0 + (out["benchmark_gain"] / 100.0)
    valid_bench = gross_benchmark > 0
    out["annualized_benchmark_gain"] = np.nan
    out.loc[valid_bench, "annualized_benchmark_gain"] = (
        np.power(gross_benchmark[valid_bench], 365.25 / out.loc[valid_bench, "duration_days"]) - 1.0
    ) * 100.0

    out["annualized_benchmark_excess"] = (
        out["annualized_return"] - out["annualized_benchmark_gain"]
    )
    return out


def _annotate_hist_stats(ax: plt.Axes, values: pd.Series, unit: str = "") -> None:
    vals = values.dropna()
    if vals.empty:
        return
    mean_v = float(vals.mean())
    med_v = float(vals.median())
    std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    ax.axvline(mean_v, color="darkorange", linestyle="-", linewidth=1.3, label="mean")
    ax.axvline(med_v, color="purple", linestyle="--", linewidth=1.3, label="median")
    txt = (
        f"n={len(vals)}\n"
        f"mean={mean_v:.2f}{unit}\n"
        f"median={med_v:.2f}{unit}\n"
        f"std={std_v:.2f}{unit}"
    )
    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="upper left")


def _add_scatter_regression(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return
    x_vals = pair["x"].to_numpy(dtype=float)
    y_vals = pair["y"].to_numpy(dtype=float)
    if float(np.std(x_vals)) == 0.0 or float(np.std(y_vals)) == 0.0:
        return
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 120)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="crimson", linewidth=1.8, label="regression")

    r = float(np.corrcoef(x_vals, y_vals)[0, 1])
    if not np.isfinite(r):
        return
    r2 = r * r
    txt = (
        f"n={len(pair)}\n"
        f"slope={slope:.4f}\n"
        f"intercept={intercept:.2f}\n"
        f"corr={r:.3f}\n"
        f"R²={r2:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="best")


def _strategy_color_map(strategies: list[str]) -> dict[str, tuple]:
    uniq = list(dict.fromkeys([str(s) for s in strategies if pd.notna(s)]))
    n = len(uniq)
    if n == 0:
        return {}

    # Use a qualitative palette when possible; fall back to a discretized HSV wheel
    # for larger numbers of strategies.
    if n <= 20:
        cmap = plt.get_cmap("tab20", n)
    else:
        cmap = plt.get_cmap("hsv", n)

    return {s: cmap(i) for i, s in enumerate(uniq)}


def _annotate_bar_values(ax: plt.Axes, max_labels: int = 16, as_pct: bool = False) -> None:
    bars = ax.patches
    if len(bars) == 0:
        return
    idxs = range(min(len(bars), max_labels))
    for i in idxs:
        b = bars[i]
        h = b.get_height()
        if np.isnan(h):
            continue
        label = f"{h:.1%}" if as_pct else f"{h:.2f}"
        y = h + (0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def generate_analysis(
    results_path: Path,
    plots_dir: Path,
    tables_dir: Path,
    prefix: str,
    backtest_version: str,
    risk_profile: str,
    top_strategies: int,
    top_runs: int,
) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing file: {results_path}")

    df = pd.read_parquet(results_path)
    df = _scope(df, backtest_version=backtest_version, risk_profile=risk_profile)
    df = _to_numeric(df)
    df = _compute_duration_metrics(df)
    df = df.dropna(subset=["strategy", "symbol", "interval"])
    df = _score(df)

    if df.empty:
        raise RuntimeError("No rows available after filtering.")

    # Exclude zero-trade runs from analysis outputs to avoid distorted "performance"
    # from strategies/parameter sets that never actually entered the market.
    traded_df = df[df["trades"].fillna(0) > 0].copy()
    if traded_df.empty:
        raise RuntimeError("No rows with trades > 0 after filtering.")
    if len(traded_df) < len(df):
        print(
            f"Excluded {len(df) - len(traded_df)} zero-trade runs "
            f"(kept {len(traded_df)} traded runs)."
        )
    df = traded_df

    removed = _clean_old_plots(plots_dir, prefix)
    print(f"Removed {removed} old plot(s) with prefix '{prefix}_'.")
    tables_dir.mkdir(parents=True, exist_ok=True)

    strategy_stats = _group_stats(df, "strategy").sort_values(
        "median_ann_benchmark_excess", ascending=False
    )
    symbol_stats = _group_stats(df, "symbol").sort_values(
        "median_ann_benchmark_excess", ascending=False
    )
    interval_stats = _group_stats(df, "interval").sort_values(
        "median_ann_benchmark_excess", ascending=False
    )
    top_df = df.sort_values("composite_score", ascending=False).head(top_runs)

    # 00a/00b Rebuilt from scratch using a single top strategy.
    score_cols = [
        "benchmark_excess_return",
        "annualized_benchmark_excess",
        "excess_return",
        "return",
    ]
    ranked = df.copy()
    rank_cols = []
    for c in score_cols:
        rc = f"rank_{c}"
        ranked[rc] = ranked[c].rank(pct=True, ascending=True)
        rank_cols.append(rc)
    ranked["benchmark_outperformance_score"] = ranked[rank_cols].mean(axis=1)

    preferred_ranked = ranked.sort_values("benchmark_outperformance_score", ascending=False)

    # Pick first candidate that can actually rerun.
    top_run = None
    data_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    for _, cand in preferred_ranked.head(120).iterrows():
        params_raw = cand.get("parameters", "{}")
        risk_raw = cand.get("risk_profile", "{}")
        try:
            param_dict = (
                json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw)
            )
        except Exception:
            param_dict = {}
        try:
            risk_cfg = json.loads(risk_raw) if isinstance(risk_raw, str) else dict(risk_raw)
        except Exception:
            risk_cfg = None

        try:
            eq_df = _run_single_strategy_from_raw(
                strategy_name=str(cand["strategy"]),
                symbol=str(cand["symbol"]),
                interval=str(cand["interval"]),
                param_dict=param_dict,
                commission=float(cand["commission"]),
                slippage=float(cand["slippage"]),
                sizer=float(cand["sizer"]),
                risk_config=risk_cfg,
                cache=data_cache,
            )
            eq = eq_df["equity"].astype(float).dropna()
            if eq.empty or float(eq.iloc[0]) == 0.0:
                continue
            top_run = cand.copy()
            top_run["run_label"] = f"{top_run['strategy']} | {top_run['symbol']} {top_run['interval']}"
            strategy_curve = (eq / float(eq.iloc[0])) * 100.0
            break
        except Exception:
            continue

    if top_run is None:
        top_run = preferred_ranked.iloc[0].copy()
        top_run["run_label"] = (
            f"{top_run['strategy']} | {top_run['symbol']} {top_run['interval']} (summary only)"
        )
        strategy_curve = None
        print("Could not rerun any top candidate; equity chart will show benchmarks only.")

    # 00a Bar chart: Sharpe included in same plot as the other metrics.
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = [
        ("return", "Return"),
        ("excess_return", "Excess Return"),
        ("benchmark_excess_return", "Bench Excess"),
        ("annualized_return", "Annualized Return"),
        ("sharpe", "Sharpe"),
    ]
    x = np.arange(len(metrics))
    vals = np.array(
        [float(top_run[k]) if np.isfinite(float(top_run[k])) else 0.0 for k, _ in metrics],
        dtype=float,
    )
    colors = ["#4c78a8", "#4c78a8", "#4c78a8", "#4c78a8", "#f58518"]
    bars = ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Top Strategy Metrics: {top_run['run_label']}")
    ax.grid(axis="y", alpha=0.2)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    _save(fig, plots_dir, prefix, "00a_top_strategy_metrics")

    def _plot_equity_chart(
        run_row: pd.Series | None,
        run_curve: pd.Series | None,
        title: str,
        save_name: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_any = False
        if run_row is not None and run_curve is not None and not run_curve.empty:
            min_dt = pd.to_datetime(run_curve.index.min())
            max_dt = pd.to_datetime(run_curve.index.max())
            ax.plot(
                run_curve.index,
                run_curve.values,
                linewidth=2.0,
                color="#1f77b4",
                label=str(run_row["run_label"]),
            )
            plotted_any = True

            bench_interval = "1d"
            if _load_price_df("^GSPC", bench_interval, data_cache) is None:
                bench_interval = str(run_row["interval"])
            strategy_symbol = str(run_row["symbol"])
            if _load_price_df(strategy_symbol, bench_interval, data_cache) is None:
                bench_interval = str(run_row["interval"])

            benchmark_specs = [("^GSPC", "S&P 500", "#2ca02c")]
            if strategy_symbol != "^GSPC":
                benchmark_specs.append(
                    (strategy_symbol, f"{strategy_symbol} Buy&Hold", "#ff7f0e")
                )

            for sym, label, color in benchmark_specs:
                bench = _load_price_df(sym, bench_interval, data_cache)
                if bench is None or "close" not in bench.columns:
                    continue
                close = bench["close"].astype(float).dropna()
                close.index = pd.to_datetime(close.index)
                close = close[(close.index >= min_dt) & (close.index <= max_dt)]
                if close.empty or float(close.iloc[0]) == 0.0:
                    continue
                eq_bench = (close / float(close.iloc[0])) * 100.0
                ax.plot(eq_bench.index, eq_bench.values, linewidth=2.0, color=color, label=label)
                plotted_any = True

        ax.axhline(100.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("Normalized Equity (Start=100)")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.2)
        if plotted_any:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No curves available (missing rerun/benchmark data).",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
        _save(fig, plots_dir, prefix, save_name)

    def _find_best_rerunnable_by_metric(
        metric_col: str, top_n: int = 200
    ) -> tuple[pd.Series | None, pd.Series | None]:
        ranked_metric = df.sort_values(metric_col, ascending=False).head(top_n)
        for _, cand in ranked_metric.iterrows():
            params_raw = cand.get("parameters", "{}")
            risk_raw = cand.get("risk_profile", "{}")
            try:
                param_dict = (
                    json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw)
                )
            except Exception:
                param_dict = {}
            try:
                risk_cfg = json.loads(risk_raw) if isinstance(risk_raw, str) else dict(risk_raw)
            except Exception:
                risk_cfg = None

            try:
                eq_df = _run_single_strategy_from_raw(
                    strategy_name=str(cand["strategy"]),
                    symbol=str(cand["symbol"]),
                    interval=str(cand["interval"]),
                    param_dict=param_dict,
                    commission=float(cand["commission"]),
                    slippage=float(cand["slippage"]),
                    sizer=float(cand["sizer"]),
                    risk_config=risk_cfg,
                    cache=data_cache,
                )
                eq = eq_df["equity"].astype(float).dropna()
                if eq.empty or float(eq.iloc[0]) == 0.0:
                    continue
                run_row = cand.copy()
                run_row["run_label"] = (
                    f"{run_row['strategy']} | {run_row['symbol']} {run_row['interval']}"
                )
                run_curve = (eq / float(eq.iloc[0])) * 100.0
                return run_row, run_curve
            except Exception:
                continue
        return None, None

    # 00b Equity curves: top strategy (benchmark outperformance composite) + benchmarks.
    _plot_equity_chart(
        run_row=top_run if strategy_curve is not None else None,
        run_curve=strategy_curve if strategy_curve is not None else None,
        title="Top Strategy Equity Curve vs S&P 500 and Market Buy&Hold",
        save_name="00b_top_strategy_equity_vs_sp500_market",
    )

    # 00c/00d/00e Same equity chart for best excess return, best sharpe, best return.
    metric_specs = [
        ("excess_return", "Best Excess Return Equity Curve vs S&P 500 and Market Buy&Hold", "00c_best_excess_return_equity_vs_sp500_market"),
        ("sharpe", "Best Sharpe Equity Curve vs S&P 500 and Market Buy&Hold", "00d_best_sharpe_equity_vs_sp500_market"),
        ("return", "Best Return Equity Curve vs S&P 500 and Market Buy&Hold", "00e_best_return_equity_vs_sp500_market"),
    ]
    for metric_col, title, save_name in metric_specs:
        best_row, best_curve = _find_best_rerunnable_by_metric(metric_col=metric_col)
        if best_row is None:
            print(f"No rerunnable candidate found for metric '{metric_col}'.")
        _plot_equity_chart(
            run_row=best_row,
            run_curve=best_curve,
            title=title,
            save_name=save_name,
        )

    # Save key tables at shallow path.
    strategy_stats.to_csv(tables_dir / f"{prefix}_strategy_stats.csv", index=False)
    symbol_stats.to_csv(tables_dir / f"{prefix}_symbol_stats.csv", index=False)
    interval_stats.to_csv(tables_dir / f"{prefix}_interval_stats.csv", index=False)
    top_df.to_csv(tables_dir / f"{prefix}_top_runs.csv", index=False)

    # 1 Return histogram.
    fig, ax = plt.subplots(figsize=(10, 5))
    ret = df["return"].dropna()
    ax.hist(ret, bins=80, alpha=0.85)
    bounds = _robust_bounds(ret)
    if bounds:
        ax.set_xlim(bounds)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Return Distribution (%)")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    _annotate_hist_stats(ax, ret, unit="%")
    _save(fig, plots_dir, prefix, "01_return_hist")

    # 2 Sharpe histogram.
    fig, ax = plt.subplots(figsize=(10, 5))
    shp = df["sharpe"].dropna()
    ax.hist(shp, bins=80, alpha=0.85)
    bounds = _robust_bounds(shp)
    if bounds:
        ax.set_xlim(bounds)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Sharpe Distribution")
    ax.set_xlabel("Sharpe")
    ax.set_ylabel("Count")
    _annotate_hist_stats(ax, shp, unit="")
    _save(fig, plots_dir, prefix, "02_sharpe_hist")

    # 3 Drawdown histogram.
    fig, ax = plt.subplots(figsize=(10, 5))
    dd = df["max_drawdown_pct"].dropna()
    ax.hist(dd, bins=80, alpha=0.85)
    bounds = _robust_bounds(dd)
    if bounds:
        ax.set_xlim(bounds)
    ax.set_title("Max Drawdown Distribution (%)")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Count")
    _annotate_hist_stats(ax, dd, unit="%")
    _save(fig, plots_dir, prefix, "03_drawdown_hist")

    # 3b Duration histogram to show window differences across assets/intervals.
    fig, ax = plt.subplots(figsize=(10, 5))
    dur = df["duration_days"].dropna()
    ax.hist(dur, bins=60, alpha=0.85)
    bounds = _robust_bounds(dur)
    if bounds:
        ax.set_xlim(bounds)
    ax.set_title("Backtest Window Length Distribution (Days)")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")
    _annotate_hist_stats(ax, dur, unit="d")
    _save(fig, plots_dir, prefix, "03b_duration_days_hist")

    # 4 Return vs Sharpe.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot = df.sort_values("composite_score", ascending=False).head(min(2500, len(df)))
    ax.scatter(plot["sharpe"], plot["return"], s=14, alpha=0.35)
    x_bounds = _robust_bounds(plot["sharpe"])
    y_bounds = _robust_bounds(plot["return"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)
    ax.set_title("Return vs Sharpe (Top Composite Runs)")
    ax.set_xlabel("Sharpe")
    ax.set_ylabel("Return (%)")
    _add_scatter_regression(ax, plot["sharpe"], plot["return"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "04_scatter_return_vs_sharpe")

    # 4b Return vs Sharpe (only positive Sharpe).
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pos_sharpe = plot[plot["sharpe"] > 0].copy()
    ax.scatter(plot_pos_sharpe["sharpe"], plot_pos_sharpe["return"], s=14, alpha=0.35)
    x_bounds = _robust_bounds(plot_pos_sharpe["sharpe"])
    y_bounds = _robust_bounds(plot_pos_sharpe["return"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)
    ax.set_title("Return vs Sharpe (Sharpe > 0)")
    ax.set_xlabel("Sharpe")
    ax.set_ylabel("Return (%)")
    _add_scatter_regression(ax, plot_pos_sharpe["sharpe"], plot_pos_sharpe["return"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "04b_scatter_return_vs_sharpe_positive_only")

    # 5 Return vs Drawdown.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot["max_drawdown_pct"], plot["return"], s=14, alpha=0.35)
    x_bounds = _robust_bounds(plot["max_drawdown_pct"])
    y_bounds = _robust_bounds(plot["return"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)
    ax.set_title("Return vs Max Drawdown (Top Composite Runs)")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Return (%)")
    _add_scatter_regression(ax, plot["max_drawdown_pct"], plot["return"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "05_scatter_return_vs_drawdown")

    # 6 Trades vs Return.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot["trades"], plot["return"], s=14, alpha=0.35)
    x_bounds = _robust_bounds(plot["trades"])
    y_bounds = _robust_bounds(plot["return"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)
    ax.set_title("Trades vs Return")
    ax.set_xlabel("Trades")
    ax.set_ylabel("Return (%)")
    _add_scatter_regression(ax, plot["trades"], plot["return"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "06_scatter_trades_vs_return")

    # 06b Market excess vs benchmark outperformance (top runs, colored by strategy).
    fig, ax = plt.subplots(figsize=(11, 7))
    top_n = 1000
    plot_base = df.sort_values("composite_score", ascending=False).head(min(top_n, len(df)))
    plot_all = plot_base[["strategy", "excess_return", "benchmark_excess_return"]].copy()
    plot_all = plot_all.dropna(subset=["strategy", "excess_return", "benchmark_excess_return"])
    plot_all = plot_all[
        np.isfinite(plot_all["excess_return"].astype(float))
        & np.isfinite(plot_all["benchmark_excess_return"].astype(float))
    ]

    strategies = sorted(plot_all["strategy"].astype(str).unique().tolist())
    color_map = _strategy_color_map(strategies)

    # Plot each strategy in a distinct color.
    for strat in strategies:
        sub = plot_all[plot_all["strategy"].astype(str) == strat]
        if sub.empty:
            continue
        ax.scatter(
            sub["excess_return"].astype(float),
            sub["benchmark_excess_return"].astype(float),
            s=18,
            alpha=0.55,
            color=color_map.get(strat, "#4c78a8"),
        )

    # Dotted black zero lines on both axes.
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.9)

    x_bounds = _robust_bounds(plot_all["excess_return"])
    y_bounds = _robust_bounds(plot_all["benchmark_excess_return"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)

    ax.set_title(f"Benchmark Outperformance vs Market Excess Return (Top {len(plot_base)} Composite Runs)")
    ax.set_xlabel("Excess Return vs Tested Market Buy&Hold (%)")
    ax.set_ylabel("Benchmark Outperformance (%)")
    ax.grid(alpha=0.2)

    if strategies:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=color_map.get(s, "#4c78a8"),
                markeredgecolor="none",
                label=s,
            )
            for s in strategies
        ]
        ax.legend(
            handles=handles,
            title="Strategy",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
            frameon=False,
        )

    _save(fig, plots_dir, prefix, "06b_scatter_benchmark_outperformance_vs_market_excess_by_strategy")

    # 7 Strategy mean return.
    fig, ax = plt.subplots(figsize=(12, 5))
    s = strategy_stats.sort_values("mean_return", ascending=False)
    ax.bar(s["strategy"], s["mean_return"])
    ax.set_title("Mean Return by Strategy")
    ax.set_ylabel("Mean Return (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_bar_values(ax, max_labels=12, as_pct=False)
    _save(fig, plots_dir, prefix, "07_bar_strategy_mean_return")

    # 8 Strategy benchmark excess (duration-adjusted).
    fig, ax = plt.subplots(figsize=(12, 5))
    s = strategy_stats.sort_values(
        "duration_weighted_ann_benchmark_excess", ascending=False
    )
    ax.bar(s["strategy"], s["duration_weighted_ann_benchmark_excess"])
    ax.set_title("Duration-Weighted Annualized Benchmark Excess by Strategy")
    ax.set_ylabel("Annualized Excess Return (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_bar_values(ax, max_labels=12, as_pct=False)
    _save(fig, plots_dir, prefix, "08_bar_strategy_duration_weighted_ann_benchmark_excess")

    # 8b Strategy annualized benchmark excess (length-aware).
    fig, ax = plt.subplots(figsize=(12, 5))
    s = strategy_stats.sort_values("median_ann_benchmark_excess", ascending=False)
    ax.bar(s["strategy"], s["median_ann_benchmark_excess"])
    ax.set_title("Median Annualized Benchmark Excess by Strategy")
    ax.set_ylabel("Annualized Excess Return (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_bar_values(ax, max_labels=12, as_pct=False)
    _save(fig, plots_dir, prefix, "08b_bar_strategy_median_ann_benchmark_excess")

    # 9 Strategy benchmark beat rate.
    fig, ax = plt.subplots(figsize=(12, 5))
    s = strategy_stats.sort_values("pct_beat_benchmark", ascending=False)
    ax.bar(s["strategy"], s["pct_beat_benchmark"])
    ax.set_title("Benchmark Beat Rate by Strategy")
    ax.set_ylabel("Beat Rate")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    _annotate_bar_values(ax, max_labels=12, as_pct=True)
    _save(fig, plots_dir, prefix, "09_bar_strategy_beat_rate")

    # 10 Symbol benchmark excess (duration-adjusted).
    fig, ax = plt.subplots(figsize=(12, 5))
    s = symbol_stats.sort_values(
        "duration_weighted_ann_benchmark_excess", ascending=False
    )
    ax.bar(s["symbol"], s["duration_weighted_ann_benchmark_excess"])
    ax.set_title("Duration-Weighted Annualized Benchmark Excess by Symbol")
    ax.set_ylabel("Annualized Excess Return (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0, color="black", linewidth=1)
    _annotate_bar_values(ax, max_labels=16, as_pct=False)
    _save(fig, plots_dir, prefix, "10_bar_symbol_duration_weighted_ann_benchmark_excess")

    # 11 Boxplot return by interval.
    fig, ax = plt.subplots(figsize=(8, 5))
    intervals = sorted(df["interval"].dropna().unique())
    vals = [df[df["interval"] == i]["return"].dropna().values for i in intervals]
    ax.boxplot(vals, tick_labels=intervals, showfliers=False)
    ax.set_title("Return Distribution by Interval")
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="black", linewidth=1)
    for idx, arr in enumerate(vals, start=1):
        if len(arr) == 0:
            continue
        ax.scatter(idx, float(np.mean(arr)), marker="D", s=34, color="darkorange", zorder=3)
        ax.text(idx, float(np.mean(arr)), f"{np.mean(arr):.2f}", fontsize=8, ha="center", va="bottom")
    _save(fig, plots_dir, prefix, "11_boxplot_return_by_interval")

    # 12 Boxplot benchmark excess by top strategies.
    fig, ax = plt.subplots(figsize=(13, 5))
    top_names = strategy_stats.sort_values(
        "median_benchmark_excess", ascending=False
    )["strategy"].head(top_strategies)
    vals = [
        df[df["strategy"] == n]["benchmark_excess_return"].dropna().values
        for n in top_names
    ]
    bp = ax.boxplot(vals, tick_labels=list(top_names), showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#cfe8ff", edgecolor="#4a4a4a", linewidth=1.0)
    for median in bp["medians"]:
        median.set(color="#d62728", linewidth=1.4)
    for whisker in bp["whiskers"]:
        whisker.set(color="#666666", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="#666666", linewidth=1.0)
    ax.set_title("Benchmark Excess by Top Strategies")
    ax.set_ylabel("Benchmark Excess Return (%)")
    ax.axhline(0, color="black", linewidth=1)
    ax.tick_params(axis="x", rotation=35)
    for idx, arr in enumerate(vals, start=1):
        if len(arr) == 0:
            continue
        mean_v = float(np.mean(arr))
        ax.scatter(idx, mean_v, marker="D", s=24, color="darkorange", zorder=3)

    legend_handles = [
        Patch(facecolor="#cfe8ff", edgecolor="#4a4a4a", label="IQR (Q1-Q3)"),
        Line2D([0], [0], color="#d62728", lw=1.4, label="Median"),
        Line2D([0], [0], color="#666666", lw=1.0, label="Whiskers (excl. outliers)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="darkorange", markersize=7, label="Mean"),
    ]
    non_empty = sum(1 for v in vals if len(v) > 0)
    total_points = int(sum(len(v) for v in vals))
    ax.legend(
        handles=legend_handles,
        title=f"Top strategies: {non_empty} | Points: {total_points}",
        loc="best",
        framealpha=0.9,
    )
    _save(fig, plots_dir, prefix, "12_boxplot_benchmark_excess_top_strategies")

    # 13 Heatmap strategy vs symbol.
    pivot = df.pivot_table(
        index="strategy",
        columns="symbol",
        values="benchmark_excess_return",
        aggfunc="median",
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(pivot.columns)), 6))
        img = ax.imshow(pivot.values, aspect="auto")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_title("Median Benchmark Excess: Strategy vs Symbol")
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Median Benchmark Excess Return (%)")
        if pivot.shape[0] <= 14 and pivot.shape[1] <= 14:
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    v = pivot.values[r, c]
                    if pd.notna(v):
                        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7, color="black")
        _save(fig, plots_dir, prefix, "13_heatmap_strategy_symbol")

    # 14 Correlation heatmap.
    corr_cols = [
        "return",
        "sharpe",
        "trades",
        "win_rate",
        "benchmark_excess_return",
        "annualized_return",
        "max_drawdown_pct",
        "calmar",
        "profit_factor",
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    img = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Metric Correlation Heatmap")
    for r in range(corr.shape[0]):
        for c in range(corr.shape[1]):
            v = corr.values[r, c]
            if pd.notna(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(img, ax=ax)
    _save(fig, plots_dir, prefix, "14_heatmap_metric_correlation")

    # 15 CDF of returns.
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_returns = np.sort(df["return"].dropna().values)
    y = np.linspace(0, 1, len(sorted_returns), endpoint=True)
    ax.plot(sorted_returns, y)
    bounds = _robust_bounds(pd.Series(sorted_returns))
    if bounds:
        ax.set_xlim(bounds)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Empirical CDF of Returns")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Cumulative Probability")
    if len(sorted_returns) > 0:
        p25 = float(np.quantile(sorted_returns, 0.25))
        p50 = float(np.quantile(sorted_returns, 0.50))
        p75 = float(np.quantile(sorted_returns, 0.75))
        for p, label in [(p25, "p25"), (p50, "p50"), (p75, "p75")]:
            ax.axvline(p, linestyle=":", linewidth=1.0, color="gray")
            ax.text(p, 0.02, f"{label}={p:.1f}", rotation=90, fontsize=8, va="bottom", ha="right")
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "15_return_cdf")

    # 16 Annualized benchmark excess distribution (length-aware metric).
    fig, ax = plt.subplots(figsize=(10, 5))
    ann_exc = df["annualized_benchmark_excess"].dropna()
    ax.hist(ann_exc, bins=80, alpha=0.85)
    bounds = _robust_bounds(ann_exc)
    if bounds:
        ax.set_xlim(bounds)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Annualized Benchmark Excess Distribution (%)")
    ax.set_xlabel("Annualized Benchmark Excess (%)")
    ax.set_ylabel("Count")
    _annotate_hist_stats(ax, ann_exc, unit="%")
    _save(fig, plots_dir, prefix, "16_ann_benchmark_excess_hist")

    # 17 Duration vs annualized benchmark excess with regression.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["duration_days"], df["annualized_benchmark_excess"], s=14, alpha=0.35)
    x_bounds = _robust_bounds(df["duration_days"])
    y_bounds = _robust_bounds(df["annualized_benchmark_excess"])
    if x_bounds:
        ax.set_xlim(x_bounds)
    if y_bounds:
        ax.set_ylim(y_bounds)
    ax.set_title("Duration vs Annualized Benchmark Excess")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Annualized Benchmark Excess (%)")
    _add_scatter_regression(ax, df["duration_days"], df["annualized_benchmark_excess"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "17_scatter_duration_vs_ann_benchmark_excess")

    # 18 Top results: annualized benchmark excess leaderboard.
    top_ann = df.sort_values("annualized_benchmark_excess", ascending=False).head(25).copy()
    top_ann["label"] = (
        top_ann["strategy"].astype(str)
        + " | "
        + top_ann["symbol"].astype(str)
        + " "
        + top_ann["interval"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(top_ann["label"], top_ann["annualized_benchmark_excess"])
    ax.set_title("Top 25 Runs by Annualized Benchmark Excess")
    ax.set_xlabel("Annualized Benchmark Excess (%)")
    ax.invert_yaxis()
    values = top_ann["annualized_benchmark_excess"].to_numpy(dtype=float)
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size > 0:
        lo = float(np.min(finite_vals))
        hi = float(np.max(finite_vals))
        span = max(1.0, hi - lo)
        pad = 0.04 * span
        ax.set_xlim(lo - pad * 2.0, hi + pad * 2.0)
    for bar, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        y = bar.get_y() + bar.get_height() / 2.0
        if v >= 0:
            x = v + (0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
            ha = "left"
        else:
            x = v - (0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
            ha = "right"
        ax.text(x, y, f"{v:.2f}", va="center", ha=ha, fontsize=8, clip_on=True)
    _save(fig, plots_dir, prefix, "18_top25_ann_benchmark_excess")

    # 19 Top results profile: duration and drawdown for top composite runs.
    top_comp = df.sort_values("composite_score", ascending=False).head(40).copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        top_comp["duration_days"],
        top_comp["annualized_benchmark_excess"],
        c=top_comp["max_drawdown_pct"],
        s=np.clip(top_comp["trades"].fillna(1).values, 10, 250),
        alpha=0.7,
    )
    ax.set_title("Top Composite Runs: Duration vs Annualized Excess")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Annualized Benchmark Excess (%)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Max Drawdown (%)")
    _add_scatter_regression(ax, top_comp["duration_days"], top_comp["annualized_benchmark_excess"])
    ax.grid(alpha=0.2)
    _save(fig, plots_dir, prefix, "19_top_composite_duration_vs_ann_excess")

    # Additional top-results tables for deeper review.
    top_ann_cols = [
        "strategy",
        "symbol",
        "interval",
        "annualized_benchmark_excess",
        "benchmark_excess_return",
        "annualized_return",
        "annualized_benchmark_gain",
        "duration_days",
        "max_drawdown_pct",
        "sharpe",
        "trades",
        "parameters",
    ]
    top_ann[top_ann_cols].to_csv(
        tables_dir / f"{prefix}_top_annualized_benchmark_excess.csv", index=False
    )

    top_composite_cols = [
        "strategy",
        "symbol",
        "interval",
        "composite_score",
        "annualized_benchmark_excess",
        "return",
        "benchmark_excess_return",
        "duration_days",
        "max_drawdown_pct",
        "sharpe",
        "trades",
        "parameters",
    ]
    top_comp[top_composite_cols].to_csv(
        tables_dir / f"{prefix}_top_composite_detailed.csv", index=False
    )

    print(f"Analysis done. Plots saved in : {plots_dir}")
    print(f"Analysis done. Tables saved in: {tables_dir}")
    print(f"File prefix: {prefix}_*")


def main() -> None:
    args = parse_args()
    generate_analysis(
        results_path=args.results_path,
        plots_dir=args.plots_dir,
        tables_dir=args.tables_dir,
        prefix=args.prefix,
        backtest_version=args.backtest_version,
        risk_profile=args.risk_profile,
        top_strategies=args.top_strategies,
        top_runs=args.top_runs,
    )


if __name__ == "__main__":
    main()

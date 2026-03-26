import argparse
import json
from copy import deepcopy
from pathlib import Path

import backtrader as bt
import pandas as pd
import yaml

from strategies import ALL_STRATEGIES


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RESULTS_PATH = PROJECT_ROOT / "database" / "results.parquet"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "results" / "risk_optimization.csv"


SEARCH_SPACE = {
    "risk_per_trade": [0.003, 0.005, 0.0075, 0.01, 0.015],
    "max_position_value_pct": [0.35, 0.50, 0.70],
    "drawdown_step_1": [0.07, 0.10, 0.12],
    "drawdown_mult_1": [0.40, 0.50, 0.60],
    "drawdown_step_2": [0.15, 0.20, 0.25],
    "drawdown_mult_2": [0.20, 0.25, 0.33],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize risk config for backtests.")
    parser.add_argument("--symbols", default="AAPL,EURUSD=X,USO")
    parser.add_argument("--intervals", default="1d,1h,15m")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--version", default=None)
    return parser.parse_args()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_interval_to_timeframe(interval_cfg):
    bt_timeframe_map = {
        "Minutes": bt.TimeFrame.Minutes,
        "Days": bt.TimeFrame.Days,
        "Weeks": bt.TimeFrame.Weeks,
    }
    return {
        interval: bt_timeframe_map[name]
        for interval, name in interval_cfg.items()
        if name in bt_timeframe_map
    }


def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    return pd.read_parquet(RESULTS_PATH)


def choose_reference_version(df, requested_version=None):
    if requested_version is not None:
        return requested_version
    if "backtest_version" not in df.columns:
        return None
    return df["backtest_version"].value_counts().index[0]


def pick_best_parameters(df, symbols, intervals, version):
    work = df.copy()
    if version is not None and "backtest_version" in work.columns:
        work = work[work["backtest_version"] == version]
    work = work[work["symbol"].isin(symbols) & work["interval"].isin(intervals)]
    if work.empty:
        raise RuntimeError(
            "No baseline runs found for selected symbols/intervals/version."
        )

    grouped = (
        work.groupby(["strategy", "interval", "parameters"], as_index=False)
        .agg(
            sharpe_mean=("sharpe", "mean"),
            return_mean=("return", "mean"),
            trades_mean=("trades", "mean"),
        )
        .sort_values(
            by=["strategy", "interval", "sharpe_mean", "return_mean", "trades_mean"],
            ascending=[True, True, False, False, False],
        )
    )
    best = grouped.groupby(["strategy", "interval"], as_index=False).first()
    best["param_dict"] = best["parameters"].apply(json.loads)
    return best


def build_scenarios(best_params, symbols):
    scenarios = []
    for _, row in best_params.iterrows():
        strategy = row["strategy"]
        interval = row["interval"]
        param_dict = row["param_dict"]
        for symbol in symbols:
            data_path = DATA_DIR / f"{symbol}_{interval}.csv"
            if data_path.exists():
                scenarios.append(
                    {
                        "strategy": strategy,
                        "symbol": symbol,
                        "interval": interval,
                        "params": param_dict,
                        "data_path": data_path,
                    }
                )
    if not scenarios:
        raise RuntimeError("No scenarios could be built from selected symbols/intervals.")
    return scenarios


def valid_risk_config(cfg):
    return cfg["drawdown_step_2"] > cfg["drawdown_step_1"]


def evaluate_config(
    scenarios,
    commission,
    sizer,
    interval_to_timeframe,
    risk_config,
    data_cache,
):
    rows = []
    for sc in scenarios:
        tuned_risk = dict(risk_config)
        tuned_risk["sizer_pct"] = float(sizer)
        tuned_risk["max_position_value_pct"] = float(sizer) / 100.0
        key = str(sc["data_path"])
        if key not in data_cache:
            data_cache[key] = pd.read_csv(sc["data_path"], index_col=0, parse_dates=True)

        price_df = data_cache[key]
        data_feed = bt.feeds.PandasData(dataname=price_df.copy())
        run_fn = ALL_STRATEGIES[sc["strategy"]]

        res = run_fn(
            data=data_feed,
            commission_=commission,
            sizer=sizer,
            interval=sc["interval"],
            interval_to_timeframe=interval_to_timeframe,
            risk_config=tuned_risk,
            **sc["params"],
        )

        ta = res.analyzers.trades.get_analysis()
        sa = res.analyzers.sharpe.get_analysis()
        total_trades = ta.total.total if "total" in ta else 0
        strategy_return = ((res.broker.getvalue() - 1000) / 1000) * 100

        rows.append(
            {
                "strategy": sc["strategy"],
                "symbol": sc["symbol"],
                "interval": sc["interval"],
                "return": strategy_return,
                "sharpe": sa.get("sharperatio", 0)
                if sa.get("sharperatio") is not None
                else 0,
                "trades": total_trades,
            }
        )

    df = pd.DataFrame(rows)
    sharpe_clipped = df["sharpe"].clip(-3.0, 3.0)
    mean_sharpe = float(sharpe_clipped.mean())
    median_sharpe = float(sharpe_clipped.median())
    mean_return = float(df["return"].mean())
    median_return = float(df["return"].median())
    positive_return_rate = float((df["return"] > 0).mean())
    trade_coverage = float((df["trades"] > 0).mean())

    # Robust objective: clipped Sharpe + return + robustness terms.
    score = (
        (30.0 * median_sharpe)
        + (20.0 * mean_sharpe)
        + (0.5 * mean_return)
        + (0.5 * median_return)
        + (10.0 * positive_return_rate)
        + (5.0 * trade_coverage)
    )

    return {
        "score": score,
        "mean_sharpe": mean_sharpe,
        "median_sharpe": median_sharpe,
        "mean_return": mean_return,
        "median_return": median_return,
        "positive_return_rate": positive_return_rate,
        "trade_coverage": trade_coverage,
        "runs": len(df),
    }


def run_coordinate_search(
    base_risk,
    scenarios,
    commission,
    sizer,
    interval_to_timeframe,
    passes,
):
    data_cache = {}
    results_rows = []

    current = deepcopy(base_risk)
    current_metrics = evaluate_config(
        scenarios, commission, sizer, interval_to_timeframe, current, data_cache
    )
    results_rows.append(
        {
            "step": "baseline",
            "key": "baseline",
            "value": json.dumps(current, sort_keys=True),
            **current_metrics,
        }
    )

    for pass_idx in range(passes):
        for key, values in SEARCH_SPACE.items():
            best_local_cfg = deepcopy(current)
            best_local_metrics = current_metrics

            for v in values:
                candidate = deepcopy(current)
                candidate[key] = float(v)
                if not valid_risk_config(candidate):
                    continue

                metrics = evaluate_config(
                    scenarios,
                    commission,
                    sizer,
                    interval_to_timeframe,
                    candidate,
                    data_cache,
                )
                results_rows.append(
                    {
                        "step": f"pass_{pass_idx + 1}",
                        "key": key,
                        "value": v,
                        "candidate": json.dumps(candidate, sort_keys=True),
                        **metrics,
                    }
                )

                if metrics["score"] > best_local_metrics["score"]:
                    best_local_cfg = candidate
                    best_local_metrics = metrics

            current = best_local_cfg
            current_metrics = best_local_metrics

    return current, current_metrics, pd.DataFrame(results_rows)


def apply_best_to_config(cfg, best_risk):
    cfg["risk"] = {
        "risk_per_trade": float(best_risk["risk_per_trade"]),
        "max_position_value_pct": float(best_risk["max_position_value_pct"]),
        "drawdown_step_1": float(best_risk["drawdown_step_1"]),
        "drawdown_step_2": float(best_risk["drawdown_step_2"]),
        "drawdown_mult_1": float(best_risk["drawdown_mult_1"]),
        "drawdown_mult_2": float(best_risk["drawdown_mult_2"]),
        "min_risk_per_trade": float(best_risk["min_risk_per_trade"]),
        "min_stop_distance": float(best_risk["min_stop_distance"]),
        "min_trade_size": float(best_risk["min_trade_size"]),
        "default_stop_atr": float(best_risk["default_stop_atr"]),
        "atr_len_for_sizing": int(best_risk["atr_len_for_sizing"]),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    cfg = load_config()
    df = load_results()
    version = choose_reference_version(df, args.version)
    interval_to_timeframe = make_interval_to_timeframe(cfg["INTERVAL_TO_TIMEFRAME"])
    commission = float(cfg["commission"])
    sizer = float(cfg["sizer"])

    base_risk = deepcopy(cfg.get("risk", {}))
    for k, v in {
        "risk_per_trade": 0.005,
        "max_position_value_pct": 0.50,
        "drawdown_step_1": 0.10,
        "drawdown_step_2": 0.20,
        "drawdown_mult_1": 0.50,
        "drawdown_mult_2": 0.25,
        "min_risk_per_trade": 0.001,
        "min_stop_distance": 1.0e-8,
        "min_trade_size": 1.0e-6,
        "default_stop_atr": 2.0,
        "atr_len_for_sizing": 14,
    }.items():
        base_risk.setdefault(k, v)

    best_params = pick_best_parameters(df, symbols, intervals, version)
    scenarios = build_scenarios(best_params, symbols)

    best_risk, best_metrics, log_df = run_coordinate_search(
        base_risk=base_risk,
        scenarios=scenarios,
        commission=commission,
        sizer=sizer,
        interval_to_timeframe=interval_to_timeframe,
        passes=args.passes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(output_path, index=False)

    if args.apply_best:
        apply_best_to_config(cfg, best_risk)

    print(f"Reference version: {version}")
    print(f"Scenarios evaluated per candidate: {len(scenarios)}")
    print("Best risk config:")
    print(json.dumps(best_risk, indent=2, sort_keys=True))
    print("Best metrics:")
    print(json.dumps(best_metrics, indent=2, sort_keys=True))
    print(f"Optimization log saved to: {output_path}")
    if args.apply_best:
        print(f"Applied best config to: {CONFIG_PATH}")


if __name__ == "__main__":
    main()

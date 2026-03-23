import json
from itertools import product
from pathlib import Path

import backtrader as bt
import pandas as pd
import yaml
from tqdm import tqdm

from strategies import ALL_STRATEGIES
from utils.results_logger import make_file


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "raw"
RESULTS_PATH = PROJECT_ROOT / "database" / "results.parquet"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
BACKTEST_VERSION = "long_short_risk_v3"
STARTING_CASH = 1000

BT_TIMEFRAME_MAP = {
    "Minutes": bt.TimeFrame.Minutes,
    "Days": bt.TimeFrame.Days,
    "Weeks": bt.TimeFrame.Weeks,
}

# Maps config parameter names to strategy runtime argument names.
STRATEGY_PARAM_SPECS = {
    "DMAC": [
        ("short_ema", "pfast"),
        ("long_ema", "pslow"),
        ("adx_period", "adx_period"),
        ("adx_threshold", "adx_threshold"),
    ],
    "RSI_MA": [
        ("ma", "ma"),
        ("buy_rsi", "buy_rsi"),
        ("sell_rsi", "sell_rsi"),
    ],
    "BBANDS_MR": [
        ("bb_len", "bb_len"),
        ("bb_k", "bb_k"),
        ("rsi_os", "rsi_os"),
        ("adx_max", "adx_max"),
        ("stop_atr", "stop_atr"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "DONCHIAN_BO": [
        ("entry_len", "entry_len"),
        ("exit_len", "exit_len"),
        ("atr_len", "atr_len"),
        ("entry_buffer_atr", "entry_buffer_atr"),
        ("stop_atr", "stop_atr"),
        ("trail_atr", "trail_atr"),
        ("max_hold_bars", "max_hold_bars"),
        ("cooldown_bars", "cooldown_bars"),
    ],
    "RSI_PULLBACK": [
        ("trend_len", "trend_len"),
        ("slope_lookback", "slope_lookback"),
        ("rsi_len", "rsi_len"),
        ("rsi_pullback", "rsi_pullback"),
        ("rsi_recover", "rsi_recover"),
        ("atr_len", "atr_len"),
        ("stop_atr", "stop_atr"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "ATR_VOL_BO": [
        ("atr_len", "atr_len"),
        ("atr_expansion_mult", "atr_expansion_mult"),
        ("stop_atr", "stop_atr"),
        ("breakout_len", "breakout_len"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "MA_TREND_CONT": [
        ("trend_len", "trend_len"),
        ("slope_lookback", "slope_lookback"),
        ("stop_atr", "stop_atr"),
        ("atr_len", "atr_len"),
        ("max_hold_bars", "max_hold_bars"),
        ("momentum_exit", "momentum_exit"),
    ],
    "VWAP_RECLAIM": [
        ("vwap_period", "vwap_period"),
        ("slope_lookback", "slope_lookback"),
        ("stop_atr", "stop_atr"),
        ("setup_lookback", "setup_lookback"),
        ("atr_len", "atr_len"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "EMA_ACCEL_BO": [
        ("trend_len", "trend_len"),
        ("slope_lookback", "slope_lookback"),
        ("accel_mult", "accel_mult"),
        ("stop_atr", "stop_atr"),
        ("atr_len", "atr_len"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "HHHL_STRUCT_BO": [
        ("swing_len", "swing_len"),
        ("break_buffer", "break_buffer"),
        ("stop_buffer", "stop_buffer"),
        ("atr_len", "atr_len"),
        ("slope_lookback", "slope_lookback"),
        ("range_mult", "range_mult"),
        ("min_structure_size", "min_structure_size"),
        ("atr_regime_exit", "atr_regime_exit"),
        ("atr_collapse_mult", "atr_collapse_mult"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "MA200_PULLBACK": [
        ("long_len", "long_len"),
        ("short_len", "short_len"),
        ("stop_atr", "stop_atr"),
        ("atr_len", "atr_len"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "INSIDE_BAR_CONT": [
        ("trend_len", "trend_len"),
        ("slope_lookback", "slope_lookback"),
        ("stop_atr", "stop_atr"),
        ("atr_len", "atr_len"),
        ("max_hold_bars", "max_hold_bars"),
    ],
    "SMC_SWEEP_OBFVG": [
        ("pivot_L", "pivot_L"),
        ("eq_tol_atr", "eq_tol_atr"),
        ("disp_atr", "disp_atr"),
        ("fvg_min_atr", "fvg_min_atr"),
        ("atr_period", "atr_period"),
        ("risk_per_trade", "risk_per_trade"),
        ("timeout_bars", "timeout_bars"),
    ],
}


def normalize_params(param_dict):
    return json.dumps(param_dict, sort_keys=True)


def safe_nested_get(data, path, default=0.0):
    current = data
    for key in path:
        if current is None:
            return default
        try:
            if key in current:
                current = current[key]
            else:
                return default
        except TypeError:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def annualized_return_pct(total_return_pct, start_dt, end_dt):
    days = max((end_dt - start_dt).days, 1)
    gross = 1.0 + (total_return_pct / 100.0)
    if gross <= 0:
        return -100.0
    return ((gross ** (365.25 / days)) - 1.0) * 100.0


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_or_init_results():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        print("Making results file")
        make_file()

    df = pd.read_parquet(RESULTS_PATH)
    if "backtest_version" not in df.columns:
        df["backtest_version"] = "legacy"
    if "risk_profile" not in df.columns:
        df["risk_profile"] = "legacy"
    return df


def build_interval_to_timeframe(interval_cfg):
    return {
        interval: BT_TIMEFRAME_MAP[name]
        for interval, name in interval_cfg.items()
        if name in BT_TIMEFRAME_MAP
    }


def validate_strategy_setup(params_cfg):
    for strategy_name in params_cfg.keys():
        if strategy_name not in ALL_STRATEGIES:
            raise KeyError(f"Strategy '{strategy_name}' not found in ALL_STRATEGIES.")
        if strategy_name not in STRATEGY_PARAM_SPECS:
            raise KeyError(
                f"Missing parameter spec mapping for strategy '{strategy_name}'."
            )


def build_param_sets(params_cfg):
    param_sets = {}
    for strategy_name, strategy_params in params_cfg.items():
        spec = STRATEGY_PARAM_SPECS[strategy_name]
        missing = [cfg_key for cfg_key, _ in spec if cfg_key not in strategy_params]
        if missing:
            raise KeyError(
                f"Strategy '{strategy_name}' missing config keys: {', '.join(missing)}"
            )

        values = [strategy_params[cfg_key] for cfg_key, _ in spec]
        param_sets[strategy_name] = [
            {runtime_key: combo[idx] for idx, (_, runtime_key) in enumerate(spec)}
            for combo in product(*values)
        ]
    return param_sets


def build_existing_run_keys(df):
    key_cols = ["strategy", "symbol", "interval", "parameters", "backtest_version", "risk_profile"]
    if df.empty:
        return set()

    for col in key_cols:
        if col not in df.columns:
            df[col] = "legacy"
    return set(tuple(row) for row in df[key_cols].itertuples(index=False, name=None))


def get_price_df(symbol, interval, cache):
    cache_key = (symbol, interval)
    if cache_key in cache:
        return cache[cache_key]

    data_path = DATA_PROCESSED / f"{symbol}_{interval}.csv"
    if not data_path.exists():
        cache[cache_key] = None
        return None

    price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    cache[cache_key] = price_df
    return price_df


def run_single_backtest(
    strategy_name,
    param_dict,
    price_df,
    commission,
    sizer,
    interval,
    interval_to_timeframe,
    risk_config,
):
    data_feed = bt.feeds.PandasData(dataname=price_df)
    run_fn = ALL_STRATEGIES[strategy_name]
    return run_fn(
        data=data_feed,
        commission_=commission,
        sizer=sizer,
        interval=interval,
        interval_to_timeframe=interval_to_timeframe,
        risk_config=risk_config,
        **param_dict,
    )


def extract_result_row(
    strategy_name,
    symbol,
    interval,
    param_key,
    result,
    price_df,
    commission,
    sizer,
    risk_profile,
):
    ta = result.analyzers.trades.get_analysis()
    sa = result.analyzers.sharpe.get_analysis()
    da = result.analyzers.drawdown.get_analysis()

    total_trades = int(safe_nested_get(ta, ["total", "closed"], 0.0))
    won_trades = int(safe_nested_get(ta, ["won", "total"], 0.0))
    win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0

    long_trades = int(safe_nested_get(ta, ["long", "total"], 0.0))
    short_trades = int(safe_nested_get(ta, ["short", "total"], 0.0))
    long_won = int(safe_nested_get(ta, ["long", "won"], 0.0))
    short_won = int(safe_nested_get(ta, ["short", "won"], 0.0))
    long_win_rate = (long_won / long_trades) if long_trades > 0 else 0.0
    short_win_rate = (short_won / short_trades) if short_trades > 0 else 0.0

    gross_profit = safe_nested_get(ta, ["won", "pnl", "total"], 0.0)
    gross_loss = safe_nested_get(ta, ["lost", "pnl", "total"], 0.0)
    net_profit = safe_nested_get(ta, ["pnl", "net", "total"], 0.0)
    avg_trade_pnl = safe_nested_get(ta, ["pnl", "net", "average"], 0.0)
    avg_win_pnl = safe_nested_get(ta, ["won", "pnl", "average"], 0.0)
    avg_loss_pnl = safe_nested_get(ta, ["lost", "pnl", "average"], 0.0)

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    max_drawdown_pct = safe_nested_get(da, ["max", "drawdown"], 0.0)
    max_moneydown = safe_nested_get(da, ["max", "moneydown"], 0.0)

    start_price = price_df["close"].iloc[0]
    end_price = price_df["close"].iloc[-1]
    market_gain = ((end_price - start_price) / start_price) * 100

    strategy_return = ((result.broker.getvalue() - STARTING_CASH) / STARTING_CASH) * 100
    excess_return = strategy_return - market_gain
    start_date = pd.Timestamp(bt.num2date(result.data.datetime.array[0]))
    end_date = pd.Timestamp(bt.num2date(result.data.datetime.array[-1]))
    annualized_return = annualized_return_pct(strategy_return, start_date, end_date)
    calmar = (annualized_return / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "interval": interval,
        "parameters": param_key,
        "return": strategy_return,
        "sharpe": sa.get("sharperatio", 0) if sa.get("sharperatio") is not None else 0,
        "trades": total_trades,
        "win_rate": win_rate,
        "start_date": start_date,
        "end_date": end_date,
        "commission": commission,
        "sizer": sizer,
        "market_gain": market_gain,
        "excess_return": excess_return,
        "annualized_return": annualized_return,
        "max_drawdown_pct": max_drawdown_pct,
        "max_moneydown": max_moneydown,
        "calmar": calmar,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_win_pnl": avg_win_pnl,
        "avg_loss_pnl": avg_loss_pnl,
        "profit_factor": profit_factor,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "backtest_version": BACKTEST_VERSION,
        "risk_profile": risk_profile,
    }


def run_backtests():
    df = load_or_init_results()
    config = load_config()
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Missing data directory: {DATA_PROCESSED}. Run utils/data_loader.py first."
        )

    commission = config["commission"]
    sizer = config["sizer"]
    risk_config = config.get("risk", {})
    risk_profile = json.dumps(risk_config, sort_keys=True)

    validate_strategy_setup(config["params"])
    param_sets = build_param_sets(config["params"])
    strategy_names = list(config["params"].keys())

    interval_to_timeframe = build_interval_to_timeframe(config["INTERVAL_TO_TIMEFRAME"])
    intervals = config["intervals"]
    all_symbols = (
        config["stocks"]
        + config["forex"]
        + config["indices"]
        + config.get("crypto", [])
    )

    existing_run_keys = build_existing_run_keys(df)
    all_new_results = []
    data_cache = {}

    for symbol in tqdm(all_symbols, desc="Backtesting Symbols"):
        for interval in intervals:
            price_df = get_price_df(symbol, interval, data_cache)
            if price_df is None:
                continue

            for strategy_name in strategy_names:
                combos = param_sets[strategy_name]
                desc = f"Running {strategy_name} on {symbol}, interval: {interval}"
                for param_dict in tqdm(combos, desc=desc, leave=False):
                    param_key = normalize_params(param_dict)
                    run_key = (
                        strategy_name,
                        symbol,
                        interval,
                        param_key,
                        BACKTEST_VERSION,
                        risk_profile,
                    )
                    if run_key in existing_run_keys:
                        continue

                    result = run_single_backtest(
                        strategy_name=strategy_name,
                        param_dict=param_dict,
                        price_df=price_df,
                        commission=commission,
                        sizer=sizer,
                        interval=interval,
                        interval_to_timeframe=interval_to_timeframe,
                        risk_config=risk_config,
                    )
                    row = extract_result_row(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        interval=interval,
                        param_key=param_key,
                        result=result,
                        price_df=price_df,
                        commission=commission,
                        sizer=sizer,
                        risk_profile=risk_profile,
                    )
                    all_new_results.append(row)
                    existing_run_keys.add(run_key)

    if all_new_results:
        new_results_df = pd.DataFrame(all_new_results)
        final_df = pd.concat([df, new_results_df], ignore_index=True)
        final_df.to_parquet(RESULTS_PATH, index=False)
        print(f"Done. Added {len(new_results_df)} new results.")
    else:
        print("Done. No new results to add.")


if __name__ == "__main__":
    run_backtests()

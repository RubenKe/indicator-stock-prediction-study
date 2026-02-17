import backtrader as bt
import datetime
import pandas as pd
from strategies import ALL_STRATEGIES
from utils.results_logger import make_file
import yaml
import os
from pathlib import Path
from itertools import product
import json
from tqdm import tqdm

# --- 1. Setup Paths ---
DATA_PROCESSED = Path("data") / "raw"
RESULTS_PATH = Path("database/results.parquet")
BACKTEST_VERSION = "long_short_v1"

# --- 2. Load or Create Results File ---
if not RESULTS_PATH.exists():
    print("Making results file")
    make_file()

# Load existing data to check for duplicates later
df = pd.read_parquet(RESULTS_PATH)
if "backtest_version" not in df.columns:
    df["backtest_version"] = "legacy"

# --- 3. Load Config ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

commission = config['commission']
sizer = config['sizer']
INTERVAL_TO_TIMEFRAME = config['INTERVAL_TO_TIMEFRAME']

intervals = config['intervals']
params = config['params']

# Setting configs
starting_cash = 1000
all_symbols = config['stocks'] + config['forex'] + config['indices']
strategy_names = list(params.keys())

# Prepare parameter combinations
param_sets = {
    "DMAC": list(product(
        params["DMAC"]["short_ema"],
        params["DMAC"]["long_ema"],
        params["DMAC"]["adx_period"],
        params["DMAC"]["adx_threshold"],
    )),
    "RSI_MA": list(product(
        params["RSI_MA"]["ma"],
        params["RSI_MA"]["buy_rsi"],
        params["RSI_MA"]["sell_rsi"],
    )),
    "BBANDS_MR": list(product(
        params["BBANDS_MR"]["bb_len"],
        params["BBANDS_MR"]["bb_k"],
        params["BBANDS_MR"]["rsi_os"],
        params["BBANDS_MR"]["adx_max"],
        params["BBANDS_MR"]["stop_atr"],
        params["BBANDS_MR"]["max_hold_bars"],
    )),
    "DONCHIAN_BO": list(product(
        params["DONCHIAN_BO"]["entry_len"],
        params["DONCHIAN_BO"]["exit_len"],
        params["DONCHIAN_BO"]["atr_len"],
        params["DONCHIAN_BO"]["entry_buffer_atr"],
        params["DONCHIAN_BO"]["stop_atr"],
        params["DONCHIAN_BO"]["trail_atr"],
        params["DONCHIAN_BO"]["max_hold_bars"],
        params["DONCHIAN_BO"]["cooldown_bars"],
    )),
    "RSI_PULLBACK": list(product(
        params["RSI_PULLBACK"]["trend_len"],
        params["RSI_PULLBACK"]["slope_lookback"],
        params["RSI_PULLBACK"]["rsi_len"],
        params["RSI_PULLBACK"]["rsi_pullback"],
        params["RSI_PULLBACK"]["rsi_recover"],
        params["RSI_PULLBACK"]["atr_len"],
        params["RSI_PULLBACK"]["stop_atr"],
        params["RSI_PULLBACK"]["max_hold_bars"],
    )),
    "ATR_VOL_BO": list(product(
        params["ATR_VOL_BO"]["atr_len"],
        params["ATR_VOL_BO"]["atr_expansion_mult"],
        params["ATR_VOL_BO"]["stop_atr"],
        params["ATR_VOL_BO"]["breakout_len"],
        params["ATR_VOL_BO"]["max_hold_bars"],
    )),
    "MA_TREND_CONT": list(product(
        params["MA_TREND_CONT"]["trend_len"],
        params["MA_TREND_CONT"]["slope_lookback"],
        params["MA_TREND_CONT"]["stop_atr"],
        params["MA_TREND_CONT"]["atr_len"],
        params["MA_TREND_CONT"]["max_hold_bars"],
        params["MA_TREND_CONT"]["momentum_exit"],
    )),
    "VWAP_RECLAIM": list(product(
        params["VWAP_RECLAIM"]["vwap_period"],
        params["VWAP_RECLAIM"]["slope_lookback"],
        params["VWAP_RECLAIM"]["stop_atr"],
        params["VWAP_RECLAIM"]["setup_lookback"],
        params["VWAP_RECLAIM"]["atr_len"],
        params["VWAP_RECLAIM"]["max_hold_bars"],
    )),
    "EMA_ACCEL_BO": list(product(
        params["EMA_ACCEL_BO"]["trend_len"],
        params["EMA_ACCEL_BO"]["slope_lookback"],
        params["EMA_ACCEL_BO"]["accel_mult"],
        params["EMA_ACCEL_BO"]["stop_atr"],
        params["EMA_ACCEL_BO"]["atr_len"],
        params["EMA_ACCEL_BO"]["max_hold_bars"],
    )),
    "HHHL_STRUCT_BO": list(product(
        params["HHHL_STRUCT_BO"]["swing_len"],
        params["HHHL_STRUCT_BO"]["break_buffer"],
        params["HHHL_STRUCT_BO"]["stop_buffer"],
        params["HHHL_STRUCT_BO"]["atr_len"],
        params["HHHL_STRUCT_BO"]["slope_lookback"],
        params["HHHL_STRUCT_BO"]["range_mult"],
        params["HHHL_STRUCT_BO"]["min_structure_size"],
        params["HHHL_STRUCT_BO"]["atr_regime_exit"],
        params["HHHL_STRUCT_BO"]["atr_collapse_mult"],
        params["HHHL_STRUCT_BO"]["max_hold_bars"],
    )),
    "MA200_PULLBACK": list(product(
        params["MA200_PULLBACK"]["long_len"],
        params["MA200_PULLBACK"]["short_len"],
        params["MA200_PULLBACK"]["stop_atr"],
        params["MA200_PULLBACK"]["atr_len"],
        params["MA200_PULLBACK"]["max_hold_bars"],
    )),
    "INSIDE_BAR_CONT": list(product(
        params["INSIDE_BAR_CONT"]["trend_len"],
        params["INSIDE_BAR_CONT"]["slope_lookback"],
        params["INSIDE_BAR_CONT"]["stop_atr"],
        params["INSIDE_BAR_CONT"]["atr_len"],
        params["INSIDE_BAR_CONT"]["max_hold_bars"],
    )),
    "SMC_SWEEP_OBFVG": list(product(
        params["SMC_SWEEP_OBFVG"]["pivot_L"],
        params["SMC_SWEEP_OBFVG"]["eq_tol_atr"],
        params["SMC_SWEEP_OBFVG"]["disp_atr"],
        params["SMC_SWEEP_OBFVG"]["fvg_min_atr"],
        params["SMC_SWEEP_OBFVG"]["atr_period"],
        params["SMC_SWEEP_OBFVG"]["risk_per_trade"],
        params["SMC_SWEEP_OBFVG"]["timeout_bars"],
    )),
}

BT_TIMEFRAME_MAP = {
    "Minutes": bt.TimeFrame.Minutes,
    "Days": bt.TimeFrame.Days,
    "Weeks": bt.TimeFrame.Weeks,
}

INTERVAL_TO_TIMEFRAME = {
    interval: BT_TIMEFRAME_MAP[name]
    for interval, name in config["INTERVAL_TO_TIMEFRAME"].items()
}

# --- 4. Helper Functions ---
def make_param_dict(strategy_name, param_tuple):
    if strategy_name == "DMAC":
        return {
            "pfast": param_tuple[0],
            "pslow": param_tuple[1],
            "adx_period": param_tuple[2],
            "adx_threshold": param_tuple[3],
        }
    elif strategy_name == "RSI_MA":
        return {
            "ma": param_tuple[0],
            "buy_rsi": param_tuple[1],
            "sell_rsi": param_tuple[2],
        }
    elif strategy_name == "BBANDS_MR":
        return {
            "bb_len": param_tuple[0],
            "bb_k": param_tuple[1],
            "rsi_os": param_tuple[2],
            "adx_max": param_tuple[3],
            "stop_atr": param_tuple[4],
            "max_hold_bars": param_tuple[5],
        }
    elif strategy_name == "DONCHIAN_BO":
        return {
            "entry_len": param_tuple[0],
            "exit_len": param_tuple[1],
            "atr_len": param_tuple[2],
            "entry_buffer_atr": param_tuple[3],
            "stop_atr": param_tuple[4],
            "trail_atr": param_tuple[5],
            "max_hold_bars": param_tuple[6],
            "cooldown_bars": param_tuple[7],
        }
    elif strategy_name == "RSI_PULLBACK":
        return {
            "trend_len": param_tuple[0],
            "slope_lookback": param_tuple[1],
            "rsi_len": param_tuple[2],
            "rsi_pullback": param_tuple[3],
            "rsi_recover": param_tuple[4],
            "atr_len": param_tuple[5],
            "stop_atr": param_tuple[6],
            "max_hold_bars": param_tuple[7],
        }
    elif strategy_name == "ATR_VOL_BO":
        return {
            "atr_len": param_tuple[0],
            "atr_expansion_mult": param_tuple[1],
            "stop_atr": param_tuple[2],
            "breakout_len": param_tuple[3],
            "max_hold_bars": param_tuple[4],
        }
    elif strategy_name == "MA_TREND_CONT":
        return {
            "trend_len": param_tuple[0],
            "slope_lookback": param_tuple[1],
            "stop_atr": param_tuple[2],
            "atr_len": param_tuple[3],
            "max_hold_bars": param_tuple[4],
            "momentum_exit": param_tuple[5],
        }
    elif strategy_name == "VWAP_RECLAIM":
        return {
            "vwap_period": param_tuple[0],
            "slope_lookback": param_tuple[1],
            "stop_atr": param_tuple[2],
            "setup_lookback": param_tuple[3],
            "atr_len": param_tuple[4],
            "max_hold_bars": param_tuple[5],
        }
    elif strategy_name == "EMA_ACCEL_BO":
        return {
            "trend_len": param_tuple[0],
            "slope_lookback": param_tuple[1],
            "accel_mult": param_tuple[2],
            "stop_atr": param_tuple[3],
            "atr_len": param_tuple[4],
            "max_hold_bars": param_tuple[5],
        }
    elif strategy_name == "HHHL_STRUCT_BO":
        return {
            "swing_len": param_tuple[0],
            "break_buffer": param_tuple[1],
            "stop_buffer": param_tuple[2],
            "atr_len": param_tuple[3],
            "slope_lookback": param_tuple[4],
            "range_mult": param_tuple[5],
            "min_structure_size": param_tuple[6],
            "atr_regime_exit": param_tuple[7],
            "atr_collapse_mult": param_tuple[8],
            "max_hold_bars": param_tuple[9],
        }
    elif strategy_name == "MA200_PULLBACK":
        return {
            "long_len": param_tuple[0],
            "short_len": param_tuple[1],
            "stop_atr": param_tuple[2],
            "atr_len": param_tuple[3],
            "max_hold_bars": param_tuple[4],
        }
    elif strategy_name == "INSIDE_BAR_CONT":
        return {
            "trend_len": param_tuple[0],
            "slope_lookback": param_tuple[1],
            "stop_atr": param_tuple[2],
            "atr_len": param_tuple[3],
            "max_hold_bars": param_tuple[4],
        }
    elif strategy_name == "SMC_SWEEP_OBFVG":
        return {
            "pivot_L": param_tuple[0],
            "eq_tol_atr": param_tuple[1],
            "disp_atr": param_tuple[2],
            "fvg_min_atr": param_tuple[3],
            "atr_period": param_tuple[4],
            "risk_per_trade": param_tuple[5],
            "timeout_bars": param_tuple[6],
        }
    return {}

def normalize_params(param_dict):
    return json.dumps(param_dict, sort_keys=True)

# --- 5. Main Loop ---
all_new_results = []

for symbol in tqdm(all_symbols, desc="Backtesting Symbols"):
    for interval in intervals:
        for strategy_name in strategy_names:
            for param_tuple in tqdm(param_sets[strategy_name], desc=f"Running {strategy_name} on {symbol}, interval: {interval} "):

                # Prepare parameters
                param_dict = make_param_dict(strategy_name, param_tuple)
                param_key = normalize_params(param_dict)

                # Check if this specific run already exists in the loaded DataFrame
                if not df.empty:
                    exists = (
                        (df["strategy"] == strategy_name) &
                        (df["symbol"] == symbol) &
                        (df["interval"] == interval) &
                        (df["parameters"] == param_key) &
                        (df["backtest_version"] == BACKTEST_VERSION)
                    ).any()

                    if exists:
                        continue

                # Load data
                data_path = DATA_PROCESSED / f"{symbol}_{interval}.csv"
                if not data_path.exists():
                    continue 

                price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                data_feed = bt.feeds.PandasData(dataname=price_df)

                # Run Strategy
                run_fn = ALL_STRATEGIES[strategy_name]
                res = run_fn(
                    data=data_feed,
                    commission_=commission,
                    sizer=sizer,
                    interval=interval,
                    interval_to_timeframe = INTERVAL_TO_TIMEFRAME,
                    **param_dict)
                
                # --- ANALYZER EXTRACTION ---
                # 1. Get the analyzer objects safely
                ta = res.analyzers.trades.get_analysis()
                sa = res.analyzers.sharpe.get_analysis()
                
                # 2. Extract Trade Counts 
                total_trades = ta.total.total if 'total' in ta else 0
                won_trades = ta.won.total if 'won' in ta else 0 
                win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0

                # 3. calulate market gain
                start_price = price_df['close'].iloc[0]
                end_price = price_df['close'].iloc[-1]
                market_gain = ((end_price - start_price)/ start_price) *100

                # 4. calculate stratey gain and excess return
                strategy_return = ((res.broker.getvalue() - starting_cash) / starting_cash) * 100
                excess_return = strategy_return - market_gain
                new_result = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "interval": interval,
                    "parameters": param_key,
                    "return": strategy_return,                    
                    "sharpe": sa.get('sharperatio', 0) if sa.get('sharperatio') is not None else 0,                    
                    "trades": total_trades,
                    "win_rate": win_rate,
                    "start_date": pd.Timestamp(bt.num2date(res.data.datetime.array[0])),
                    "end_date": pd.Timestamp(bt.num2date(res.data.datetime.array[-1])),
                    "commission": commission,
                    "sizer": sizer,
                    "market_gain": market_gain,
                    "excess_return": excess_return,
                    "backtest_version": BACKTEST_VERSION,
                }

                all_new_results.append(new_result)

# --- 6. Save Results ---
if all_new_results:
    # Convert new results to a DataFrame
    new_results_df = pd.DataFrame(all_new_results)
    
    # Concatenate with the original df (so we don't delete old history)
    final_df = pd.concat([df, new_results_df], ignore_index=True)
    
    # Save back to Parquet
    final_df.to_parquet(RESULTS_PATH, index=False)
    print(f"Done. Added {len(new_results_df)} new results.")
else:
    print("Done. No new results to add.")

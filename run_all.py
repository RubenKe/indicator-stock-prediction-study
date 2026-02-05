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

# --- 2. Load or Create Results File ---
if not RESULTS_PATH.exists():
    print("Making results file")
    make_file()

# Load existing data to check for duplicates later
df = pd.read_parquet(RESULTS_PATH)

# --- 3. Load Config ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

commission = config['commission']
sizer = config['sizer']
INTERVAL_TO_TIMEFRAME = config['INTERVAL_TO_TIMEFRAME']

stocks = config['stocks'] # Unused variable in your original snippet, but fine to keep if needed
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
    "RSI_MA": list(product(params["RSI_MA"]["ma"], params["RSI_MA"]["buy_rsi"], params["RSI_MA"]["sell_rsi"]))
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
    return {}

def normalize_params(param_dict):
    return json.dumps(param_dict, sort_keys=True)

# --- 5. Main Loop ---
all_new_results = []

for symbol in tqdm(all_symbols, desc="Backtesting Symbols"):
    for interval in intervals:
        for strategy_name in strategy_names:
            for param_tuple in param_sets[strategy_name]:

                # Prepare parameters
                param_dict = make_param_dict(strategy_name, param_tuple)
                param_key = normalize_params(param_dict)

                # Check if this specific run already exists in the loaded DataFrame
                if not df.empty:
                    exists = (
                        (df["strategy"] == strategy_name) &
                        (df["symbol"] == symbol) &
                        (df["interval"] == interval) &
                        (df["parameters"] == param_key)
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
                    "excess_return": excess_return
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

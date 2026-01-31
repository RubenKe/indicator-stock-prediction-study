import backtrader as bt
import datetime
import pandas as pd
from strategies import ALL_STRATEGIES
import yaml
import os
from pathlib import Path
from itertools import product
import json
from tqdm import tqdm

# make the path with data
DATA_PROCESSED = Path("data") / "raw"
RESULTS_PATH = Path("database/results.parquet")

# Load the results df
if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
    df = pd.read_parquet(RESULTS_PATH)
else:
    print("File could not be loaded")


# Open the config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load config
commission = config['commission']
sizer = config['sizer']
stocks = config['stocks']
forex = config['forex']
indices = config['indices']
intervals = config['intervals']
params = config['params']

# Setting configs
starting_cash = 1000
all_symbols = stocks + forex + indices
strategy_names = list(params.keys())
param_sets = {
    "DMAC": list(product(params["DMAC"]["short_ema"], params["DMAC"]["long_ema"])),
    "RSI_MA": list(product(params["RSI_MA"]["ma"], params["RSI_MA"]["buy_rsi"], params["RSI_MA"]["sell_rsi"]))
}

# Normalize functions
def make_param_dict(strategy_name, param_tuple):
    if strategy_name == "DMAC":
        return {
            "pfast": param_tuple[0],
            "pslow": param_tuple[1],
        }
    elif strategy_name == "RSI_MA":
        return {
            "ma": param_tuple[0],
            "buy_rsi": param_tuple[1],
            "sell_rsi": param_tuple[2],
        }

def normalize_params(param_dict):
    return json.dumps(param_dict, sort_keys=True)


# Main loop
all_new_results = []
for symbol in tqdm(all_symbols, desc="Backtesting Symbols"):
    for interval in intervals:
        for strategy_name in strategy_names:
            for param_tuple in param_sets[strategy_name]:

                param_dict = make_param_dict(strategy_name, param_tuple)
                param_key = normalize_params(param_dict)

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
                price_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                data_feed = bt.feeds.PandasData(dataname=price_df)

                run_fn = ALL_STRATEGIES[strategy_name]
                res = run_fn(
                    data=data_feed,
                    commission_=commission,
                    sizer=sizer,
                    **param_dict)
                
                # 1. Get the final total value (Cash + Stocks)
                final_value = res.broker.getvalue()

                # 2. Get the remaining uninvested cash
                final_cash = res.broker.get_cash()

                # 3. Get the total trades and win rate in a safe way
                total_trades = res.total.total if 'total' in res else 0
                won_trades = res.won.total if 'won' in res else 0     


                new_result = {
                "strategy": strategy_name,
                "symbol": symbol,
                "interval": interval,
                "parameters": param_key,
                "return": ((res.broker.getvalue() - starting_cash) / starting_cash) * 100,
                "sharpe": res.analyzers.sharpe.get_analysis().get('sharperatio', 0),
                "trades": res.analyzers.trades.get_analysis().total.total,
                "win_rate": won_trades / total_trades if total_trades > 0 else 0,
                "start_date": pd.Timestamp(bt.num2date(res.data.datetime.array[0])),
                "end_date": pd.Timestamp(bt.num2date(res.data.datetime.array[-1])),
                "commission": commission,
                "sizer": sizer
                }

                all_new_results.append(new_result)

df = pd.DataFrame(all_new_results) # Add all new rows at once 
df.to_parquet(RESULTS_PATH, index=False) # Save the results again



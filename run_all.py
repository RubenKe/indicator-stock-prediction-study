from tkinter import ALL
import backtrader as bt
import datetime
import pandas as pd
from strategies import ALL_STRATEGIES
import yaml
import os
from pathlib import Path
from itertools import product


# make the path with data
DATA_PROCESSED = Path("data") / "raw"

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
all_symbols = stocks + forex + indices
strategy_names = list(params.keys())
param_sets = {
    "DMAC": list(product(params["DMAC"]["short_ema"], params["DMAC"]["long_ema"])),
    "RSI_MA": list(product(params["RSI_MA"]["ma"], params["RSI_MA"]["buy_rsi"], params["RSI_MA"]["sell_rsi"]))
}



"""
# Main loop
for symbol in all_symbols:
    for interval in intervals:


df = pd.read_csv(DATA_PROCESSED / f'{symbol}_{interval}.csv', index_col=0, parse_dates=True)
data = bt.feeds.PandasData(dataname=df)
strat = ALL_STRATEGIES['run_dmac'](data, commission, sizer)
"""

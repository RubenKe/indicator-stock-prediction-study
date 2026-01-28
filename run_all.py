import backtrader as bt
import datetime
import pandas as pd
from strategies import ALL_STRATEGIES
import yaml
import os
from pathlib import Path

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



df = pd.read_csv(DATA_PROCESSED / "AAPL_1d.csv", index_col=0, parse_dates=True)
data = bt.feeds.PandasData(dataname=df)

strat = ALL_STRATEGIES['run_dmac'](data, commission, sizer)
print(strat.broker.getvalue())

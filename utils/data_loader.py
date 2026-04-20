import yfinance as yf
from pathlib import Path
import shutil
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_PROCESSED = DATA_ROOT / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


stocks_pairs = config.get("stocks") or []
forex_pairs = config.get("forex") or []
index_pairs = config.get("indices") or []
crypto_pairs = config.get("crypto") or []
benchmark_symbol = config.get("benchmark_symbol")

intervals = config["intervals"]
# # periods = config["periods"]  # Not used anymore, using start/end for more data  # Not used anymore, using start/end for more data
max_candles = int(config.get("max_candles", 5000))

def download_pair(pair: str, interval: str) -> pd.DataFrame:
    # Use start and end to get more historical data, yfinance period limits intraday
    data = yf.download(pair, start='2015-01-01', end=None, interval=interval, progress=False)

    # This flattens the columns so headers are just 'Open', 'High', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.columns = data.columns.str.lower()

    
    # Keep the most recent N rows so all df are equal in length (crypto may have fewer).
    return data.tail(max_candles)

# Cleanup and directory creation
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
for raw_path in list(DATA_PROCESSED.glob("*.csv")) + list(DATA_PROCESSED.glob("*.parquet")):
    raw_path.unlink(missing_ok=True)

all_pairs = []
for group in (stocks_pairs, forex_pairs, index_pairs, crypto_pairs):
    for pair in group:
        if pair not in all_pairs:
            all_pairs.append(pair)
if benchmark_symbol and benchmark_symbol not in all_pairs:
    all_pairs.append(benchmark_symbol)

# Main loop
for pair in all_pairs:
    for interval in intervals:

        # Save with index=True to keep the proper Datetime Index
        df = download_pair(pair, interval)
        df.to_parquet(DATA_PROCESSED / f'{pair}_{interval}.parquet', index=True)

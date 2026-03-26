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


stocks_pairs = config["stocks"]
forex_pairs = config["forex"]
index_pairs = config["indices"]
crypto_pairs = config.get("crypto", [])
benchmark_symbol = config.get("benchmark_symbol")

intervals = config["intervals"]
periods = config["periods"]

def download_pair(pair: str, interval: str) -> pd.DataFrame:
    data = yf.download(pair, interval=interval, period=periods[interval], progress=False)

    # This flattens the columns so headers are just 'Open', 'High', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.columns = data.columns.str.lower()

    
    return data.tail(5000) # Keep the most recent 5000 rows so all df are equal in length

# Cleanup and directory creation
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
for csv_path in DATA_PROCESSED.glob("*.csv"):
    csv_path.unlink(missing_ok=True)

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
        # We use index_label='Date' to ensure the first column is named correctly in CSV
        df = download_pair(pair, interval)
        df.to_csv(DATA_PROCESSED / f'{pair}_{interval}.csv', index=True)

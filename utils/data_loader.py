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
periods = config.get("periods", {})
max_candles = int(config.get("max_candles", 5000))
max_candles_by_interval = config.get("max_candles_by_interval", {})


def _unique_in_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _candles_for_interval(interval: str) -> int:
    raw_value = max_candles_by_interval.get(interval, max_candles)
    return int(raw_value)

def download_pair(pair: str, interval: str) -> pd.DataFrame:
    # Use interval-aware periods to satisfy Yahoo limits for intraday data.
    period = periods.get(interval)
    if period:
        data = yf.download(pair, period=period, interval=interval, progress=False)
    else:
        data = yf.download(pair, start="2015-01-01", end=None, interval=interval, progress=False)

    # This flattens the columns so headers are just 'Open', 'High', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.columns = data.columns.str.lower()

    
    # Keep the most recent N rows per interval (crypto may have fewer from source limits).
    return data.tail(_candles_for_interval(interval))

# Cleanup and directory creation
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
for raw_path in list(DATA_PROCESSED.glob("*.csv")) + list(DATA_PROCESSED.glob("*.parquet")):
    raw_path.unlink(missing_ok=True)

backtest_cfg = config.get("backtest", {})
ml_cfg = config.get("ml", {})

base_pairs = stocks_pairs + forex_pairs + index_pairs
backtest_pairs = list(base_pairs)
if bool(backtest_cfg.get("include_crypto", True)):
    backtest_pairs += crypto_pairs

ml_pairs = list(base_pairs)
if bool(ml_cfg.get("include_crypto", True)):
    ml_pairs += crypto_pairs

all_pairs = _unique_in_order(backtest_pairs + ml_pairs)
if benchmark_symbol and benchmark_symbol not in all_pairs:
    all_pairs.append(benchmark_symbol)

backtest_intervals = backtest_cfg.get("intervals", intervals)
ml_intervals = ml_cfg.get("intervals", intervals)
download_intervals = _unique_in_order(list(backtest_intervals) + list(ml_intervals))

# Main loop
for pair in all_pairs:
    for interval in download_intervals:

        # Save with index=True to keep the proper Datetime Index
        df = download_pair(pair, interval)
        df.to_parquet(DATA_PROCESSED / f'{pair}_{interval}.parquet', index=True)

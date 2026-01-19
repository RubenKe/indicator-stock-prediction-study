import yfinance as yf
from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "raw"

Stocks_pairs = ['AAPL', 'MSFT', 'AMZN']
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X']
index_pairs = ['^GSPC', 'GLD', 'USO']

intervals = ['1d', '1h', '15m']
periods = { '1d':'5y',
            '1h':'1y',
            '15m':'60d' }

def download_pair(pair, interval):
    data = yf.download(pair, interval=interval, period=periods[interval], progress=False)

    # 2. Fix the "Row 0" / MultiIndex issue
    # This flattens the columns so headers are just 'Open', 'High', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    
    return data.tail(1200) # Keep the most recent 1200 rows so all df are equal in lenght

# Cleanup and directory creation
if DATA_PROCESSED.exists():
    shutil.rmtree(DATA_PROCESSED)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Main loop
for pair in (Stocks_pairs + forex_pairs + index_pairs):
    for interval in intervals:

        # Save with index=True to keep the proper Datetime Index
        # We use index_label='Date' to ensure the first column is named correctly in CSV
        df = download_pair(pair, interval)
        df.to_csv(DATA_PROCESSED / f'{pair}_{interval}.csv', index=True)

import yfinance as yf
import fastparquet
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

Stocks_pairs = ['AAPL', 'MSFT', 'AMZN']
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X']
index_pairs = ['^GSPC', 'GLD', 'USO']

intervals = ['1d', '1h', '15m']
periods = { '1d':'5y',
            '1h':'1y',
            '15m':'60d' }


def download_pair(pair, interval):
    data = yf.download(pair, interval=interval, period=periods[interval], progress=False)
    data.index.name = 'datetime'

    return data


for pair in (Stocks_pairs + forex_pairs + index_pairs):
    for interval in intervals:

        df = download_pair(pair, interval)
        df.to_parquet(path = f'{DATA_PROCESSED}\{pair}_{interval}')
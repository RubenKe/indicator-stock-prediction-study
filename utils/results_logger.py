import pandas as pd
from pathlib import Path

def make_file(): 
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SAVE_PATH = PROJECT_ROOT / "database"

    RESULT_COLUMNS = [
        "strategy",           # e.g. dmac
        "symbol",             # AAPL, EURUSD=X
        "interval",           # 15m, 1h, 1d
        "parameters",         # dict of strategy parameters
        "return",             # return in % (15% meaning 15% was made)
        "sharpe",             # sharpe ratio
        "trades",             # total number of trades
        "win_rate",           # percentage of winning trades
        "start_date",         # pd.Timestamp
        "end_date",           # pd.Timestamp
        "commission",         # commission in % per trade
        "sizer",              # % of capital used per trade
    ]

    results_df = pd.DataFrame(columns=RESULT_COLUMNS)
    results_df.to_parquet(SAVE_PATH / 'results.parquet', index=True)

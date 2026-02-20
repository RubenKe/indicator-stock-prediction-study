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
        "market_gain",        # % of market grown during trading period
        "excess_return",      # % difference of return to market gain
        "annualized_return",  # annualized strategy return in %
        "max_drawdown_pct",   # maximum drawdown in %
        "max_moneydown",      # maximum monetary drawdown
        "calmar",             # annualized_return / max_drawdown_pct
        "gross_profit",       # total pnl from winning trades
        "gross_loss",         # total pnl from losing trades (negative)
        "net_profit",         # total pnl from all closed trades
        "avg_trade_pnl",      # average pnl per closed trade
        "avg_win_pnl",        # average pnl per winning trade
        "avg_loss_pnl",       # average pnl per losing trade
        "profit_factor",      # gross_profit / abs(gross_loss)
        "long_trades",        # closed long trades count
        "short_trades",       # closed short trades count
        "long_win_rate",      # winning long trades / long trades
        "short_win_rate",     # winning short trades / short trades
        "backtest_version",   # strategy logic/version tag
        "risk_profile",       # risk config used for sizing
    ]

    results_df = pd.DataFrame(columns=RESULT_COLUMNS)
    results_df.to_parquet(SAVE_PATH / 'results.parquet', index=True)

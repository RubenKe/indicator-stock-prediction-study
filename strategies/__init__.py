import backtrader as bt
import pandas as pd
from .dmac import run as dmac
from .rsi_ma import run as rsi_ma

ALL_STRATEGIES = {
    "run_DMAC": dmac,
    "run_RSI_MA": rsi_ma
}
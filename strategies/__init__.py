import backtrader as bt
import pandas as pd
from .dmac_strat import run as dmac
from .rsi_ma import run as rsi_ma

ALL_STRATEGIES = {
    "run_dmac": dmac,
    "run_rsi_ma": rsi_ma
}
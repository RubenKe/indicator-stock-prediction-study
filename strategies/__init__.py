import backtrader as bt
import pandas as pd
from .dmac_strat import run as dmac  

ALL_STRATEGIES = {
    "DMAC": dmac
}
from .dmac import run as run_DMAC
from .rsi_ma import run as run_RSI_MA

ALL_STRATEGIES = {
    "DMAC": run_DMAC,
    "RSI_MA": run_RSI_MA
}
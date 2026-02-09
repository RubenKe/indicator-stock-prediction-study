from .dmac import run as run_DMAC
from .rsi_ma import run as run_RSI_MA
from .bbands_mr import run as run_BBANDS_MR

ALL_STRATEGIES = {
    "DMAC": run_DMAC,
    "RSI_MA": run_RSI_MA,
    "BBANDS_MR": run_BBANDS_MR,
}

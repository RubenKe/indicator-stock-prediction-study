from .dmac import run as run_DMAC
from .rsi_ma import run as run_RSI_MA
from .bbands_mr import run as run_BBANDS_MR
from .donchian_breakout import run as run_DONCHIAN_BO
from .rsi_pullback import run as run_RSI_PULLBACK
from .atr_volatility_breakout import run as run_ATR_VOL_BO

ALL_STRATEGIES = {
    "DMAC": run_DMAC,
    "RSI_MA": run_RSI_MA,
    "BBANDS_MR": run_BBANDS_MR,
    "DONCHIAN_BO": run_DONCHIAN_BO,
    "RSI_PULLBACK": run_RSI_PULLBACK,
    "ATR_VOL_BO": run_ATR_VOL_BO,
}

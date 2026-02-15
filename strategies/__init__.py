from .dmac import run as run_DMAC
from .rsi_ma import run as run_RSI_MA
from .bbands_mr import run as run_BBANDS_MR
from .donchian_breakout import run as run_DONCHIAN_BO
from .rsi_pullback import run as run_RSI_PULLBACK
from .atr_volatility_breakout import run as run_ATR_VOL_BO
from .ma_trend_continuation import run as run_MA_TREND_CONT
from .vwap_trend_reclaim import run as run_VWAP_RECLAIM
from .ema_acceleration_breakout import run as run_EMA_ACCEL_BO
from .hhhl_structure_breakout import run as run_HHHL_STRUCT_BO
from .ma200_pullback_bounce import run as run_MA200_PULLBACK
from .inside_bar_breakout_continuation import run as run_INSIDE_BAR_CONT
from .smc_structure_sweep_ob_fvg import run as run_SMC_SWEEP_OBFVG

ALL_STRATEGIES = {
    "DMAC": run_DMAC,
    "RSI_MA": run_RSI_MA,
    "BBANDS_MR": run_BBANDS_MR,
    "DONCHIAN_BO": run_DONCHIAN_BO,
    "RSI_PULLBACK": run_RSI_PULLBACK,
    "ATR_VOL_BO": run_ATR_VOL_BO,
    "MA_TREND_CONT": run_MA_TREND_CONT,
    "VWAP_RECLAIM": run_VWAP_RECLAIM,
    "EMA_ACCEL_BO": run_EMA_ACCEL_BO,
    "HHHL_STRUCT_BO": run_HHHL_STRUCT_BO,
    "MA200_PULLBACK": run_MA200_PULLBACK,
    "INSIDE_BAR_CONT": run_INSIDE_BAR_CONT,
    "SMC_SWEEP_OBFVG": run_SMC_SWEEP_OBFVG,
}

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = ["open", "high", "low", "close", "volume"]

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "close_ma20_ratio",
    "close_ma50_ratio",
    "ma20_slope",
    "ma50_slope",
    "rolling_std_10",
    "atr_close",
    "rsi_14",
    "roc_10",
    "intrabar_pos",
    "breakout_dist_20",
]


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle flat sections and one-sided windows deterministically.
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.mask(avg_loss == 0.0, 100.0)
    rsi = rsi.mask(avg_gain == 0.0, 0.0)
    rsi = rsi.mask(both_zero, 50.0)
    return rsi


def _validate_and_cast_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {c.lower(): c for c in raw_df.columns}
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in lower_cols]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    out = pd.DataFrame(index=raw_df.index)
    for col in REQUIRED_RAW_COLUMNS:
        src = lower_cols[col]
        out[col] = pd.to_numeric(raw_df[src], errors="coerce")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _validate_and_cast_raw(raw_df)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    atr14 = _compute_atr(df, period=14)
    rsi14 = _compute_rsi(close, period=14)
    recent_high_20 = high.shift(1).rolling(20, min_periods=20).max()

    price_range = (high - low).replace(0.0, np.nan)

    out = df.copy()
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["close_ma20_ratio"] = close / ma20
    out["close_ma50_ratio"] = close / ma50
    out["ma20_slope"] = ma20.pct_change(5)
    out["ma50_slope"] = ma50.pct_change(5)
    out["rolling_std_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    out["atr_close"] = atr14 / close.replace(0.0, np.nan)
    out["rsi_14"] = rsi14
    out["roc_10"] = close.pct_change(10)
    out["intrabar_pos"] = (close - low) / price_range
    out["breakout_dist_20"] = (
        close - recent_high_20
    ) / recent_high_20.replace(0.0, np.nan)

    out["next_return"] = close.shift(-1) / close - 1.0
    out["y"] = (out["next_return"] > 0.0).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_feature_frame(
    raw_df: pd.DataFrame,
    dataset_id: str,
    symbol: str,
    interval: str,
    test_candles: int,
) -> pd.DataFrame:
    featured = engineer_features(raw_df)
    required = REQUIRED_RAW_COLUMNS + FEATURE_COLUMNS + ["next_return", "y"]
    featured = featured.dropna(subset=required)

    if len(featured) < test_candles:
        raise ValueError(
            f"Dataset '{dataset_id}' has {len(featured)} rows after feature engineering, "
            f"fewer than required test_candles={test_candles}."
        )

    featured = featured.tail(test_candles).copy()
    featured["dataset_id"] = dataset_id
    featured["symbol"] = symbol
    featured["interval"] = interval
    featured["y"] = featured["y"].astype(int)
    featured.index.name = "timestamp"

    ordered_cols = (
        ["dataset_id", "symbol", "interval"]
        + REQUIRED_RAW_COLUMNS
        + FEATURE_COLUMNS
        + ["next_return", "y"]
    )
    return featured[ordered_cols]

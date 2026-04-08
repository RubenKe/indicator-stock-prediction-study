import json
from typing import Any, Dict, Optional

import pandas as pd

from run_all import (
    STARTING_CASH,
    build_interval_to_timeframe,
    extract_result_row,
    get_price_df,
    load_config,
    normalize_params,
    run_single_backtest,
)


def _equity_curve_from_timereturn(result) -> pd.DataFrame:
    if not hasattr(result.analyzers, "timereturn"):
        raise RuntimeError(
            "TimeReturn analyzer missing. Make sure strategies add "
            "bt.analyzers.TimeReturn with _name='timereturn'."
        )
    returns = result.analyzers.timereturn.get_analysis()
    returns = pd.Series(returns)
    returns.index = pd.to_datetime(returns.index)
    equity = (1.0 + returns).cumprod() * STARTING_CASH
    equity_df = pd.DataFrame({"equity": equity, "returns": returns})
    return equity_df


def run_single_strategy(
    strategy_name: str,
    symbol: str,
    interval: str,
    param_dict: Dict[str, Any],
    commission: Optional[float] = None,
    slippage: Optional[float] = None,
    sizer: Optional[float] = None,
    risk_config: Optional[Dict[str, Any]] = None,
    benchmark_symbol: Optional[str] = None,
):
    config = load_config()

    commission = float(commission if commission is not None else config["commission"])
    slippage = float(slippage if slippage is not None else config.get("slippage", 0.0))
    sizer = float(sizer if sizer is not None else config.get("sizer", 100.0))

    if risk_config is None:
        risk_config = dict(config.get("risk", {}))
    else:
        risk_config = dict(risk_config)
    risk_config["sizer_pct"] = sizer
    risk_config["max_position_value_pct"] = sizer / 100.0
    risk_profile = json.dumps(risk_config, sort_keys=True)

    if benchmark_symbol is None:
        benchmark_symbol = config.get("benchmark_symbol", "^GSPC")

    interval_to_timeframe = build_interval_to_timeframe(config["INTERVAL_TO_TIMEFRAME"])

    data_cache = {}
    price_df = get_price_df(symbol, interval, data_cache)
    if price_df is None:
        raise FileNotFoundError(f"Missing data for {symbol} {interval} in data/raw")
    benchmark_df = get_price_df(benchmark_symbol, interval, data_cache)

    result = run_single_backtest(
        strategy_name=strategy_name,
        param_dict=param_dict,
        price_df=price_df,
        commission=commission,
        slippage=slippage,
        sizer=sizer,
        interval=interval,
        interval_to_timeframe=interval_to_timeframe,
        risk_config=risk_config,
    )

    param_key = normalize_params(param_dict)
    summary = extract_result_row(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        param_key=param_key,
        result=result,
        price_df=price_df,
        commission=commission,
        slippage=slippage,
        sizer=sizer,
        risk_profile=risk_profile,
        benchmark_symbol=benchmark_symbol,
        benchmark_df=benchmark_df,
    )

    equity_curve = _equity_curve_from_timereturn(result)

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "price_df": price_df,
        "result": result,
    }

import backtrader as bt


DEFAULT_RISK_CONFIG = {
    "risk_per_trade": 0.005,
    "max_position_value_pct": 0.50,
    "drawdown_step_1": 0.10,
    "drawdown_step_2": 0.20,
    "drawdown_mult_1": 0.50,
    "drawdown_mult_2": 0.25,
    "min_risk_per_trade": 0.001,
    "min_stop_distance": 1e-8,
    "min_trade_size": 1e-6,
    "default_stop_atr": 2.0,
    "atr_len_for_sizing": 14,
}


class RiskManagedMixin:
    def _init_risk(self, base_risk_per_trade=None):
        cfg = dict(DEFAULT_RISK_CONFIG)
        user_cfg = getattr(self.p, "risk_config", None) or {}
        cfg.update(user_cfg)
        if base_risk_per_trade is not None:
            cfg["risk_per_trade"] = float(base_risk_per_trade)
        self._risk_cfg = cfg
        self._risk_peak_equity = float(self.broker.getvalue())

    def _risk_update_peak_equity(self):
        equity = float(self.broker.getvalue())
        self._risk_peak_equity = max(self._risk_peak_equity, equity)
        return equity

    def _effective_risk_fraction(self):
        equity = self._risk_update_peak_equity()
        peak = max(self._risk_peak_equity, 1e-12)
        drawdown = max(0.0, (peak - equity) / peak)

        risk_frac = max(float(self._risk_cfg["risk_per_trade"]), 0.0)
        if drawdown >= float(self._risk_cfg["drawdown_step_2"]):
            risk_frac *= float(self._risk_cfg["drawdown_mult_2"])
        elif drawdown >= float(self._risk_cfg["drawdown_step_1"]):
            risk_frac *= float(self._risk_cfg["drawdown_mult_1"])

        risk_frac = max(risk_frac, float(self._risk_cfg["min_risk_per_trade"]))
        return risk_frac, equity

    def _risk_size(self, entry_price, stop_price):
        stop_distance = abs(float(entry_price) - float(stop_price))
        min_stop_distance = float(self._risk_cfg["min_stop_distance"])
        if stop_distance <= min_stop_distance:
            return 0.0

        risk_frac, equity = self._effective_risk_fraction()
        risk_amount = equity * risk_frac
        size_by_risk = risk_amount / stop_distance

        max_position_value = equity * float(self._risk_cfg["max_position_value_pct"])
        size_by_value = max_position_value / max(abs(float(entry_price)), min_stop_distance)

        size = min(size_by_risk, size_by_value)
        if size < float(self._risk_cfg["min_trade_size"]):
            return 0.0
        return float(size)

    def _risk_buy(self, stop_price, entry_price=None, **kwargs):
        price = float(self.data.close[0]) if entry_price is None else float(entry_price)
        size = self._risk_size(price, stop_price)
        if size <= 0:
            return None

        order_kwargs = dict(kwargs)
        order_kwargs["size"] = size
        if entry_price is not None and "price" not in order_kwargs:
            order_kwargs["price"] = price
        return self.buy(**order_kwargs)

    def _risk_sell(self, stop_price, entry_price=None, **kwargs):
        price = float(self.data.close[0]) if entry_price is None else float(entry_price)
        size = self._risk_size(price, stop_price)
        if size <= 0:
            return None

        order_kwargs = dict(kwargs)
        order_kwargs["size"] = size
        if entry_price is not None and "price" not in order_kwargs:
            order_kwargs["price"] = price
        return self.sell(**order_kwargs)

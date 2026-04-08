import backtrader as bt
from .risk_managed import RiskManagedMixin


class EMAAccelerationBreakout(RiskManagedMixin, bt.Strategy):
    params = dict(
        trend_len=50,
        slope_lookback=5,
        accel_mult=1.0,
        atr_len=14,
        stop_atr=2.0,
        max_hold_bars=60,
        risk_config=None,
    )

    def __init__(self):
        self._init_risk()
        self.trend_ema = bt.ind.EMA(self.data.close, period=self.p.trend_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _reset_state(self):
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _trend_direction(self):
        if len(self) <= self.p.slope_lookback:
            return 0
        if (
            self.data.close[0] > self.trend_ema[0]
            and self.trend_ema[0] > self.trend_ema[-self.p.slope_lookback]
        ):
            return 1
        if (
            self.data.close[0] < self.trend_ema[0]
            and self.trend_ema[0] < self.trend_ema[-self.p.slope_lookback]
        ):
            return -1
        return 0

    def _accel_long_active(self, idx=0):
        distance = self.data.close[idx] - self.trend_ema[idx]
        return distance > self.p.accel_mult * self.atr[idx]

    def _accel_short_active(self, idx=0):
        distance = self.trend_ema[idx] - self.data.close[idx]
        return distance > self.p.accel_mult * self.atr[idx]

    def next(self):
        min_bars = max(self.p.trend_len, self.p.atr_len, self.p.slope_lookback + 1, 2)
        if len(self.data) <= min_bars:
            return

        # Manage open position
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.data.close[0] < self.trend_ema[0]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return
        elif self.position.size < 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] >= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.data.close[0] > self.trend_ema[0]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat: enter on fresh acceleration signal inside trend
        if self.position.size == 0:
            trend_direction = self._trend_direction()
            if trend_direction == 1:
                accel_now = self._accel_long_active(0)
                accel_prev = self._accel_long_active(-1)
                if accel_now and not accel_prev:
                    entry_price = self.data.close[0]
                    entry_atr = self.atr[0]
                    stop_price = entry_price - self.p.stop_atr * entry_atr
                    order = self._risk_buy(stop_price=stop_price, entry_price=entry_price)
                    if order is not None:
                        self.entry_price = entry_price
                        self.entry_atr = entry_atr
                        self.stop_price = stop_price
                        self.bars_in_trade = 0
                        return
            elif trend_direction == -1:
                accel_now = self._accel_short_active(0)
                accel_prev = self._accel_short_active(-1)
                if accel_now and not accel_prev:
                    entry_price = self.data.close[0]
                    entry_atr = self.atr[0]
                    stop_price = entry_price + self.p.stop_atr * entry_atr
                    order = self._risk_sell(stop_price=stop_price, entry_price=entry_price)
                    if order is not None:
                        self.entry_price = entry_price
                        self.entry_atr = entry_atr
                        self.stop_price = stop_price
                        self.bars_in_trade = 0
                        return

        if self.position and self._is_last_bar():
            self.close()
            self._reset_state()


def run(
    data,
    commission_,
    slippage,
    sizer,
    interval,
    interval_to_timeframe,
    trend_len=50,
    slope_lookback=5,
    accel_mult=1.0,
    atr_len=14,
    stop_atr=2.0,
    max_hold_bars=60,
    risk_config=None,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        EMAAccelerationBreakout,
        trend_len=trend_len,
        slope_lookback=slope_lookback,
        accel_mult=accel_mult,
        atr_len=atr_len,
        stop_atr=stop_atr,
        max_hold_bars=max_hold_bars,
        risk_config=risk_config,
    )

    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)

    cerebro.broker.set_slippage_perc(perc=slippage)
    timeframe = interval_to_timeframe.get(interval, bt.TimeFrame.Days)
    cerebro.adddata(data)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, timeframe=timeframe, annualize=True, _name="sharpe"
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    return results[0]

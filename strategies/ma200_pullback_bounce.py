import backtrader as bt
from .risk_managed import RiskManagedMixin


class MA200PullbackBounce(RiskManagedMixin, bt.Strategy):
    params = dict(
        long_len=200,
        short_len=20,
        atr_len=14,
        stop_atr=2.0,
        max_hold_bars=60,
        risk_config=None,
    )

    def __init__(self):
        self._init_risk()
        self.long_ma = bt.ind.SMA(self.data.close, period=self.p.long_len)
        self.short_ma = bt.ind.SMA(self.data.close, period=self.p.short_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.long_pullback_armed = False
        self.short_pullback_armed = False
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _reset_state(self):
        self.long_pullback_armed = False
        self.short_pullback_armed = False
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _bias_direction(self):
        if self.data.close[0] > self.long_ma[0]:
            return 1
        if self.data.close[0] < self.long_ma[0]:
            return -1
        return 0

    def next(self):
        min_bars = max(self.p.long_len, self.p.short_len, self.p.atr_len) + 1
        if len(self.data) <= min_bars:
            return

        # Manage open position
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.data.close[0] < self.long_ma[0]:
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

            if self.data.close[0] > self.long_ma[0]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat: arm on pullback against dominant long/short bias
        if self.position.size == 0:
            bias_direction = self._bias_direction()
            if bias_direction != 1:
                self.long_pullback_armed = False
            if bias_direction != -1:
                self.short_pullback_armed = False

            if bias_direction == 1:
                if self.data.close[0] < self.short_ma[0]:
                    self.long_pullback_armed = True
                    return

                cross_up_short = (
                    self.data.close[-1] <= self.short_ma[-1]
                    and self.data.close[0] > self.short_ma[0]
                )
                if self.long_pullback_armed and cross_up_short:
                    entry_price = self.data.close[0]
                    entry_atr = self.atr[0]
                    stop_price = entry_price - self.p.stop_atr * entry_atr
                    order = self._risk_buy(stop_price=stop_price, entry_price=entry_price)
                    if order is not None:
                        self.entry_price = entry_price
                        self.entry_atr = entry_atr
                        self.stop_price = stop_price
                        self.bars_in_trade = 0
                        self.long_pullback_armed = False
                        self.short_pullback_armed = False
                        return

            if bias_direction == -1:
                if self.data.close[0] > self.short_ma[0]:
                    self.short_pullback_armed = True
                    return

                cross_down_short = (
                    self.data.close[-1] >= self.short_ma[-1]
                    and self.data.close[0] < self.short_ma[0]
                )
                if self.short_pullback_armed and cross_down_short:
                    entry_price = self.data.close[0]
                    entry_atr = self.atr[0]
                    stop_price = entry_price + self.p.stop_atr * entry_atr
                    order = self._risk_sell(stop_price=stop_price, entry_price=entry_price)
                    if order is not None:
                        self.entry_price = entry_price
                        self.entry_atr = entry_atr
                        self.stop_price = stop_price
                        self.bars_in_trade = 0
                        self.short_pullback_armed = False
                        self.long_pullback_armed = False
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
    long_len=200,
    short_len=20,
    atr_len=14,
    stop_atr=2.0,
    max_hold_bars=60,
    risk_config=None,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MA200PullbackBounce,
        long_len=long_len,
        short_len=short_len,
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

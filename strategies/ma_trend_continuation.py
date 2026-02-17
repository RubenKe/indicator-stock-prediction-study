import backtrader as bt


class MATrendContinuation(bt.Strategy):
    params = dict(
        trend_len=100,
        slope_lookback=5,
        stop_atr=2.0,
        atr_len=14,
        max_hold_bars=100,
        momentum_exit=True,  # toggle for Close < Low[-1] filter
    )

    def __init__(self):
        self.trend_ma = bt.ind.SMA(self.data.close, period=self.p.trend_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.entry_price = None
        self.stop_price = None
        self.bars_in_trade = 0

    # --- Helpers ---------------------------------------------------------
    def _reset_state(self):
        self.entry_price = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _trend_direction(self):
        if len(self) <= self.p.trend_len or len(self) <= self.p.slope_lookback:
            return 0
        if (
            self.trend_ma[0] > self.trend_ma[-self.p.slope_lookback]
            and self.data.close[0] > self.trend_ma[0]
        ):
            return 1
        if (
            self.trend_ma[0] < self.trend_ma[-self.p.slope_lookback]
            and self.data.close[0] < self.trend_ma[0]
        ):
            return -1
        return 0

    # --- Core loop -------------------------------------------------------
    def next(self):
        min_bars = max(
            self.p.trend_len,
            self.p.atr_len,
            self.p.slope_lookback + 1,
            2,  # for previous high/low lookback
        )
        if len(self.data) <= min_bars:
            return

        # Manage open position
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.data.close[0] < self.trend_ma[0]:
                self.close()
                self._reset_state()
                return

            if self.p.momentum_exit and self.data.close[0] < self.data.low[-1]:
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

            if self.data.close[0] > self.trend_ma[0]:
                self.close()
                self._reset_state()
                return

            if self.p.momentum_exit and self.data.close[0] > self.data.high[-1]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat: look for continuation with trend alignment
        if self.position.size == 0:
            trend_direction = self._trend_direction()
            if trend_direction == 1 and self.data.close[0] > self.data.high[-1]:
                self.buy()
                self.entry_price = self.data.close[0]
                self.stop_price = self.entry_price - self.p.stop_atr * self.atr[0]
                self.bars_in_trade = 0
                return
            if trend_direction == -1 and self.data.close[0] < self.data.low[-1]:
                self.sell()
                self.entry_price = self.data.close[0]
                self.stop_price = self.entry_price + self.p.stop_atr * self.atr[0]
                self.bars_in_trade = 0
                return

        if self.position and self._is_last_bar():
            self.close()
            self._reset_state()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    trend_len=100,
    slope_lookback=5,
    stop_atr=2.0,
    atr_len=14,
    max_hold_bars=100,
    momentum_exit=True,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MATrendContinuation,
        trend_len=trend_len,
        slope_lookback=slope_lookback,
        stop_atr=stop_atr,
        atr_len=atr_len,
        max_hold_bars=max_hold_bars,
        momentum_exit=momentum_exit,
    )

    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer)

    timeframe = interval_to_timeframe.get(interval, bt.TimeFrame.Days)
    cerebro.adddata(data)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, timeframe=timeframe, annualize=True, _name="sharpe"
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    return results[0]

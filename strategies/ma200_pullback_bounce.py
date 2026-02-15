import backtrader as bt


class MA200PullbackBounce(bt.Strategy):
    params = dict(
        long_len=200,
        short_len=20,
        atr_len=14,
        stop_atr=2.0,
        max_hold_bars=60,
    )

    def __init__(self):
        self.long_ma = bt.ind.SMA(self.data.close, period=self.p.long_len)
        self.short_ma = bt.ind.SMA(self.data.close, period=self.p.short_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.pullback_armed = False
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _reset_state(self):
        self.pullback_armed = False
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _long_bias_valid(self):
        return self.data.close[0] > self.long_ma[0]

    def next(self):
        min_bars = max(self.p.long_len, self.p.short_len, self.p.atr_len) + 1
        if len(self.data) <= min_bars:
            return

        if self.position.size < 0:
            self.close()
            self._reset_state()
            return

        # Manage open long
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

        # Flat: arm on dip below short MA while long-term bias is bullish
        if self.position.size == 0:
            if not self._long_bias_valid():
                self.pullback_armed = False
                return

            if self.data.close[0] < self.short_ma[0]:
                self.pullback_armed = True
                return

            cross_up_short = (
                self.data.close[-1] <= self.short_ma[-1]
                and self.data.close[0] > self.short_ma[0]
            )
            if self.pullback_armed and cross_up_short:
                self.buy()
                self.entry_price = self.data.close[0]
                self.entry_atr = self.atr[0]
                self.stop_price = self.entry_price - self.p.stop_atr * self.entry_atr
                self.bars_in_trade = 0
                self.pullback_armed = False
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
    long_len=200,
    short_len=20,
    atr_len=14,
    stop_atr=2.0,
    max_hold_bars=60,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        MA200PullbackBounce,
        long_len=long_len,
        short_len=short_len,
        atr_len=atr_len,
        stop_atr=stop_atr,
        max_hold_bars=max_hold_bars,
    )

    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    cerebro.broker.set_shortcash(False)
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

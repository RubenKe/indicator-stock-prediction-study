import backtrader as bt


class RSIPullbackTrend(bt.Strategy):
    params = dict(
        trend_len=100,
        slope_lookback=5,
        rsi_len=14,
        rsi_pullback=40,
        rsi_recover=50,
        atr_len=14,
        stop_atr=2.0,
        max_hold_bars=100,
    )

    def __init__(self):
        self.trend_ma = bt.ind.SMA(self.data.close, period=self.p.trend_len)
        self.rsi = bt.ind.RSI(period=self.p.rsi_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.entry_price = None
        self.stop_price = None
        self.bars_in_trade = 0
        self.long_pullback_armed = False
        self.short_pullback_armed = False

    def _trend_direction(self):
        if len(self.data) <= max(self.p.trend_len, self.p.slope_lookback):
            return 0
        ma_now = self.trend_ma[0]
        ma_past = self.trend_ma[-self.p.slope_lookback]
        if self.data.close[0] > ma_now and ma_now > ma_past:
            return 1
        if self.data.close[0] < ma_now and ma_now < ma_past:
            return -1
        return 0

    def _reset_state(self):
        self.entry_price = None
        self.stop_price = None
        self.bars_in_trade = 0
        self.long_pullback_armed = False
        self.short_pullback_armed = False

    def next(self):
        min_bars = max(self.p.trend_len, self.p.rsi_len, self.p.atr_len) + self.p.slope_lookback
        if len(self.data) <= min_bars:
            return

        short_pullback = 100 - self.p.rsi_pullback
        short_recover = 100 - self.p.rsi_recover
        trend_direction = self._trend_direction()

        # Exit management
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

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat logic
        if self.position.size == 0:
            if trend_direction != 1:
                self.long_pullback_armed = False
            if trend_direction != -1:
                self.short_pullback_armed = False

            if trend_direction == 1:
                if not self.long_pullback_armed and self.rsi[0] < self.p.rsi_pullback:
                    self.long_pullback_armed = True
                    return

                if self.long_pullback_armed and self.rsi[-1] <= self.p.rsi_recover < self.rsi[0]:
                    self.buy()
                    self.entry_price = self.data.close[0]
                    self.stop_price = self.entry_price - self.p.stop_atr * self.atr[0]
                    self.bars_in_trade = 0
                    self.long_pullback_armed = False
                    self.short_pullback_armed = False
                    return

            if trend_direction == -1:
                if not self.short_pullback_armed and self.rsi[0] > short_pullback:
                    self.short_pullback_armed = True
                    return

                if self.short_pullback_armed and self.rsi[-1] >= short_recover > self.rsi[0]:
                    self.sell()
                    self.entry_price = self.data.close[0]
                    self.stop_price = self.entry_price + self.p.stop_atr * self.atr[0]
                    self.bars_in_trade = 0
                    self.short_pullback_armed = False
                    self.long_pullback_armed = False
                    return

        if self.position and self._is_last_bar():
            self.close()
            self._reset_state()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    trend_len=100,
    slope_lookback=5,
    rsi_len=14,
    rsi_pullback=40,
    rsi_recover=50,
    atr_len=14,
    stop_atr=2.0,
    max_hold_bars=100,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        RSIPullbackTrend,
        trend_len=trend_len,
        slope_lookback=slope_lookback,
        rsi_len=rsi_len,
        rsi_pullback=rsi_pullback,
        rsi_recover=rsi_recover,
        atr_len=atr_len,
        stop_atr=stop_atr,
        max_hold_bars=max_hold_bars,
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

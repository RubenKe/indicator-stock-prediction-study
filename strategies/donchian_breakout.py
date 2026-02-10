import backtrader as bt


class DonchianBreakout(bt.Strategy):
    params = dict(
        entry_len=20,
        exit_len=10,
        atr_len=14,
        entry_buffer_atr=0.0,
        stop_atr=2.0,
        trail_atr=3.0,
        max_hold_bars=200,
        cooldown_bars=0,
    )

    def __init__(self):
        self.dc_high = bt.ind.Highest(self.data.high, period=self.p.entry_len)
        self.dc_low = bt.ind.Lowest(self.data.low, period=self.p.exit_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.entry_price = None
        self.entry_atr = None
        self.entry_bar = None
        self.highest_close = None
        self.cooldown_end_bar = None

    def _in_cooldown(self):
        return self.cooldown_end_bar is not None and len(self) <= self.cooldown_end_bar

    def _reset_trade(self):
        self.entry_price = None
        self.entry_atr = None
        self.entry_bar = None
        self.highest_close = None

    def next(self):
        min_bars = max(self.p.entry_len, self.p.exit_len)
        if len(self) <= min_bars:
            return

        close_prev = self.data.close[-1]
        close_now = self.data.close[0]
        upper_prev = self.dc_high[-1]
        lower_prev = self.dc_low[-1]
        atr_now = self.atr[0]

        # Manage open long
        if self.position.size > 0:
            self.highest_close = max(self.highest_close, close_now)

            stop_price = self.entry_price - self.p.stop_atr * self.entry_atr
            trail_stop = self.highest_close - self.p.trail_atr * atr_now
            max_hold_hit = (len(self) - self.entry_bar) >= self.p.max_hold_bars

            if (
                close_now <= lower_prev
                or close_now <= stop_price
                or close_now <= trail_stop
                or max_hold_hit
            ):
                self.close()
                self._reset_trade()
                if self.p.cooldown_bars > 0:
                    self.cooldown_end_bar = len(self) + self.p.cooldown_bars
                return

        # Flat or cooldown: look for entries
        if self.position.size == 0 and not self._in_cooldown():
            buffered_upper = upper_prev + self.p.entry_buffer_atr * atr_now
            if close_prev <= upper_prev and close_now > buffered_upper:
                self.buy()
                self.entry_price = close_now
                self.entry_atr = atr_now
                self.entry_bar = len(self)
                self.highest_close = close_now
                self.cooldown_end_bar = None
                return

        if self.position and self._is_last_bar():
            self.close()
            self._reset_trade()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    entry_len=20,
    exit_len=10,
    atr_len=14,
    entry_buffer_atr=0.0,
    stop_atr=2.0,
    trail_atr=3.0,
    max_hold_bars=200,
    cooldown_bars=0,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        DonchianBreakout,
        entry_len=entry_len,
        exit_len=exit_len,
        atr_len=atr_len,
        entry_buffer_atr=entry_buffer_atr,
        stop_atr=stop_atr,
        trail_atr=trail_atr,
        max_hold_bars=max_hold_bars,
        cooldown_bars=cooldown_bars,
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

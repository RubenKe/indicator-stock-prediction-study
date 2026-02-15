import backtrader as bt


class HHHLStructureBreakout(bt.Strategy):
    params = dict(
        swing_len=5,
        break_buffer=0.3,
        stop_buffer=1.0,
        atr_len=14,
        slope_lookback=5,
        range_mult=1.0,
        min_structure_size=1.0,
        atr_regime_exit=False,
        atr_collapse_mult=0.7,
        max_hold_bars=60,
    )

    def __init__(self):
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.last_swing_high = None
        self.last_swing_high_bar = None
        self.last_swing_low = None
        self.last_swing_low_bar = None
        self.prev_swing_low = None

        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _reset_trade_state(self):
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _update_structure(self):
        k = self.p.swing_len
        pivot_bar = len(self) - k

        high_window = [self.data.high[i] for i in range(-2 * k, 1)]
        pivot_high = self.data.high[-k]
        if pivot_high == max(high_window) and self.last_swing_high_bar != pivot_bar:
            self.last_swing_high = pivot_high
            self.last_swing_high_bar = pivot_bar

        low_window = [self.data.low[i] for i in range(-2 * k, 1)]
        pivot_low = self.data.low[-k]
        if pivot_low == min(low_window) and self.last_swing_low_bar != pivot_bar:
            self.prev_swing_low = self.last_swing_low
            self.last_swing_low = pivot_low
            self.last_swing_low_bar = pivot_bar

    def _bullish_structure(self):
        if self.prev_swing_low is None or self.last_swing_low is None:
            return False
        return self.last_swing_low > self.prev_swing_low

    def _volatility_expansion_ok(self):
        if len(self) <= self.p.slope_lookback:
            return False
        atr_rising = self.atr[0] > self.atr[-self.p.slope_lookback]
        range_expanding = (self.data.high[0] - self.data.low[0]) > (
            self.p.range_mult * self.atr[0]
        )
        return atr_rising or range_expanding

    def _structure_size_ok(self):
        if self.last_swing_high is None or self.last_swing_low is None:
            return False
        structure_size = self.last_swing_high - self.last_swing_low
        return structure_size >= self.p.min_structure_size * self.atr[0]

    def next(self):
        min_bars = max(
            2 * self.p.swing_len + 1,
            self.p.atr_len + 1,
            self.p.slope_lookback + 1,
            2,
        )
        if len(self.data) <= min_bars:
            return

        if self.position.size < 0:
            self.close()
            self._reset_trade_state()
            return

        self._update_structure()

        # Manage open long
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_trade_state()
                return

            if (
                self.last_swing_low is not None
                and self.data.close[0] < self.last_swing_low
            ):
                self.close()
                self._reset_trade_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_trade_state()
                return

            if (
                self.p.atr_regime_exit
                and self.entry_atr is not None
                and self.atr[0] < self.p.atr_collapse_mult * self.entry_atr
            ):
                self.close()
                self._reset_trade_state()
                return

        # Flat: enter on break of latest swing high in bullish structure
        if self.position.size == 0:
            break_level = None
            break_level_prev = None
            if self.last_swing_high is not None:
                break_level = self.last_swing_high + self.p.break_buffer * self.atr[0]
                break_level_prev = (
                    self.last_swing_high + self.p.break_buffer * self.atr[-1]
                )

            if (
                self._bullish_structure()
                and self.last_swing_high is not None
                and self.last_swing_low is not None
                and self._structure_size_ok()
                and self._volatility_expansion_ok()
                and self.data.close[-1] <= break_level_prev
                and self.data.close[0] > break_level
            ):
                self.buy()
                self.entry_price = self.data.close[0]
                self.entry_atr = self.atr[0]
                self.stop_price = (
                    self.last_swing_low - self.p.stop_buffer * self.entry_atr
                )
                self.bars_in_trade = 0
                return

        if self.position and self._is_last_bar():
            self.close()
            self._reset_trade_state()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    swing_len=5,
    break_buffer=0.3,
    stop_buffer=1.0,
    atr_len=14,
    slope_lookback=5,
    range_mult=1.0,
    min_structure_size=1.0,
    atr_regime_exit=False,
    atr_collapse_mult=0.7,
    max_hold_bars=60,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        HHHLStructureBreakout,
        swing_len=swing_len,
        break_buffer=break_buffer,
        stop_buffer=stop_buffer,
        atr_len=atr_len,
        slope_lookback=slope_lookback,
        range_mult=range_mult,
        min_structure_size=min_structure_size,
        atr_regime_exit=atr_regime_exit,
        atr_collapse_mult=atr_collapse_mult,
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

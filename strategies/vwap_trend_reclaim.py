import backtrader as bt


class VWAPTrendReclaim(bt.Strategy):
    params = dict(
        vwap_period=50,
        slope_lookback=5,
        setup_lookback=5,
        atr_len=14,
        stop_atr=2.0,
        max_hold_bars=80,
    )

    def __init__(self):
        volume_sum = bt.ind.SumN(self.data.volume, period=self.p.vwap_period)
        pv_sum = bt.ind.SumN(
            self.data.close * self.data.volume, period=self.p.vwap_period
        )
        vwap_raw = pv_sum / bt.ind.Max(volume_sum, 1e-12)
        close_sma = bt.ind.SMA(self.data.close, period=self.p.vwap_period)

        # Some feeds (notably FX) may provide zero volume; fallback keeps logic stable.
        self.vwap = bt.If(volume_sum > 0, vwap_raw, close_sma)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        self.long_setup_bar = None
        self.short_setup_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _reset_state(self):
        self.long_setup_bar = None
        self.short_setup_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.bars_in_trade = 0

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()

    def _vwap_trend_up(self):
        if len(self) <= self.p.slope_lookback:
            return False
        return self.vwap[0] > self.vwap[-self.p.slope_lookback]

    def _vwap_trend_down(self):
        if len(self) <= self.p.slope_lookback:
            return False
        return self.vwap[0] < self.vwap[-self.p.slope_lookback]

    def _setup_valid(self, setup_bar):
        if setup_bar is None:
            return False
        return (len(self) - setup_bar) <= self.p.setup_lookback

    def next(self):
        min_bars = max(
            self.p.vwap_period,
            self.p.atr_len,
            self.p.slope_lookback + 1,
            self.p.setup_lookback + 1,
        )
        if len(self.data) <= min_bars:
            return

        # Track setup context while flat.
        if self.position.size == 0:
            if self.data.close[0] < self.vwap[0]:
                self.long_setup_bar = len(self)
            if self.data.close[0] > self.vwap[0]:
                self.short_setup_bar = len(self)

        # Manage open position.
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.data.close[0] < self.vwap[0]:
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

            if self.data.close[0] > self.vwap[0]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat: reclaim / fail trigger.
        if self.position.size == 0:
            reclaim_up = (
                self.data.close[-1] <= self.vwap[-1]
                and self.data.close[0] > self.vwap[0]
            )
            reclaim_down = (
                self.data.close[-1] >= self.vwap[-1]
                and self.data.close[0] < self.vwap[0]
            )

            if (
                self._setup_valid(self.long_setup_bar)
                and reclaim_up
                and self._vwap_trend_up()
            ):
                self.buy()
                self.entry_price = self.data.close[0]
                self.entry_atr = self.atr[0]
                self.stop_price = self.entry_price - self.p.stop_atr * self.entry_atr
                self.bars_in_trade = 0
                self.long_setup_bar = None
                self.short_setup_bar = None
                return

            if (
                self._setup_valid(self.short_setup_bar)
                and reclaim_down
                and self._vwap_trend_down()
            ):
                self.sell()
                self.entry_price = self.data.close[0]
                self.entry_atr = self.atr[0]
                self.stop_price = self.entry_price + self.p.stop_atr * self.entry_atr
                self.bars_in_trade = 0
                self.long_setup_bar = None
                self.short_setup_bar = None
                return

            if not self._setup_valid(self.long_setup_bar):
                self.long_setup_bar = None
            if not self._setup_valid(self.short_setup_bar):
                self.short_setup_bar = None

        if self.position and self._is_last_bar():
            self.close()
            self._reset_state()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    vwap_period=50,
    slope_lookback=5,
    setup_lookback=5,
    atr_len=14,
    stop_atr=2.0,
    max_hold_bars=80,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        VWAPTrendReclaim,
        vwap_period=vwap_period,
        slope_lookback=slope_lookback,
        setup_lookback=setup_lookback,
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

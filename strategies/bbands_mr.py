import backtrader as bt


class BBandsMeanReversion(bt.Strategy):
    params = dict(
        bb_len=20,
        bb_k=2.0,
        rsi_len=14,
        rsi_os=30,
        adx_len=14,
        adx_max=20,
        trend_len=100,
        slope_len=10,
        slope_max=0.001,  # max abs slope per bar (fractional)
        atr_len=14,
        setup_lookback=2,
        z_atr_min=0.0,
        stop_atr=1.5,
        stop_confirm_bars=1,
        max_hold_bars=20,
        fail_bars=5,
        rsi_fail=40,
    )

    def __init__(self):
        # Core indicators
        self.bb_mid = bt.ind.SMA(self.data.close, period=self.p.bb_len)
        self.bb_std = bt.ind.StdDev(self.data.close, period=self.p.bb_len)
        self.bb_up = self.bb_mid + self.p.bb_k * self.bb_std
        self.bb_low = self.bb_mid - self.p.bb_k * self.bb_std
        self.bandwidth = (self.bb_up - self.bb_low) / bt.ind.Max(1e-12, self.bb_mid)

        self.rsi = bt.ind.RSI(period=self.p.rsi_len)
        self.adx = bt.ind.ADX(period=self.p.adx_len)
        self.trend_sma = bt.ind.SMA(self.data.close, period=self.p.trend_len)
        self.atr = bt.ind.ATR(period=self.p.atr_len)

        # State
        self.long_setup_bar = None
        self.short_setup_bar = None
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None

    # --- Helpers ---------------------------------------------------------
    def _regime_ok(self):
        min_bars = max(self.p.bb_len, self.p.trend_len, self.p.atr_len) + self.p.slope_len
        if len(self.data) < min_bars:
            return False
        weak_trend = self.adx[0] < self.p.adx_max
        slope = 0.0
        if len(self) > self.p.slope_len:
            # Normalize slope by SMA level to keep it scale-free across symbols
            denom = abs(self.trend_sma[-self.p.slope_len])
            slope = (
                (self.trend_sma[0] - self.trend_sma[-self.p.slope_len])
                / max(denom, 1e-12)
                / max(self.p.slope_len, 1)
            )
        slope_flat = abs(slope) <= self.p.slope_max
        return weak_trend and slope_flat

    def _record_setup(self):
        # Long setup
        dist_atr = (self.bb_low[0] - self.data.close[0]) / max(self.atr[0], 1e-12)
        if self.data.close[0] < self.bb_low[0] and self.rsi[0] < self.p.rsi_os:
            if dist_atr >= self.p.z_atr_min:
                self.long_setup_bar = len(self)

        # Short setup
        rsi_ob = 100 - self.p.rsi_os
        dist_atr_short = (self.data.close[0] - self.bb_up[0]) / max(self.atr[0], 1e-12)
        if self.data.close[0] > self.bb_up[0] and self.rsi[0] > rsi_ob:
            if dist_atr_short >= self.p.z_atr_min:
                self.short_setup_bar = len(self)

    def _setup_valid(self, setup_bar):
        if setup_bar is None:
            return False
        return (len(self) - setup_bar) <= self.p.setup_lookback

    def _enter_long(self):
        self.buy()
        self.entry_bar = len(self)
        self.entry_price = self.data.close[0]
        self.entry_atr = self.atr[0]

    def _enter_short(self):
        self.sell()
        self.entry_bar = len(self)
        self.entry_price = self.data.close[0]
        self.entry_atr = self.atr[0]

    def _should_exit_long(self):
        if self.entry_bar is None:
            return False

        # 1) hard stop (ATR-based)
        stop_price = self.entry_price - self.p.stop_atr * self.entry_atr
        if self.data.close[0] <= stop_price:
            return True

        # 2) time stop
        if (len(self) - self.entry_bar) >= self.p.max_hold_bars:
            return True

        # 3) primary target: middle band
        if self.data.close[0] >= self.bb_mid[0]:
            return True

        # 4) early exit: fails to rebound
        if (
            (len(self) - self.entry_bar) >= self.p.fail_bars
            and self.rsi[0] < self.p.rsi_fail
        ):
            return True
        if self.data.close[0] < self.bb_low[0]:
            return True

        return False

    def _should_exit_short(self):
        if self.entry_bar is None:
            return False

        stop_price = self.entry_price + self.p.stop_atr * self.entry_atr
        if self.data.close[0] >= stop_price:
            return True

        if (len(self) - self.entry_bar) >= self.p.max_hold_bars:
            return True

        if self.data.close[0] <= self.bb_mid[0]:
            return True

        if (
            (len(self) - self.entry_bar) >= self.p.fail_bars
            and self.rsi[0] > (100 - self.p.rsi_fail)
        ):
            return True
        if self.data.close[0] > self.bb_up[0]:
            return True

        return False

    def _reset_trade_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.long_setup_bar = None
        self.short_setup_bar = None

    # --- Core loop -------------------------------------------------------
    def next(self):
        # Reset setups when out of lookback
        if not self.position:
            if not self._setup_valid(self.long_setup_bar):
                self.long_setup_bar = None
            if not self._setup_valid(self.short_setup_bar):
                self.short_setup_bar = None

        # Record setups
        self._record_setup()

        regime_ok = self._regime_ok()

        # Exits
        if self.position.size > 0:
            if self._should_exit_long():
                self.close()
                self._reset_trade_state()
                return
        elif self.position.size < 0:
            if self._should_exit_short():
                self.close()
                self._reset_trade_state()
                return

        # Entries only if flat
        if not self.position and regime_ok:
            # Long trigger: close back above lower band within setup window
            if (
                self._setup_valid(self.long_setup_bar)
                and self.data.close[-1] < self.bb_low[-1]
                and self.data.close[0] > self.bb_low[0]
            ):
                self._enter_long()
                return
            if (
                self._setup_valid(self.short_setup_bar)
                and self.data.close[-1] > self.bb_up[-1]
                and self.data.close[0] < self.bb_up[0]
            ):
                self._enter_short()
                return

        # Safety: close on last bar
        if self.position and self._is_last_bar():
            self.close()
            self._reset_trade_state()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    bb_len=20,
    bb_k=2.0,
    rsi_len=14,
    rsi_os=30,
    adx_len=14,
    adx_max=20,
    trend_len=100,
    slope_len=10,
    slope_max=0.001,
    atr_len=14,
    setup_lookback=2,
    z_atr_min=0.0,
    stop_atr=1.5,
    stop_confirm_bars=1,
    max_hold_bars=20,
    fail_bars=5,
    rsi_fail=40,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        BBandsMeanReversion,
        bb_len=bb_len,
        bb_k=bb_k,
        rsi_len=rsi_len,
        rsi_os=rsi_os,
        adx_len=adx_len,
        adx_max=adx_max,
        trend_len=trend_len,
        slope_len=slope_len,
        slope_max=slope_max,
        atr_len=atr_len,
        setup_lookback=setup_lookback,
        z_atr_min=z_atr_min,
        stop_atr=stop_atr,
        stop_confirm_bars=stop_confirm_bars,
        max_hold_bars=max_hold_bars,
        fail_bars=fail_bars,
        rsi_fail=rsi_fail,
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

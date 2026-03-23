import backtrader as bt
from .risk_managed import RiskManagedMixin


class ATRVolatilityExpansionBreakout(RiskManagedMixin, bt.Strategy):
    params = dict(
        atr_len=14,
        atr_expansion_mult=1.3,
        stop_atr=2.0,
        breakout_len=3,
        max_hold_bars=100,
        risk_config=None,
    )

    def __init__(self):
        self._init_risk()
        self.atr = bt.ind.ATR(period=self.p.atr_len)
        self.atr_ma = bt.ind.SMA(self.atr, period=self.p.atr_len)
        self.highest_high = bt.ind.Highest(self.data.high, period=self.p.breakout_len)
        self.lowest_low = bt.ind.Lowest(self.data.low, period=self.p.breakout_len)

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

    def _expansion_ready(self):
        if len(self) <= 1:
            return False
        expanding = self.atr[0] > self.p.atr_expansion_mult * self.atr_ma[0]
        compressed_prior = self.atr[-1] <= self.atr_ma[-1]
        return expanding and compressed_prior

    def next(self):
        min_bars = max(self.p.atr_len * 2, self.p.breakout_len + 1)
        if len(self.data) <= min_bars:
            return

        # Manage open position
        if self.position.size > 0:
            self.bars_in_trade += 1

            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return

            if self.atr[0] < self.atr_ma[0]:
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

            if self.atr[0] < self.atr_ma[0]:
                self.close()
                self._reset_state()
                return

            if self.bars_in_trade >= self.p.max_hold_bars:
                self.close()
                self._reset_state()
                return

        # Flat: look for expansion + breakout
        if self.position.size == 0:
            if self._expansion_ready():
                breakout_up = self.highest_high[-1]
                breakout_down = self.lowest_low[-1]
                if self.data.close[0] > breakout_up:
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
                if self.data.close[0] < breakout_down:
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
    atr_len=14,
    atr_expansion_mult=1.3,
    stop_atr=2.0,
    breakout_len=3,
    max_hold_bars=100,
    risk_config=None,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ATRVolatilityExpansionBreakout,
        atr_len=atr_len,
        atr_expansion_mult=atr_expansion_mult,
        stop_atr=stop_atr,
        breakout_len=breakout_len,
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

    results = cerebro.run()
    return results[0]

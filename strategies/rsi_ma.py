import backtrader as bt
from .risk_managed import RiskManagedMixin

class RSI_MA(RiskManagedMixin, bt.Strategy):
    params = dict(
        ma=200,
        buy_rsi=30,
        sell_rsi=40,
        risk_config=None,
    )

    def __init__(self):
        self._init_risk()
        self.sma = bt.ind.SMA(period=self.p.ma)
        self.rsi = bt.ind.RSI(period=10)
        self.atr = bt.ind.ATR(period=int(self._risk_cfg["atr_len_for_sizing"]))
        self.entry_bar = None
        self.stop_price = None

    def _reset_state(self):
        self.entry_bar = None
        self.stop_price = None

    def next(self):
        # Called on each bar (candle) after sufficient data is available
        short_entry_rsi = 100 - self.p.buy_rsi
        short_exit_rsi = 100 - self.p.sell_rsi
        
        # Entry logic
        if not self.position:
            if self.data.close[0] > self.sma[0] and self.rsi[0] < self.p.buy_rsi:
                entry_price = self.data.close[0]
                stop_price = entry_price - float(self._risk_cfg["default_stop_atr"]) * self.atr[0]
                order = self._risk_buy(stop_price=stop_price, entry_price=entry_price)
                if order is not None:
                    self.entry_bar = len(self)
                    self.stop_price = stop_price
            elif self.data.close[0] < self.sma[0] and self.rsi[0] > short_entry_rsi:
                entry_price = self.data.close[0]
                stop_price = entry_price + float(self._risk_cfg["default_stop_atr"]) * self.atr[0]
                order = self._risk_sell(stop_price=stop_price, entry_price=entry_price)
                if order is not None:
                    self.entry_bar = len(self)
                    self.stop_price = stop_price
        elif self.position.size > 0:
            if (
                (self.stop_price is not None and self.data.close[0] <= self.stop_price)
                or self.rsi[0] > self.p.sell_rsi
                or (len(self) - self.entry_bar >= 10)
            ):
                self.close()
                self._reset_state()
        else:
            if (
                (self.stop_price is not None and self.data.close[0] >= self.stop_price)
                or self.rsi[0] < short_exit_rsi
                or (len(self) - self.entry_bar >= 10)
            ):
                self.close()
                self._reset_state()
        
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
    ma=200,
    buy_rsi=30,
    sell_rsi=40,
    risk_config=None,
):
    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Strategy
    cerebro.addstrategy(
        RSI_MA,
        ma=ma,
        buy_rsi=buy_rsi,
        sell_rsi=sell_rsi,
        risk_config=risk_config,
    )

    # Broker
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    
    # Interval-aware Sharpe
    timeframe = interval_to_timeframe.get(interval, bt.TimeFrame.Days)

    # add data 
    cerebro.adddata(data)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=timeframe,
        annualize=True,
        _name="sharpe"
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    return results[0]

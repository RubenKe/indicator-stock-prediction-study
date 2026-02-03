import backtrader as bt

class RSI_MA(bt.Strategy):
    params = dict(
        ma=200,
        buy_rsi=30,
        sell_rsi=40,
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.p.ma)
        self.rsi = bt.ind.RSI(period=10)
        self.buy_bar = None

    def next(self):
        # Called on each bar (candle) after sufficient data is available
        
        # BUY signal: close above 200 MA and RSI < 30 (oversold)
        if not self.position:  # Not in position
            if self.data.close[0] > self.sma[0] and self.rsi[0] < self.p.buy_rsi:
                self.buy()
                self.buy_bar = len(self)
        else:  # In position
            # SELL signal: RSI > 40 or after 10 periods
            if self.rsi[0] > self.p.sell_rsi or (len(self) - self.buy_bar >= 10):
                self.close()
        
        if self.position and self._is_last_bar():
            self.close()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(data, commission_, sizer, interval, interval_to_timeframe, ma=200, buy_rsi=30, sell_rsi=40):
    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Strategy
    cerebro.addstrategy(RSI_MA, ma=ma, buy_rsi=buy_rsi, sell_rsi=sell_rsi)

    # Broker
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer)
    
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
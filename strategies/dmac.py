import backtrader as bt

# Dual Moving Average Crossover (DMAC) Strategy
# Buys when fast MA crosses above slow MA, sells when it crosses below
class DMAC(bt.Strategy):
    # Define customizable parameters (can be overridden when running)
    params = dict(
        pfast=50,      # Fast moving average period
        pslow=100,     # Slow moving average period
        adx_period=14, # ADX lookback period
        adx_threshold=20  # Minimum ADX to allow entries
    )

    def __init__(self):
        # Create two simple moving averages with different periods
        sma1 = bt.ind.SMA(period=self.p.pfast)   # Fast MA (shorter period)
        sma2 = bt.ind.SMA(period=self.p.pslow)   # Slow MA (longer period)
        
        # CrossOver indicator: returns > 0 when sma1 crosses above sma2,
        # returns < 0 when sma1 crosses below sma2
        self.crossover = bt.ind.CrossOver(sma1, sma2)
        self.adx = bt.ind.ADX(period=self.p.adx_period)

    def next(self):
        # Called on each bar (candle) after sufficient data is available
        # Enforce long-only behavior: close any accidental short and skip new orders
        if self.position.size < 0:
            self.close()
            return

        # BUY signal: fast MA crosses above slow MA (uptrend) and ADX confirms trend strength
        if not self.position and self.crossover > 0 and self.adx[0] >= self.p.adx_threshold:
            self.buy()
        
        # SELL signal: fast MA crosses below slow MA (downtrend)
        elif self.position.size > 0 and self.crossover < 0:
            self.close()

        if self.position.size > 0 and self._is_last_bar():
            self.close()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(data, commission_, sizer, interval, interval_to_timeframe, pfast=50, pslow=100, adx_period=14, adx_threshold=20):

    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Strategy
    cerebro.addstrategy(
        DMAC,
        pfast=pfast,
        pslow=pslow,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
    )

    # Broker
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    # Long-only: avoid short cash accounting (still safe for long trades)
    cerebro.broker.set_shortcash(False)
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

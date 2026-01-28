import backtrader as bt

# Dual Moving Average Crossover (DMAC) Strategy
# Buys when fast MA crosses above slow MA, sells when it crosses below
class DMAC(bt.Strategy):
    # Define customizable parameters (can be overridden when running)
    params = dict(
        pfast=50,      # Fast moving average period
        pslow=100      # Slow moving average period
    )

    def __init__(self):
        # Create two simple moving averages with different periods
        sma1 = bt.ind.SMA(period=self.p.pfast)   # Fast MA (shorter period)
        sma2 = bt.ind.SMA(period=self.p.pslow)   # Slow MA (longer period)
        
        # CrossOver indicator: returns > 0 when sma1 crosses above sma2,
        # returns < 0 when sma1 crosses below sma2
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        # Called on each bar (candle) after sufficient data is available
        
        # BUY signal: fast MA crosses above slow MA (uptrend)
        if not self.position and self.crossover > 0:
            self.buy()
        
        # SELL signal: fast MA crosses below slow MA (downtrend)
        elif self.position and self.crossover < 0:
            self.close()

def run(data, commission_, sizer, pfast=50, pslow=100):

    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with custom parameters
    cerebro.addstrategy(DMAC, pfast=pfast, pslow=pslow)
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer)

    
    # Add price data to the engine
    cerebro.adddata(data)
    
    # Run the backtest
    results = cerebro.run()
    strat = results[0]
    
    # Return the engine (contains all results)
    return strat  
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
                self.sell()


def run(data, ma=200, buy_rsi=30, sell_rsi=40):

    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with custom parameters
    cerebro.addstrategy(RSI_MA, ma=ma, sell_rsi=sell_rsi, buy_rsi=buy_rsi)
    
    # Add price data to the engine
    cerebro.adddata(data)
    
    # Run the backtest
    cerebro.run()
    
    # Return the engine (contains all results)
    return cerebro  
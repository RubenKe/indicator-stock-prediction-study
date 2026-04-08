import backtrader as bt
from .risk_managed import RiskManagedMixin

# Dual Moving Average Crossover (DMAC) Strategy
# Buys when fast MA crosses above slow MA, sells when it crosses below
class DMAC(RiskManagedMixin, bt.Strategy):
    # Define customizable parameters (can be overridden when running)
    params = dict(
        pfast=50,      # Fast moving average period
        pslow=100,     # Slow moving average period
        adx_period=14, # ADX lookback period
        adx_threshold=20,  # Minimum ADX to allow entries
        risk_config=None,
    )

    def __init__(self):
        self._init_risk()

        # Create two simple moving averages with different periods
        sma1 = bt.ind.SMA(period=self.p.pfast)   # Fast MA (shorter period)
        sma2 = bt.ind.SMA(period=self.p.pslow)   # Slow MA (longer period)
        
        # CrossOver indicator: returns > 0 when sma1 crosses above sma2,
        # returns < 0 when sma1 crosses below sma2
        self.crossover = bt.ind.CrossOver(sma1, sma2)
        self.adx = bt.ind.ADX(period=self.p.adx_period)
        self.atr = bt.ind.ATR(period=int(self._risk_cfg["atr_len_for_sizing"]))

        self.stop_price = None

    def _reset_state(self):
        self.stop_price = None

    def next(self):
        # Called on each bar (candle) after sufficient data is available
        if self.position.size > 0:
            if self.stop_price is not None and self.data.close[0] <= self.stop_price:
                self.close()
                self._reset_state()
                return
            if self.crossover < 0:
                self.close()
                self._reset_state()
                return

        if self.position.size < 0:
            if self.stop_price is not None and self.data.close[0] >= self.stop_price:
                self.close()
                self._reset_state()
                return
            if self.crossover > 0:
                self.close()
                self._reset_state()
                return

        # Entry signals: trade both sides when trend strength is present
        if not self.position and self.adx[0] >= self.p.adx_threshold:
            stop_mult = float(self._risk_cfg["default_stop_atr"])
            if self.crossover > 0:
                entry_price = self.data.close[0]
                stop_price = entry_price - stop_mult * self.atr[0]
                order = self._risk_buy(stop_price=stop_price, entry_price=entry_price)
                if order is not None:
                    self.stop_price = stop_price
            elif self.crossover < 0:
                entry_price = self.data.close[0]
                stop_price = entry_price + stop_mult * self.atr[0]
                order = self._risk_sell(stop_price=stop_price, entry_price=entry_price)
                if order is not None:
                    self.stop_price = stop_price

        if self.position and self._is_last_bar():
            self.close()
            self._reset_state()

    def _is_last_bar(self):
        return len(self.data) - 1 == self.data._last()


def run(
    data,
    commission_,
    slippage,
    sizer,
    interval,
    interval_to_timeframe,
    pfast=50,
    pslow=100,
    adx_period=14,
    adx_threshold=20,
    risk_config=None,
):

    # Create Backtrader engine
    cerebro = bt.Cerebro()
    
    # Strategy
    cerebro.addstrategy(
        DMAC,
        pfast=pfast,
        pslow=pslow,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        risk_config=risk_config,
    )

    # Broker
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    
    cerebro.broker.set_slippage_perc(perc=slippage)
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
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    return results[0]

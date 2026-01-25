import backtrader as bt

class DMAC(bt.Strategy):
    params = dict(
        pfast=50,
        pslow=100
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)
        sma2 = bt.ind.SMA(period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()

def run(data, pfast=50, pslow=100):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DMAC, pfast=pfast, pslow=pslow)
    cerebro.adddata(data)
    cerebro.run()
    return cerebro  
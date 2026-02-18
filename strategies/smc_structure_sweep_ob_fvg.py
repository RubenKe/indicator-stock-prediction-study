from collections import deque

import backtrader as bt
from .risk_managed import RiskManagedMixin


class StructureSweepOBFVG(RiskManagedMixin, bt.Strategy):
    params = dict(
        pivot_L=3,
        eq_tol_atr=0.15,
        disp_atr=1.2,
        fvg_min_atr=0.4,
        atr_period=14,
        risk_per_trade=0.005,
        timeout_bars=40,
        debug=False,
        risk_config=None,
    )

    def __init__(self):
        self._init_risk(base_risk_per_trade=self.p.risk_per_trade)
        self.ltf = self.datas[0]
        self.htf = self.datas[1] if len(self.datas) > 1 else self.datas[0]

        self.atr = bt.ind.ATR(self.ltf, period=self.p.atr_period)

        self.ltf_swing_highs = deque(maxlen=200)
        self.ltf_swing_lows = deque(maxlen=200)
        self.htf_swing_highs = deque(maxlen=200)
        self.htf_swing_lows = deque(maxlen=200)

        self._last_ltf_len = 0
        self._last_htf_len = 0

        self.state = "IDLE"
        self.pending_sweep = None
        self.setup = None

        self.entry_order = None
        self.exit_order = None

        self.position_side = None
        self.entry_fill_price = None
        self.entry_bar = None
        self.stop_price = None
        self.initial_r = None
        self.hit_one_r = False
        self.hit_t1 = False
        self.t1 = None
        self.t2 = None

    # ---------- Utility ----------
    def _log(self, message):
        if self.p.debug:
            dt = self.ltf.datetime.datetime(0)
            print(f"{dt} | {self.state} | {message}")

    def _is_last_bar(self):
        return len(self.ltf) - 1 == self.ltf._last()

    def _reset_to_idle(self):
        self.state = "IDLE"
        self.pending_sweep = None
        self.setup = None
        self.entry_order = None
        self.exit_order = None
        self.position_side = None
        self.entry_fill_price = None
        self.entry_bar = None
        self.stop_price = None
        self.initial_r = None
        self.hit_one_r = False
        self.hit_t1 = False
        self.t1 = None
        self.t2 = None

    # ---------- Market Structure ----------
    def _update_pivots(self, data, L):
        swings_high = self.ltf_swing_highs if data is self.ltf else self.htf_swing_highs
        swings_low = self.ltf_swing_lows if data is self.ltf else self.htf_swing_lows

        if len(data) <= 2 * L:
            return

        pivot_bar = len(data) - 1 - L
        cand_high = float(data.high[-L])
        cand_low = float(data.low[-L])

        left_highs = [float(data.high[i]) for i in range(-2 * L, -L)]
        right_highs = [float(data.high[i]) for i in range(-L + 1, 1)]

        left_lows = [float(data.low[i]) for i in range(-2 * L, -L)]
        right_lows = [float(data.low[i]) for i in range(-L + 1, 1)]

        if left_highs and right_highs:
            if cand_high > max(left_highs) and cand_high >= max(right_highs):
                if not swings_high or swings_high[-1][0] != pivot_bar:
                    swings_high.append((pivot_bar, cand_high))

        if left_lows and right_lows:
            if cand_low < min(left_lows) and cand_low <= min(right_lows):
                if not swings_low or swings_low[-1][0] != pivot_bar:
                    swings_low.append((pivot_bar, cand_low))

    def _current_htf_bias(self):
        if len(self.htf_swing_highs) < 2 or len(self.htf_swing_lows) < 2:
            return "NEUTRAL"

        h_prev = self.htf_swing_highs[-2][1]
        h_last = self.htf_swing_highs[-1][1]
        l_prev = self.htf_swing_lows[-2][1]
        l_last = self.htf_swing_lows[-1][1]

        if h_last > h_prev and l_last > l_prev:
            return "UP"
        if h_last < h_prev and l_last < l_prev:
            return "DOWN"
        return "NEUTRAL"

    # ---------- Liquidity ----------
    def _dedupe_levels(self, levels, tol):
        if not levels:
            return []
        levels = sorted(levels)
        out = [levels[0]]
        for level in levels[1:]:
            if abs(level - out[-1]) > tol:
                out.append(level)
            else:
                out[-1] = 0.5 * (out[-1] + level)
        return out

    def _find_equal_highs_lows(self):
        atr_now = max(float(self.atr[0]), 1e-12)
        tol = self.p.eq_tol_atr * atr_now

        highs = [p for _, p in list(self.ltf_swing_highs)[-10:]]
        lows = [p for _, p in list(self.ltf_swing_lows)[-10:]]

        buy_side = highs[-1:] if highs else []
        sell_side = lows[-1:] if lows else []

        for i in range(len(highs)):
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) <= tol:
                    buy_side.append(0.5 * (highs[i] + highs[j]))

        for i in range(len(lows)):
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) <= tol:
                    sell_side.append(0.5 * (lows[i] + lows[j]))

        buy_side = self._dedupe_levels(buy_side, tol)
        sell_side = self._dedupe_levels(sell_side, tol)
        return {"buy_side": buy_side, "sell_side": sell_side}

    def _detect_sweep(self, direction, pools):
        if direction == "LONG":
            hits = [
                lvl
                for lvl in pools["sell_side"]
                if float(self.ltf.low[0]) < lvl and float(self.ltf.close[0]) > lvl
            ]
            if not hits:
                return None
            return {
                "direction": "LONG",
                "level": max(hits),
                "extreme": float(self.ltf.low[0]),
                "bar": len(self.ltf),
            }

        hits = [
            lvl
            for lvl in pools["buy_side"]
            if float(self.ltf.high[0]) > lvl and float(self.ltf.close[0]) < lvl
        ]
        if not hits:
            return None
        return {
            "direction": "SHORT",
            "level": min(hits),
            "extreme": float(self.ltf.high[0]),
            "bar": len(self.ltf),
        }

    # ---------- Displacement / BOS / FVG / OB ----------
    def _detect_displacement_and_bos(self, direction):
        atr_now = max(float(self.atr[0]), 1e-12)
        body = abs(float(self.ltf.close[0]) - float(self.ltf.open[0]))
        displacement = body >= self.p.disp_atr * atr_now

        if direction == "LONG":
            if len(self.ltf_swing_highs) < 1:
                return False, False
            bos = float(self.ltf.close[0]) > self.ltf_swing_highs[-1][1]
            return displacement and float(self.ltf.close[0]) > float(self.ltf.open[0]), bos

        if len(self.ltf_swing_lows) < 1:
            return False, False
        bos = float(self.ltf.close[0]) < self.ltf_swing_lows[-1][1]
        return displacement and float(self.ltf.close[0]) < float(self.ltf.open[0]), bos

    def _detect_fvg(self, direction):
        if len(self.ltf) < 3:
            return False, None, None

        atr_now = max(float(self.atr[0]), 1e-12)
        min_size = self.p.fvg_min_atr * atr_now

        if direction == "LONG":
            hi_2 = float(self.ltf.high[-2])
            lo_0 = float(self.ltf.low[0])
            if hi_2 < lo_0 and (lo_0 - hi_2) >= min_size:
                return True, hi_2, lo_0
            return False, None, None

        lo_2 = float(self.ltf.low[-2])
        hi_0 = float(self.ltf.high[0])
        if lo_2 > hi_0 and (lo_2 - hi_0) >= min_size:
            return True, hi_0, lo_2
        return False, None, None

    def _find_order_block(self, direction):
        max_lookback = 6
        for k in range(1, max_lookback + 1):
            o = float(self.ltf.open[-k])
            c = float(self.ltf.close[-k])
            h = float(self.ltf.high[-k])
            l = float(self.ltf.low[-k])

            if direction == "LONG" and c < o:
                return l, h
            if direction == "SHORT" and c > o:
                return l, h
        return None, None

    # ---------- Targets ----------
    def _find_targets(self, direction, entry_price):
        if direction == "LONG":
            t1_candidates = [p for _, p in self.ltf_swing_highs if p > entry_price]
            t2_candidates = [p for _, p in self.htf_swing_highs if p > entry_price]
            t1 = min(t1_candidates) if t1_candidates else None
            t2 = min(t2_candidates) if t2_candidates else t1
            return t1, t2

        t1_candidates = [p for _, p in self.ltf_swing_lows if p < entry_price]
        t2_candidates = [p for _, p in self.htf_swing_lows if p < entry_price]
        t1 = max(t1_candidates) if t1_candidates else None
        t2 = max(t2_candidates) if t2_candidates else t1
        return t1, t2

    # ---------- Order callbacks ----------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if self.entry_order is not None and order.ref == self.entry_order.ref:
            if order.status == order.Completed:
                self.entry_fill_price = float(order.executed.price)
                self.position_side = "LONG" if order.isbuy() else "SHORT"
                self.entry_bar = len(self.ltf)
                self.initial_r = abs(self.entry_fill_price - self.stop_price)
                self.hit_one_r = False
                self.hit_t1 = False
                self.t1, self.t2 = self._find_targets(self.position_side, self.entry_fill_price)
                self.state = "IN_POSITION"
                self._log(
                    f"ENTRY filled {self.position_side} @ {self.entry_fill_price:.5f}, "
                    f"stop={self.stop_price:.5f}, t1={self.t1}, t2={self.t2}"
                )
            else:
                self._log("ENTRY canceled/rejected")
                self._reset_to_idle()
            self.entry_order = None
            return

        if self.exit_order is not None and order.ref == self.exit_order.ref:
            if order.status == order.Completed:
                self._log(f"EXIT filled @ {order.executed.price:.5f}")
                self._reset_to_idle()
            else:
                self._log("EXIT canceled/rejected")
                self.exit_order = None

    # ---------- State handlers ----------
    def _build_setup_from_sweep(self):
        if self.pending_sweep is None:
            return

        bars_since = len(self.ltf) - self.pending_sweep["bar"]
        if bars_since < 1:
            return
        if bars_since > 6:
            self._log("Pending sweep expired (no displacement in 1-6 bars)")
            self.pending_sweep = None
            return

        direction = self.pending_sweep["direction"]
        has_disp, has_bos = self._detect_displacement_and_bos(direction)
        if not (has_disp and has_bos):
            return

        has_fvg, fvg_low, fvg_high = self._detect_fvg(direction)
        if not has_fvg:
            return

        ob_low, ob_high = self._find_order_block(direction)
        if ob_low is None:
            return

        entry_zone_low = max(ob_low, fvg_low)
        entry_zone_high = min(ob_high, fvg_high)
        if entry_zone_low >= entry_zone_high:
            return

        atr_now = max(float(self.atr[0]), 1e-12)
        buffer_ = 0.1 * atr_now

        if direction == "LONG":
            stop_price = min(self.pending_sweep["extreme"], ob_low) - buffer_
        else:
            stop_price = max(self.pending_sweep["extreme"], ob_high) + buffer_

        self.setup = {
            "direction": direction,
            "sweep_level": self.pending_sweep["level"],
            "sweep_extreme": self.pending_sweep["extreme"],
            "displacement_bar_index": len(self.ltf),
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "stop_price": stop_price,
            "setup_expiry_bar": len(self.ltf) + self.p.timeout_bars,
        }
        self.pending_sweep = None
        self.state = "SETUP_FOUND"
        self._log(
            f"SETUP {direction} zone=[{entry_zone_low:.5f}, {entry_zone_high:.5f}] "
            f"stop={stop_price:.5f}"
        )

    def _place_setup_order(self):
        if self.setup is None:
            self._reset_to_idle()
            return

        direction = self.setup["direction"]
        if direction == "LONG":
            entry_price = self.setup["entry_zone_high"]
        else:
            entry_price = self.setup["entry_zone_low"]

        stop_price = self.setup["stop_price"]
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 1e-12:
            self._log("Invalid setup (stop distance ~ 0)")
            self._reset_to_idle()
            return

        self.stop_price = stop_price
        if direction == "LONG":
            self.entry_order = self._risk_buy(
                stop_price=stop_price,
                entry_price=entry_price,
                exectype=bt.Order.Limit,
            )
        else:
            self.entry_order = self._risk_sell(
                stop_price=stop_price,
                entry_price=entry_price,
                exectype=bt.Order.Limit,
            )

        if self.entry_order is None:
            self._log("Invalid setup (size <= 0)")
            self._reset_to_idle()
            return

        size = self.entry_order.size

        self.state = "ORDER_PLACED"
        self._log(
            f"ORDER {direction} limit={entry_price:.5f} size={size:.4f} "
            f"expiry_bar={self.setup['setup_expiry_bar']}"
        )

    def _manage_pending_order(self):
        if self.entry_order is None:
            return

        if self.setup is None:
            self.cancel(self.entry_order)
            return

        if len(self.ltf) > self.setup["setup_expiry_bar"]:
            self._log("Cancel entry (setup timeout)")
            self.cancel(self.entry_order)

    def _request_exit(self, reason):
        if self.exit_order is None:
            self._log(f"Exit requested: {reason}")
            self.exit_order = self.close()

    def _manage_open_position(self):
        if self.position.size == 0:
            self._reset_to_idle()
            return

        bars_in_trade = len(self.ltf) - self.entry_bar if self.entry_bar is not None else 0

        if self.position_side == "LONG":
            if self.initial_r and not self.hit_one_r:
                if float(self.ltf.high[0]) >= self.entry_fill_price + self.initial_r:
                    self.hit_one_r = True

            if self.t1 is not None and not self.hit_t1 and float(self.ltf.high[0]) >= self.t1:
                self.hit_t1 = True
                self.stop_price = max(self.stop_price, self.entry_fill_price)

            if float(self.ltf.low[0]) <= self.stop_price:
                self._request_exit("stop hit long")
                return

            if self.t2 is not None and float(self.ltf.high[0]) >= self.t2:
                self._request_exit("t2 hit long")
                return

        else:
            if self.initial_r and not self.hit_one_r:
                if float(self.ltf.low[0]) <= self.entry_fill_price - self.initial_r:
                    self.hit_one_r = True

            if self.t1 is not None and not self.hit_t1 and float(self.ltf.low[0]) <= self.t1:
                self.hit_t1 = True
                self.stop_price = min(self.stop_price, self.entry_fill_price)

            if float(self.ltf.high[0]) >= self.stop_price:
                self._request_exit("stop hit short")
                return

            if self.t2 is not None and float(self.ltf.low[0]) <= self.t2:
                self._request_exit("t2 hit short")
                return

        if bars_in_trade >= self.p.timeout_bars and not self.hit_one_r:
            self._request_exit("time stop without +1R")

    # ---------- Main loop ----------
    def next(self):
        min_bars = max(2 * self.p.pivot_L + 2, self.p.atr_period + 3, 10)
        if len(self.ltf) <= min_bars or len(self.htf) <= (2 * self.p.pivot_L + 2):
            return

        if len(self.ltf) != self._last_ltf_len:
            self._last_ltf_len = len(self.ltf)
            self._update_pivots(self.ltf, self.p.pivot_L)

        if len(self.htf) != self._last_htf_len:
            self._last_htf_len = len(self.htf)
            self._update_pivots(self.htf, self.p.pivot_L)

        if self.state == "IDLE" and self.position.size == 0:
            bias = self._current_htf_bias()
            if bias == "NEUTRAL":
                self.pending_sweep = None
            else:
                direction = "LONG" if bias == "UP" else "SHORT"
                pools = self._find_equal_highs_lows()

                if self.pending_sweep is None:
                    sweep = self._detect_sweep(direction, pools)
                    if sweep is not None:
                        self.pending_sweep = sweep
                        self._log(
                            f"Sweep detected {direction} level={sweep['level']:.5f} "
                            f"extreme={sweep['extreme']:.5f}"
                        )

                self._build_setup_from_sweep()

        if self.state == "SETUP_FOUND":
            self._place_setup_order()
        elif self.state == "ORDER_PLACED":
            self._manage_pending_order()
        elif self.state == "IN_POSITION":
            self._manage_open_position()

        if self.position and self._is_last_bar():
            self._request_exit("last bar cleanup")


def _infer_htf_resample(interval):
    mapping = {
        "1m": (bt.TimeFrame.Minutes, 5),
        "5m": (bt.TimeFrame.Minutes, 15),
        "15m": (bt.TimeFrame.Minutes, 60),
        "30m": (bt.TimeFrame.Minutes, 120),
        "1h": (bt.TimeFrame.Minutes, 240),
        "4h": (bt.TimeFrame.Days, 1),
        "1d": (bt.TimeFrame.Weeks, 1),
    }
    return mapping.get(interval, (bt.TimeFrame.Days, 1))


def run(
    data,
    commission_,
    sizer,
    interval,
    interval_to_timeframe,
    pivot_L=3,
    eq_tol_atr=0.15,
    disp_atr=1.2,
    fvg_min_atr=0.4,
    atr_period=14,
    risk_per_trade=0.005,
    timeout_bars=40,
    htf_data=None,
    htf_timeframe=None,
    htf_compression=None,
    debug=False,
    risk_config=None,
):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        StructureSweepOBFVG,
        pivot_L=pivot_L,
        eq_tol_atr=eq_tol_atr,
        disp_atr=disp_atr,
        fvg_min_atr=fvg_min_atr,
        atr_period=atr_period,
        risk_per_trade=risk_per_trade,
        timeout_bars=timeout_bars,
        debug=debug,
        risk_config=risk_config,
    )

    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=commission_)
    # Ensure end-of-data market exits are executed on the current close.
    cerebro.broker.set_coc(True)

    if isinstance(data, (list, tuple)):
        ltf_data = data[0]
        ext_htf_data = data[1] if len(data) > 1 else None
    elif isinstance(data, dict):
        ltf_data = data.get("ltf") or data.get("data0")
        ext_htf_data = data.get("htf") or data.get("data1")
    else:
        ltf_data = data
        ext_htf_data = None

    cerebro.adddata(ltf_data)

    chosen_htf = htf_data if htf_data is not None else ext_htf_data
    if chosen_htf is not None:
        cerebro.adddata(chosen_htf)
    else:
        if htf_timeframe is None or htf_compression is None:
            htf_timeframe, htf_compression = _infer_htf_resample(interval)
        cerebro.resampledata(
            ltf_data, timeframe=htf_timeframe, compression=htf_compression
        )

    timeframe = interval_to_timeframe.get(interval, bt.TimeFrame.Days)
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, timeframe=timeframe, annualize=True, _name="sharpe"
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    return results[0]

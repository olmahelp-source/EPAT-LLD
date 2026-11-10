"""Built-in strategies. Each demonstrates one family and doubles as a template."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..core.exceptions import StrategyError
from ..core.models import Signal
from ..quant import indicators as ind
from ..stats.regression import ols
from .base import STRATEGIES, Strategy, StrategyContext


@STRATEGIES.register("sma_crossover")
class SMACrossover(Strategy):
    """Trend following: long when the fast average is above the slow one."""
    name = "sma_crossover"
    PARAMS = {"fast": 10, "slow": 40, "allow_short": False}
    PARAM_BOUNDS = {"fast": (3, 30), "slow": (20, 120)}

    def validate_params(self) -> None:
        if not 1 <= self.p("fast") < self.p("slow"):
            raise StrategyError("require 1 <= fast < slow", details=self.params)

    @property
    def warmup(self) -> int:
        return int(self.p("slow")) + 1

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out = []
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            c = ctx.closes(s, self.warmup + 1)
            f, sl = ind.sma(c, int(self.p("fast")))[-1], ind.sma(c, int(self.p("slow")))[-1]
            if not (np.isfinite(f) and np.isfinite(sl)):
                continue
            w = 1.0 if f > sl else (-1.0 if self.p("allow_short") else 0.0)
            out.append(Signal(s, w, f"sma{self.p('fast')}{'>' if f > sl else '<='}sma{self.p('slow')}"))
        return out


@STRATEGIES.register("rsi_reversion")
class RSIReversion(Strategy):
    """Mean reversion: buy oversold, exit at the mean; mirror for shorts."""
    name = "rsi_reversion"
    PARAMS = {"period": 14, "oversold": 30.0, "overbought": 70.0, "exit_level": 50.0, "allow_short": False}
    PARAM_BOUNDS = {"period": (5, 30), "oversold": (15, 40), "overbought": (60, 85)}

    def validate_params(self) -> None:
        if not self.p("oversold") < self.p("exit_level") < self.p("overbought"):
            raise StrategyError("require oversold < exit_level < overbought", details=self.params)
        self._state: Dict[str, int] = {}

    @property
    def warmup(self) -> int:
        return int(self.p("period")) + 2

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out = []
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            r = ind.rsi(ctx.closes(s, int(self.p("period")) * 5 + 5), int(self.p("period")))[-1]
            if not np.isfinite(r):
                continue
            st = self._state.get(s, 0)
            if st == 0:
                if r < self.p("oversold"):
                    st = 1
                elif r > self.p("overbought") and self.p("allow_short"):
                    st = -1
            elif (st == 1 and r >= self.p("exit_level")) or (st == -1 and r <= self.p("exit_level")):
                st = 0
            self._state[s] = st
            out.append(Signal(s, float(st), f"rsi={r:.1f}"))
        return out


@STRATEGIES.register("zscore_reversion")
class ZScoreReversion(Strategy):
    """Statistical mean reversion on the price z-score (Bollinger-style)."""
    name = "zscore_reversion"
    PARAMS = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "allow_short": True}
    PARAM_BOUNDS = {"lookback": (10, 60), "entry_z": (1.0, 3.0), "exit_z": (0.0, 1.0)}

    def validate_params(self) -> None:
        if self.p("exit_z") >= self.p("entry_z") or self.p("lookback") < 5:
            raise StrategyError("require exit_z < entry_z and lookback >= 5", details=self.params)
        self._state: Dict[str, int] = {}

    @property
    def warmup(self) -> int:
        return int(self.p("lookback")) + 1

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out = []
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            z = ind.zscore(ctx.closes(s, int(self.p("lookback")) + 1), int(self.p("lookback")))[-1]
            if not np.isfinite(z):
                continue
            st = self._state.get(s, 0)
            if st == 0:
                if z < -self.p("entry_z"):
                    st = 1
                elif z > self.p("entry_z") and self.p("allow_short"):
                    st = -1
            elif (st == 1 and z > -self.p("exit_z")) or (st == -1 and z < self.p("exit_z")):
                st = 0
            self._state[s] = st
            out.append(Signal(s, float(st), f"z={z:.2f}"))
        return out


@STRATEGIES.register("donchian_breakout")
class DonchianBreakout(Strategy):
    """Turtle-style channel breakout with a shorter exit channel."""
    name = "donchian_breakout"
    PARAMS = {"entry": 20, "exit": 10, "allow_short": False}
    PARAM_BOUNDS = {"entry": (10, 80), "exit": (5, 40)}

    def validate_params(self) -> None:
        if self.p("exit") > self.p("entry"):
            raise StrategyError("exit channel must not be longer than entry channel", details=self.params)
        self._state: Dict[str, int] = {}

    @property
    def warmup(self) -> int:
        return int(self.p("entry")) + 2

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out = []
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            w = ctx.window(s, int(self.p("entry")) + 2)
            h, l, c = w["high"], w["low"], w["close"][-1]
            en, ex = int(self.p("entry")), int(self.p("exit"))
            hi_entry, lo_entry = h[-en - 1:-1].max(), l[-en - 1:-1].min()     # exclude today: no look-ahead
            hi_exit, lo_exit = h[-ex - 1:-1].max(), l[-ex - 1:-1].min()
            st = self._state.get(s, 0)
            if st == 0:
                if c > hi_entry:
                    st = 1
                elif c < lo_entry and self.p("allow_short"):
                    st = -1
            elif st == 1 and c < lo_exit:
                st = 0
            elif st == -1 and c > hi_exit:
                st = 0
            self._state[s] = st
            out.append(Signal(s, float(st), "donchian"))
        return out


@STRATEGIES.register("momentum_rotation")
class MomentumRotation(Strategy):
    """Cross-sectional momentum: hold the top-N past performers, rebalanced periodically."""
    name = "momentum_rotation"
    PARAMS = {"lookback": 60, "skip": 5, "top_n": 2, "rebalance_every": 21, "long_short": False}
    PARAM_BOUNDS = {"lookback": (20, 180), "top_n": (1, 5), "rebalance_every": (5, 42)}

    def validate_params(self) -> None:
        if self.p("top_n") < 1 or self.p("top_n") > len(self.symbols):
            raise StrategyError("top_n must be between 1 and number of symbols", details=self.params)
        if self.p("skip") >= self.p("lookback"):
            raise StrategyError("skip must be smaller than lookback", details=self.params)
        self._bar = 0
        self._targets: Dict[str, float] = {}

    @property
    def warmup(self) -> int:
        return int(self.p("lookback")) + 1

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        self._bar += 1
        ready = [s for s in self.symbols if s in ctx.bars and self.ready(ctx, s)]
        if len(ready) < len(self.symbols):
            return []
        if self._targets and (self._bar % int(self.p("rebalance_every")) != 0):
            return [Signal(s, w, "hold") for s, w in self._targets.items()]
        lb, sk = int(self.p("lookback")), int(self.p("skip"))
        score = {}
        for s in ready:
            c = ctx.closes(s, lb + 1)
            score[s] = c[-1 - sk] / c[0] - 1.0            # skip most recent bars (short-term reversal)
        ranked = sorted(score, key=score.get, reverse=True)
        n = int(self.p("top_n"))
        self._targets = {s: 0.0 for s in self.symbols}
        for s in ranked[:n]:
            self._targets[s] = 1.0
        if self.p("long_short"):
            for s in ranked[-n:]:
                self._targets[s] = -1.0
        return [Signal(s, w, f"mom={score[s]:+.1%}") for s, w in self._targets.items()]


@STRATEGIES.register("pairs_trading")
class PairsTrading(Strategy):
    """Spread mean reversion on two symbols: ``symbols = [y, x]``; hedge ratio from rolling OLS."""
    name = "pairs_trading"
    PARAMS = {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0}
    PARAM_BOUNDS = {"lookback": (30, 120), "entry_z": (1.0, 3.0), "exit_z": (0.0, 1.0)}

    def validate_params(self) -> None:
        if len(self.symbols) != 2:
            raise StrategyError("pairs_trading needs exactly two symbols [y, x]")
        if not self.p("exit_z") < self.p("entry_z") < self.p("stop_z"):
            raise StrategyError("require exit_z < entry_z < stop_z", details=self.params)
        self._state = 0

    @property
    def warmup(self) -> int:
        return int(self.p("lookback")) + 1

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        y_sym, x_sym = self.symbols
        if y_sym not in ctx.bars or x_sym not in ctx.bars or not (self.ready(ctx, y_sym) and self.ready(ctx, x_sym)):
            return []
        n = int(self.p("lookback"))
        y, x = ctx.closes(y_sym, n), ctx.closes(x_sym, n)
        fit = ols(y, x, names=["x"])
        beta = fit["x"]
        spread = y - beta * x - fit["const"]
        sd = spread[:-1].std(ddof=1)
        if sd <= 0 or not math.isfinite(sd):
            return []
        z = spread[-1] / sd
        if self._state != 0 and abs(z) > self.p("stop_z"):
            self._state = 0                                   # relationship broke: stop out
        elif self._state == 0:
            if z < -self.p("entry_z"):
                self._state = 1                               # spread cheap: long y, short x
            elif z > self.p("entry_z"):
                self._state = -1
        elif (self._state == 1 and z > -self.p("exit_z")) or (self._state == -1 and z < self.p("exit_z")):
            self._state = 0
        # dollar-hedge the x leg: |beta| * Px / Py of the y leg, capped to the allowed range
        hedge_w = min(1.0, abs(beta) * x[-1] / y[-1])
        return [Signal(y_sym, float(self._state), f"z={z:.2f} beta={beta:.2f}"),
                Signal(x_sym, -float(self._state) * hedge_w * math.copysign(1, beta), "hedge")]

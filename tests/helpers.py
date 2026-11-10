import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from algoplatform.core.config import CostConfig, PlatformConfig
from algoplatform.core.enums import Role
from algoplatform.core.models import Bar
from algoplatform.data import MarketData
from algoplatform.users import PasswordHasher, UserManager

STRONG_PW = "Correct-Horse-9"


def df_from_closes(closes, start="2021-01-01", volume=1e6):
    closes = np.asarray(closes, float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(opens, closes) * 1.002
    low = np.minimum(opens, closes) * 0.998
    idx = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": closes,
                         "volume": volume}, index=idx)


def market_from_closes(**series):
    md = MarketData()
    for sym, closes in series.items():
        md.add(sym, df_from_closes(closes))
    return md


def free_cost_config(**kw):
    """No frictions: makes expected fills and P&L exact in tests."""
    return CostConfig(commission_per_share=0.0, commission_pct=0.0, min_commission=0.0,
                      slippage_bps=0.0, spread_bps=0.0, impact_coef=0.0, **kw)


def free_config(capital=100_000.0, **kw):
    return PlatformConfig(initial_capital=capital, costs=free_cost_config(), **kw)


def fast_user_manager():
    """scrypt with tiny cost so tests stay fast."""
    return UserManager(hasher=PasswordHasher(n=2 ** 4))


def make_user(um, name="alice", role=Role.TRADER):
    return um.register(name, STRONG_PW, role)


def bar(ts, sym="X", o=100.0, h=None, l=None, c=None, v=1e6):
    c = o if c is None else c
    h = max(o, c) if h is None else h
    l = min(o, c) if l is None else l
    return Bar(pd.Timestamp(ts), sym, o, h, l, c, v)

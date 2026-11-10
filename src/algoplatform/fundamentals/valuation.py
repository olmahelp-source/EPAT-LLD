"""Valuation: DCF (with the fragility made visible), reverse DCF, dividend/residual-income models, multiples."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from .statements import Statement


def capm(rf: float, beta: float, equity_risk_premium: float) -> float:
    return rf + beta * equity_risk_premium


def wacc(equity_value: float, debt_value: float, cost_equity: float, cost_debt: float, tax_rate: float = 0.25) -> float:
    v = equity_value + debt_value
    if v <= 0:
        raise ValidationError("capital must be positive")
    return equity_value / v * cost_equity + debt_value / v * cost_debt * (1 - tax_rate)


def dcf(fcf0: float, growth: float, discount: float, terminal_growth: float, years: int = 10, net_debt: float = 0.0,
        shares: float = 1.0, fade: bool = True) -> Dict[str, float]:
    """Two-stage DCF of free cash flow to the firm. Growth fades linearly to ``terminal_growth`` over ``years``
    when ``fade`` (the sober default), otherwise stays flat. ``terminal_share`` shows how much of the value is
    just the terminal assumption: above ~70% the answer is mostly a guess."""
    if discount <= terminal_growth:
        raise ValidationError("discount rate must exceed terminal growth", details={"discount": discount, "g": terminal_growth})
    if years < 1 or shares <= 0:
        raise ValidationError("years >= 1 and shares > 0 required")
    fcf, pv = fcf0, 0.0
    for t in range(1, years + 1):
        g = growth + (terminal_growth - growth) * (t - 1) / max(years - 1, 1) if fade else growth
        fcf *= 1 + g
        pv += fcf / (1 + discount) ** t
    tv = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    pv_tv = tv / (1 + discount) ** years
    ev = pv + pv_tv
    eq = ev - net_debt
    return {"enterprise_value": ev, "equity_value": eq, "per_share": eq / shares, "pv_explicit": pv, "pv_terminal": pv_tv,
            "terminal_share": pv_tv / ev if ev else float("nan")}


def terminal_sensitivity(fcf0: float, growth: float, discounts: Sequence[float], terminal_growths: Sequence[float], years: int = 10,
                         net_debt: float = 0.0, shares: float = 1.0) -> pd.DataFrame:
    """Per-share value across discount rate x terminal growth. A wide spread is the honest answer to 'what is it worth'."""
    rows = {g: [dcf(fcf0, growth, r, g, years, net_debt, shares)["per_share"] if r > g else np.nan for r in discounts]
            for g in terminal_growths}
    return pd.DataFrame(rows, index=[f"r={r:.1%}" for r in discounts]).T.rename_axis("terminal_g")


def reverse_dcf(price: float, fcf0: float, discount: float, terminal_growth: float, years: int = 10, net_debt: float = 0.0,
                shares: float = 1.0, lo: float = -0.5, hi: float = 1.0) -> Optional[float]:
    """The growth rate the market price implies (bisection). None if no growth in [lo, hi] justifies the price."""
    f = lambda g: dcf(fcf0, g, discount, terminal_growth, years, net_debt, shares)["per_share"] - price
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def gordon_ddm(dividend0: float, growth: float, required_return: float) -> float:
    if required_return <= growth:
        raise ValidationError("required return must exceed dividend growth")
    return dividend0 * (1 + growth) / (required_return - growth)


def residual_income(book0: float, roe: Sequence[float], cost_equity: float, payout: float = 0.0, terminal_decay: float = 0.0) -> float:
    """Value = book + PV of (ROE - r) x opening book. ``terminal_decay`` fades residual income after the forecast."""
    book, pv = book0, 0.0
    for t, r in enumerate(roe, start=1):
        ri = (r - cost_equity) * book
        pv += ri / (1 + cost_equity) ** t
        book *= 1 + r * (1 - payout)
    if terminal_decay > 0 and roe:
        last_ri = (roe[-1] - cost_equity) * book / (1 + roe[-1] * (1 - payout))
        pv += last_ri * terminal_decay / (1 + cost_equity - terminal_decay) / (1 + cost_equity) ** len(roe)
    return book0 + pv


def multiples(st: Statement, price: float) -> Dict[str, Optional[float]]:
    cap = price * st["shares"]
    ev = cap + st.total_debt - (st.get("cash") or 0.0)

    def r(a, b):
        return None if b is None or b <= 0 else a / b
    return {"pe": r(price, st.eps), "pb": r(cap, st["equity"]), "ps": r(cap, st["revenue"]),
            "ev_ebitda": r(ev, st.ebitda), "ev_sales": r(ev, st["revenue"]), "earnings_yield": r(st["net_income"], cap),
            "fcf_yield": r(st.free_cash_flow, cap) if st.get("cfo") is not None else None}


def margin_of_safety(intrinsic: float, price: float) -> float:
    """Discount of price to intrinsic value (1 - price / value). Positive = priced below the estimate."""
    if intrinsic <= 0:
        raise ValidationError("intrinsic value must be positive")
    return 1.0 - price / intrinsic

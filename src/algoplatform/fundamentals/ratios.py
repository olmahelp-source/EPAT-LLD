"""Ratio analysis. Pure functions of statements; missing optional inputs yield None rather than guesses."""
from __future__ import annotations

from typing import Dict, Optional

from ..utils.helpers import safe_div
from .statements import Statement


def _r(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None or b == 0 else a / b


def _avg(cur: float, prev: Optional[Statement], key: str) -> float:
    return cur if prev is None or prev.get(key) is None else (cur + prev[key]) / 2.0


def profitability(st: Statement, prev: Optional[Statement] = None, tax_rate: float = 0.25) -> Dict[str, Optional[float]]:
    rev = st["revenue"]
    nopat = st["operating_income"] * (1 - tax_rate)
    invested = st.total_debt + st["equity"] - (st.get("cash") or 0.0)
    return {"gross_margin": _r(st.gross_profit, rev), "operating_margin": _r(st["operating_income"], rev),
            "net_margin": _r(st["net_income"], rev), "roe": _r(st["net_income"], _avg(st["equity"], prev, "equity")),
            "roa": _r(st["net_income"], _avg(st["total_assets"], prev, "total_assets")), "roic": _r(nopat, invested),
            "ebitda_margin": _r(st.ebitda, rev)}


def leverage(st: Statement) -> Dict[str, Optional[float]]:
    ebitda, cash = st.ebitda, st.get("cash") or 0.0
    return {"debt_to_equity": _r(st.total_debt, st["equity"]), "net_debt_to_ebitda": _r(st.total_debt - cash, ebitda),
            "interest_coverage": _r(st["operating_income"], st.get("interest_expense")),
            "current_ratio": _r(st["current_assets"], st["current_liabilities"]),
            "quick_ratio": _r(st["current_assets"] - (st.get("inventory") or 0.0), st["current_liabilities"]),
            "liabilities_to_assets": _r(st.get("total_liabilities"), st.get("total_assets"))}


def efficiency(st: Statement, prev: Optional[Statement] = None, days: int = 365) -> Dict[str, Optional[float]]:
    rev, cogs = st["revenue"], st["cogs"]
    inv, rec, pay = st.get("inventory"), st.get("receivables"), st.get("payables")
    dio = None if inv is None else safe_div(inv, cogs) * days
    dso = None if rec is None else safe_div(rec, rev) * days
    dpo = None if pay is None else safe_div(pay, cogs) * days
    ccc = None if None in (dio, dso, dpo) else dio + dso - dpo
    return {"asset_turnover": _r(rev, _avg(st["total_assets"], prev, "total_assets")), "inventory_days": dio,
            "receivable_days": dso, "payable_days": dpo, "cash_conversion_cycle": ccc}


def growth(st: Statement, prev: Optional[Statement]) -> Dict[str, Optional[float]]:
    if prev is None:
        return {"revenue_growth": None, "eps_growth": None, "fcf_growth": None, "book_value_growth": None}

    def g(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None or b <= 0 else a / b - 1.0       # growth from a non-positive base is meaningless
    return {"revenue_growth": g(st["revenue"], prev["revenue"]),
            "eps_growth": g(st["net_income"] / st["shares"], prev["net_income"] / prev["shares"]) if st.get("shares") and prev.get("shares") else None,
            "fcf_growth": g(st.free_cash_flow if st.get("cfo") is not None else None, prev.free_cash_flow if prev.get("cfo") is not None else None),
            "book_value_growth": g(st["equity"], prev["equity"])}


def dupont3(st: Statement) -> Dict[str, Optional[float]]:
    """ROE = net margin x asset turnover x equity multiplier (year-end balances)."""
    nm, at, em = _r(st["net_income"], st["revenue"]), _r(st["revenue"], st["total_assets"]), _r(st["total_assets"], st["equity"])
    roe = None if None in (nm, at, em) else nm * at * em
    return {"net_margin": nm, "asset_turnover": at, "equity_multiplier": em, "roe": roe}


def dupont5(st: Statement) -> Dict[str, Optional[float]]:
    """ROE = tax burden x interest burden x EBIT margin x asset turnover x equity multiplier."""
    pretax = st.get("pretax_income")
    tb, ib = _r(st["net_income"], pretax), _r(pretax, st["operating_income"])
    em_, at, lev = _r(st["operating_income"], st["revenue"]), _r(st["revenue"], st["total_assets"]), _r(st["total_assets"], st["equity"])
    parts = (tb, ib, em_, at, lev)
    roe = None if None in parts else tb * ib * em_ * at * lev
    return {"tax_burden": tb, "interest_burden": ib, "ebit_margin": em_, "asset_turnover": at, "equity_multiplier": lev, "roe": roe}


def all_ratios(st: Statement, prev: Optional[Statement] = None) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for part in (profitability(st, prev), leverage(st), efficiency(st, prev), growth(st, prev)):
        out.update(part)
    return out

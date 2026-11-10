"""Composite quality / distress / manipulation scores."""
from __future__ import annotations

from typing import Dict, Optional

from ..core.exceptions import ValidationError
from ..utils.helpers import safe_div
from .statements import Statement


def piotroski_f(cur: Statement, prev: Statement) -> Dict[str, object]:
    """Piotroski F-score (0-9): profitability, leverage/liquidity and efficiency improvements. 8-9 strong, 0-2 weak."""
    ta, pta = cur["total_assets"], prev["total_assets"]
    roa, proa = cur["net_income"] / ta, prev["net_income"] / pta
    cfo = cur["cfo"]
    lev = safe_div(cur.get("long_term_debt") or 0.0, ta)
    plev = safe_div(prev.get("long_term_debt") or 0.0, pta)
    cr = safe_div(cur["current_assets"], cur["current_liabilities"])
    pcr = safe_div(prev["current_assets"], prev["current_liabilities"])
    gm = safe_div(cur.gross_profit, cur["revenue"])
    pgm = safe_div(prev.gross_profit, prev["revenue"])
    at, pat = safe_div(cur["revenue"], ta), safe_div(prev["revenue"], pta)
    checks = {"roa_positive": roa > 0, "cfo_positive": cfo > 0, "roa_improving": roa > proa,
              "cfo_exceeds_net_income": cfo > cur["net_income"], "leverage_falling": lev < plev,
              "current_ratio_rising": cr > pcr, "no_dilution": cur["shares"] <= prev["shares"],
              "gross_margin_rising": gm > pgm, "asset_turnover_rising": at > pat}
    return {"score": int(sum(checks.values())), "checks": checks}


def altman_z(st: Statement, market_cap: float) -> Dict[str, object]:
    """Original Altman Z for public manufacturers. >2.99 safe, 1.81-2.99 grey, <1.81 distress."""
    ta = st["total_assets"]
    if ta <= 0 or st["total_liabilities"] <= 0:
        raise ValidationError("total assets and liabilities must be positive")
    x = {"wc_ta": st.working_capital / ta, "re_ta": st["retained_earnings"] / ta, "ebit_ta": st["operating_income"] / ta,
         "mve_tl": market_cap / st["total_liabilities"], "sales_ta": st["revenue"] / ta}
    z = 1.2 * x["wc_ta"] + 1.4 * x["re_ta"] + 3.3 * x["ebit_ta"] + 0.6 * x["mve_tl"] + 1.0 * x["sales_ta"]
    zone = "safe" if z > 2.99 else "distress" if z < 1.81 else "grey"
    return {"z": z, "zone": zone, "components": x}


def beneish_m(cur: Statement, prev: Statement) -> Dict[str, object]:
    """Beneish 8-variable M-score. Above -1.78 flags a likely earnings manipulator (a screen, not proof)."""
    def idx(a: float, b: float) -> float:
        return safe_div(a, b, default=1.0)

    gm, pgm = cur.gross_profit / cur["revenue"], prev.gross_profit / prev["revenue"]
    ca, pca = cur["current_assets"], prev["current_assets"]
    ta, pta = cur["total_assets"], prev["total_assets"]
    dep, pdep = cur.get("depreciation") or 0.0, prev.get("depreciation") or 0.0
    v = {"DSRI": idx(cur["receivables"] / cur["revenue"], prev["receivables"] / prev["revenue"]),
         "GMI": idx(pgm, gm),
         "AQI": idx(1 - (ca + cur["ppe"]) / ta, 1 - (pca + prev["ppe"]) / pta),
         "SGI": idx(cur["revenue"], prev["revenue"]),
         "DEPI": idx(pdep / (pdep + prev["ppe"]), dep / (dep + cur["ppe"])),
         "SGAI": idx((cur.get("sga") or 0.0) / cur["revenue"], (prev.get("sga") or 0.0) / prev["revenue"]),
         "LVGI": idx((cur["current_liabilities"] + (cur.get("long_term_debt") or 0.0)) / ta,
                     (prev["current_liabilities"] + (prev.get("long_term_debt") or 0.0)) / pta),
         "TATA": (cur["net_income"] - cur["cfo"]) / ta}
    m = (-4.84 + 0.920 * v["DSRI"] + 0.528 * v["GMI"] + 0.404 * v["AQI"] + 0.892 * v["SGI"] + 0.115 * v["DEPI"]
         - 0.172 * v["SGAI"] + 4.679 * v["TATA"] - 0.327 * v["LVGI"])
    return {"m": m, "flag": m > -1.78, "components": v}


def sloan_accruals(cur: Statement, prev: Optional[Statement] = None) -> float:
    """(net income - operating cash flow) / average assets. High accruals predict lower future returns."""
    avg_assets = cur["total_assets"] if prev is None else (cur["total_assets"] + prev["total_assets"]) / 2
    return (cur["net_income"] - cur["cfo"]) / avg_assets

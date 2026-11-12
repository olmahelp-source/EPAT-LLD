"""Option-chain data handling: from raw quotes to implied vols, Greeks and the usual chain analytics.

``pricing`` and ``strategies`` in this package value contracts; this module handles the *data*: a validated container
for one underlying's quotes on one date, with per-quote implied volatility and Greeks, open-interest analytics
(put/call ratio, max pain, OI walls, gamma exposure), the volatility smile (skew, risk reversal, butterfly), the term
structure, strike selection by delta, and a data-quality report. A quote whose price violates no-arbitrage bounds gets
a NaN implied vol and is counted, rather than aborting the whole chain.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import DataError, MissingDataError, PlatformError, ValidationError
from ..quant.models import bs_greeks, bs_price, implied_vol
from .pricing import check_no_arbitrage

_KIND = {"c": "call", "call": "call", "ce": "call", "p": "put", "put": "put", "pe": "put"}
_NUMERIC = ("bid", "ask", "last", "volume", "open_interest")


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class OptionChain:
    """Quotes for one underlying as of one date. Build with :meth:`from_frame` (columns ``expiry, strike, kind`` plus any of
    ``bid, ask, last, volume, open_interest``)."""

    def __init__(self, underlying: str, spot: float, asof: Any, quotes: pd.DataFrame, r: float = 0.0, q: float = 0.0,
                 multiplier: float = 100.0) -> None:
        if not (isinstance(spot, (int, float)) and math.isfinite(spot) and spot > 0):
            raise ValidationError("spot must be a positive number", details={"spot": spot})
        self.underlying, self.spot, self.asof, self.r, self.q, self.multiplier = underlying, float(spot), _to_date(asof), r, q, multiplier
        self.iv_failures = 0
        self._df = self._prepare(quotes)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, underlying: str, spot: float, asof: Any, **kwargs: Any) -> "OptionChain":
        return cls(underlying, spot, asof, df, **kwargs)

    # ---- preparation --------------------------------------------------------------------------------------------------------
    def _prepare(self, quotes: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ("expiry", "strike", "kind") if c not in quotes]
        if missing:
            raise DataError(f"option chain is missing column(s) {missing}")
        if not any(c in quotes for c in ("bid", "ask", "last")):
            raise DataError("option chain needs at least one price column: bid/ask or last")
        df = quotes.copy()
        kinds = df["kind"].astype(str).str.strip().str.lower().map(_KIND)
        if kinds.isna().any():
            raise DataError("option kind must be call/put (c, p, ce, pe accepted)", details={"bad": sorted(set(quotes["kind"][kinds.isna()].astype(str)))[:5]})
        df["kind"] = kinds
        df["expiry"] = [_to_date(e) for e in df["expiry"]]
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        for c in _NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else (0.0 if c in ("volume", "open_interest") else np.nan)
        if df["strike"].isna().any() or (df["strike"] <= 0).any():
            raise DataError("strikes must be positive numbers")
        if df.duplicated(["expiry", "strike", "kind"]).any():
            raise DataError("duplicate quotes for the same expiry / strike / kind")
        df["days"] = [(e - self.asof).days for e in df["expiry"]]
        df = df[df["days"] > 0].sort_values(["expiry", "strike", "kind"]).reset_index(drop=True)      # expired rows carry no information
        if df.empty:
            raise MissingDataError("no unexpired quotes in the chain")
        df["T"] = df["days"] / 365.0
        two_sided = (df["bid"] > 0) & (df["ask"] >= df["bid"])
        df["mid"] = np.where(two_sided, (df["bid"] + df["ask"]) / 2.0, df["last"])                    # fall back to last when the book is one-sided
        df["spread_pct"] = np.where(two_sided & (df["mid"] > 0), (df["ask"] - df["bid"]) / df["mid"], np.nan)
        return self._analytics(df)

    def _analytics(self, df: pd.DataFrame) -> pd.DataFrame:
        ivs, greeks = [], {k: [] for k in ("delta", "gamma", "vega", "theta")}
        for row in df.itertuples(index=False):
            iv = float("nan")
            if np.isfinite(row.mid) and row.mid > 0:
                try:
                    iv = implied_vol(float(row.mid), self.spot, float(row.strike), float(row.T), self.r, row.kind, self.q)
                except PlatformError:                              # price outside no-arbitrage bounds: stale or bad quote
                    self.iv_failures += 1
            ivs.append(iv)
            g = bs_greeks(self.spot, float(row.strike), float(row.T), self.r, iv, row.kind, self.q) if np.isfinite(iv) and iv > 1e-5 else {}
            for k in greeks:
                greeks[k].append(g.get(k, float("nan")))
        return df.assign(iv=ivs, **greeks)

    # ---- access ---------------------------------------------------------------------------------------------------------------
    def frame(self, expiry: Optional[Any] = None) -> pd.DataFrame:
        """Quotes with ``mid, iv, delta, gamma, vega, theta`` (a copy)."""
        return self._slice(expiry).copy() if expiry is not None else self._df.copy()

    def expiries(self) -> List[date]:
        return sorted(self._df["expiry"].unique())

    def nearest_expiry(self, min_days: int = 0) -> date:
        ok = self._df[self._df["days"] >= min_days]
        if ok.empty:
            raise MissingDataError(f"no expiry at least {min_days} days out")
        return ok["expiry"].min()

    def _slice(self, expiry: Optional[Any]) -> pd.DataFrame:
        exp = _to_date(expiry) if expiry is not None else self.nearest_expiry()
        df = self._df[self._df["expiry"] == exp]
        if df.empty:
            raise MissingDataError(f"no quotes for expiry {exp}", details={"available": [str(e) for e in self.expiries()]})
        return df

    def strikes(self, expiry: Optional[Any] = None) -> List[float]:
        return sorted(self._slice(expiry)["strike"].unique())

    def forward(self, expiry: Optional[Any] = None) -> float:
        T = float(self._slice(expiry)["T"].iloc[0])
        return self.spot * math.exp((self.r - self.q) * T)

    def atm_strike(self, expiry: Optional[Any] = None) -> float:
        ks = np.array(self.strikes(expiry))
        return float(ks[np.argmin(np.abs(ks - self.forward(expiry)))])

    def _at(self, expiry: Optional[Any], strike: float, kind: str) -> Optional[pd.Series]:
        df = self._slice(expiry)
        hit = df[(df["strike"] == strike) & (df["kind"] == kind)]
        return hit.iloc[0] if len(hit) else None

    # ---- volatility -----------------------------------------------------------------------------------------------------------
    def atm_iv(self, expiry: Optional[Any] = None) -> float:
        k = self.atm_strike(expiry)
        ivs = [row["iv"] for row in (self._at(expiry, k, "call"), self._at(expiry, k, "put")) if row is not None and np.isfinite(row["iv"])]
        return float(np.mean(ivs)) if ivs else float("nan")

    def smile(self, expiry: Optional[Any] = None) -> pd.DataFrame:
        """Implied vol by strike from out-of-the-money options (puts below the forward, calls above): the liquid side."""
        df, fwd = self._slice(expiry), self.forward(expiry)
        otm = df[((df["kind"] == "put") & (df["strike"] < fwd)) | ((df["kind"] == "call") & (df["strike"] >= fwd))]
        otm = otm[np.isfinite(otm["iv"])]
        return pd.DataFrame({"strike": otm["strike"], "moneyness": np.log(otm["strike"] / fwd), "iv": otm["iv"], "delta": otm["delta"],
                             "kind": otm["kind"]}).reset_index(drop=True)

    def iv_surface(self) -> pd.DataFrame:
        """Strike x expiry table of out-of-the-money implied vols."""
        rows = [self.smile(e).assign(expiry=e) for e in self.expiries()]
        return pd.concat(rows).pivot_table(index="strike", columns="expiry", values="iv")

    def iv_at_delta(self, expiry: Optional[Any], kind: str, delta: float) -> float:
        """Implied vol of the ``kind`` option whose |delta| equals ``delta`` (linear interpolation across strikes)."""
        if kind not in ("call", "put") or not 0 < delta < 1:
            raise ValidationError("kind must be call/put and delta in (0, 1)")
        df = self._slice(expiry)
        df = df[(df["kind"] == kind) & np.isfinite(df["iv"]) & np.isfinite(df["delta"])]
        if len(df) < 2:
            return float("nan")
        d = df["delta"].abs().to_numpy()
        order = np.argsort(d)
        if not d[order][0] <= delta <= d[order][-1]:
            return float("nan")                                     # the chain does not reach that delta: do not extrapolate
        return float(np.interp(delta, d[order], df["iv"].to_numpy()[order]))

    def skew(self, expiry: Optional[Any] = None, delta: float = 0.25) -> Dict[str, float]:
        """Risk reversal (call wing minus put wing: negative = downside protection is bid) and butterfly (wings minus ATM)."""
        call, put, atm = self.iv_at_delta(expiry, "call", delta), self.iv_at_delta(expiry, "put", delta), self.atm_iv(expiry)
        return {"call_iv": call, "put_iv": put, "atm_iv": atm, "risk_reversal": call - put, "butterfly": (call + put) / 2.0 - atm}

    def term_structure(self) -> pd.DataFrame:
        rows = [{"expiry": e, "days": int(self._slice(e)["days"].iloc[0]), "atm_iv": self.atm_iv(e)} for e in self.expiries()]
        return pd.DataFrame(rows)

    def expected_move(self, expiry: Optional[Any] = None) -> Dict[str, float]:
        """Market-implied one-standard-deviation move to expiry: from the ATM straddle price and from ATM implied vol."""
        k, df = self.atm_strike(expiry), self._slice(expiry)
        c, p = self._at(expiry, k, "call"), self._at(expiry, k, "put")
        straddle = float(c["mid"] + p["mid"]) if c is not None and p is not None else float("nan")
        iv_move = self.spot * self.atm_iv(expiry) * math.sqrt(float(df["T"].iloc[0]))
        return {"strike": k, "straddle": straddle, "straddle_pct": straddle / self.spot, "iv_move": iv_move, "iv_move_pct": iv_move / self.spot}

    def strike_by_delta(self, expiry: Optional[Any], kind: str, target: float) -> float:
        """Listed strike whose |delta| is closest to ``target`` (e.g. the 0.30-delta put to sell)."""
        df = self._slice(expiry)
        df = df[(df["kind"] == kind) & np.isfinite(df["delta"])]
        if df.empty:
            raise MissingDataError(f"no {kind} deltas available for that expiry")
        return float(df.loc[(df["delta"].abs() - target).abs().idxmin(), "strike"])

    # ---- open interest ----------------------------------------------------------------------------------------------------------
    def put_call_ratio(self, expiry: Optional[Any] = None, by: str = "open_interest") -> float:
        if by not in ("open_interest", "volume"):
            raise ValidationError("by must be 'open_interest' or 'volume'")
        df = self._df if expiry is None else self._slice(expiry)
        calls, puts = df.loc[df["kind"] == "call", by].sum(), df.loc[df["kind"] == "put", by].sum()
        return float(puts / calls) if calls > 0 else float("nan")

    def max_pain(self, expiry: Optional[Any] = None) -> float:
        """Settlement price at which option holders collectively receive the least (writers pay out the least)."""
        df = self._slice(expiry)
        ks = np.array(sorted(df["strike"].unique()))
        calls, puts = df[df["kind"] == "call"], df[df["kind"] == "put"]
        pain = [(calls["open_interest"] * np.maximum(k - calls["strike"], 0)).sum() + (puts["open_interest"] * np.maximum(puts["strike"] - k, 0)).sum()
                for k in ks]
        return float(ks[int(np.argmin(pain))])

    def oi_walls(self, expiry: Optional[Any] = None, n: int = 3) -> Dict[str, List[tuple]]:
        """Largest call open interest above spot (often acts as resistance) and put open interest below (support)."""
        df = self._slice(expiry)
        res = df[(df["kind"] == "call") & (df["strike"] >= self.spot)].nlargest(n, "open_interest")
        sup = df[(df["kind"] == "put") & (df["strike"] <= self.spot)].nlargest(n, "open_interest")
        pack = lambda d: [(float(k), float(oi)) for k, oi in zip(d["strike"], d["open_interest"])]      # noqa: E731
        return {"resistance": pack(res), "support": pack(sup)}

    def gamma_exposure(self, expiry: Optional[Any] = None) -> pd.Series:
        """Dealer gamma exposure per strike in currency per 1% move, under the usual assumption that dealers are long
        calls and short puts. Positive total: dealers hedge against the move (dampening); negative: with it (amplifying)."""
        df = self._df if expiry is None else self._slice(expiry)
        sign = np.where(df["kind"] == "call", 1.0, -1.0)
        gex = sign * df["gamma"].fillna(0.0) * df["open_interest"] * self.multiplier * self.spot ** 2 * 0.01
        return gex.groupby(df["strike"]).sum()

    # ---- data quality -------------------------------------------------------------------------------------------------------------
    def quality_report(self, max_spread: float = 0.25) -> Dict[str, Any]:
        """What is wrong with the quotes: crossed or empty markets, wide spreads, prices that imply arbitrage."""
        df = self._df
        crossed = int(((df["bid"] > df["ask"]) & df["ask"].notna()).sum())
        no_price = int((~np.isfinite(df["mid"]) | (df["mid"] <= 0)).sum())
        wide = int((df["spread_pct"] > max_spread).sum())
        arbitrage: List[str] = []
        for e in self.expiries():
            calls = self._slice(e)
            calls = calls[(calls["kind"] == "call") & np.isfinite(calls["mid"])]
            if len(calls) >= 3:
                arbitrage += [f"{e}: {msg}" for msg in check_no_arbitrage(calls["strike"].tolist(), calls["mid"].tolist(), self.spot, self.r,
                                                                         float(calls["T"].iloc[0]), tol=1e-6)]
        issues = crossed + no_price + self.iv_failures + len(arbitrage)
        return {"quotes": len(df), "crossed": crossed, "no_price": no_price, "wide_spread": wide, "iv_failures": self.iv_failures,
                "arbitrage": arbitrage[:20], "ok": issues == 0}

    def summary(self, expiry: Optional[Any] = None) -> Dict[str, Any]:
        exp = _to_date(expiry) if expiry is not None else self.nearest_expiry()
        return {"underlying": self.underlying, "spot": self.spot, "asof": str(self.asof), "expiry": str(exp), "atm_strike": self.atm_strike(exp),
                "atm_iv": self.atm_iv(exp), "put_call_ratio": self.put_call_ratio(exp), "max_pain": self.max_pain(exp),
                **{f"skew_{k}": v for k, v in self.skew(exp).items() if k in ("risk_reversal", "butterfly")},
                "expected_move_pct": self.expected_move(exp)["iv_move_pct"]}


def iv_rank(iv_history: Sequence[float], current: Optional[float] = None) -> float:
    """Where current implied vol sits between the lowest and highest of the lookback, 0..1 (the history must not include
    values later than ``current``)."""
    h = np.asarray(iv_history, float)
    h = h[np.isfinite(h)]
    if len(h) < 2:
        return float("nan")
    cur = h[-1] if current is None else current
    return float((cur - h.min()) / (h.max() - h.min())) if h.max() > h.min() else 0.5


def iv_percentile(iv_history: Sequence[float], current: Optional[float] = None) -> float:
    """Share of the lookback's days with implied vol below the current level (less distorted by one spike than rank)."""
    h = np.asarray(iv_history, float)
    h = h[np.isfinite(h)]
    if len(h) < 2:
        return float("nan")
    cur = h[-1] if current is None else current
    return float((h < cur).mean())


def synthetic_chain(spot: float = 100.0, asof: Any = "2024-01-02", days: Sequence[int] = (30, 60, 90), strikes: Optional[Sequence[float]] = None,
                    atm_vol: float = 0.2, skew: float = -0.1, smile: float = 0.3, term_slope: float = 0.02, r: float = 0.02, spread: float = 0.02,
                    seed: Optional[int] = 0, underlying: str = "SYN") -> OptionChain:
    """A plausible chain for tests and demos: Black-Scholes prices off a skewed smile, a bid/ask spread, and open interest
    concentrated near the money. Says nothing about real markets."""
    rng = np.random.default_rng(seed)
    base = _to_date(asof)
    ks = np.asarray(strikes, float) if strikes is not None else np.round(spot * np.arange(0.8, 1.2001, 0.025), 2)
    rows: List[Mapping[str, Any]] = []
    for d in days:
        T = d / 365.0
        expiry = (pd.Timestamp(base) + pd.Timedelta(days=d)).date()
        for k in ks:
            m = math.log(k / spot)
            vol = max(atm_vol + term_slope * math.sqrt(T) + skew * m + smile * m * m, 0.03)
            oi_shape = math.exp(-0.5 * (m / 0.08) ** 2)
            for kind in ("call", "put"):
                mid = max(bs_price(spot, float(k), T, r, vol, kind), 0.01)
                half = max(mid * spread / 2.0, 0.005)
                rows.append({"expiry": expiry, "strike": float(k), "kind": kind, "bid": round(mid - half, 4), "ask": round(mid + half, 4),
                             "last": round(mid, 4), "volume": float(rng.integers(10, 500) * oi_shape + 1),
                             "open_interest": float(int(rng.integers(500, 5000) * oi_shape * (1.3 if (kind == "put") == (k < spot) else 0.7)) + 1)})
    return OptionChain(underlying, spot, base, pd.DataFrame(rows), r=r)

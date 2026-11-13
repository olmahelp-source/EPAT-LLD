"""Margin: how much capital a set of positions ties up, and whether the account can carry it.

Two methods, as in practice:

* **Rule-based** (:func:`rule_based_margin`): a fixed percentage per position, the way a cash-equity margin account
  works (50% to open, 25% / 30% to maintain), a fixed fraction of notional for futures, and the standard formula for
  uncovered short options. Each position is charged on its own, so a hedge earns no relief.
* **Scenario-based** (:func:`scenario_margin`): the approach of exchange risk arrays and portfolio margin. All
  positions on one underlying are revalued together under a grid of price and volatility shocks (options with
  Black-Scholes), and the requirement is the worst loss. A hedged book is charged for the risk that is left, not for
  each leg.

:func:`margin_report` turns either into what an account needs to know: initial and maintenance requirement, excess
liquidity, utilisation, and whether there is a margin call. :class:`MarginCheck` is the pre-trade check that refuses an
order which would take the requirement above equity, while always letting risk-reducing orders through.

The parameters are typical defaults, **not any exchange's or broker's actual schedule**. Real margin comes from the
clearing house or broker and changes with market conditions; load their numbers into :class:`MarginRules` /
:class:`ScanSettings` before relying on this for live sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.exceptions import ValidationError
from ..core.models import Order
from ..quant.models import bs_price
from .checks import Check, RiskContext

LINEAR = ("equity", "etf", "index", "fx", "crypto", "bond")


@dataclass(frozen=True)
class MarginPosition:
    """One position with the fields margin needs. ``quantity`` is signed; ``price`` is the instrument's own price."""
    symbol: str
    quantity: float
    price: float
    asset_class: str = "equity"
    multiplier: float = 1.0
    margin_pct: float = 1.0                     # futures: initial margin as a fraction of notional
    underlying: Optional[str] = None
    underlying_price: Optional[float] = None    # options
    strike: Optional[float] = None
    right: Optional[str] = None                 # "call" | "put"
    years_to_expiry: Optional[float] = None
    volatility: Optional[float] = None          # implied volatility, for scenario repricing

    def __post_init__(self) -> None:
        if not (math.isfinite(self.quantity) and math.isfinite(self.price)) or self.price < 0 or self.multiplier <= 0:
            raise ValidationError(f"{self.symbol}: quantity and price must be finite, price >= 0, multiplier > 0")
        if self.asset_class not in LINEAR + ("future", "option"):
            raise ValidationError(f"{self.symbol}: unknown asset class '{self.asset_class}'")
        if self.asset_class == "option":
            if self.right not in ("call", "put") or not self.strike or self.strike <= 0:
                raise ValidationError(f"{self.symbol}: an option needs right 'call'/'put' and a positive strike")
            if not self.underlying_price or self.underlying_price <= 0:
                raise ValidationError(f"{self.symbol}: an option needs the underlying's price")

    @property
    def group(self) -> str:
        """What the position's risk is driven by: its underlying, or itself."""
        return self.underlying or self.symbol

    @property
    def value(self) -> float:
        return self.quantity * self.price * self.multiplier

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], quantity: float, price: float, days_to_expiry: Optional[float] = None,
                  underlying_price: Optional[float] = None, volatility: Optional[float] = None) -> "MarginPosition":
        """Build from a contract record given as a mapping (e.g. ``ContractSpec.to_dict()``), so this module does not
        depend on the instrument master."""
        return cls(str(spec["symbol"]), quantity, price, str(spec.get("asset_class", "equity")), float(spec.get("multiplier", 1.0)),
                   float(spec.get("margin_pct", 1.0)), spec.get("underlying"), underlying_price,
                   float(spec["strike"]) if spec.get("strike") else None, spec.get("right"),
                   None if days_to_expiry is None else max(days_to_expiry, 0.0) / 365.0, volatility)


# ---- rule-based -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarginRules:
    equity_initial_long: float = 0.50
    equity_initial_short: float = 0.50
    equity_maintenance_long: float = 0.25
    equity_maintenance_short: float = 0.30
    future_maintenance_ratio: float = 0.80      # maintenance as a share of initial
    option_short_pct: float = 0.20              # uncovered short option: this share of the underlying's value ...
    option_short_min_pct: float = 0.10          # ... less the out-of-the-money amount, but never below this share

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not 0 < getattr(self, name) <= 1:
                raise ValidationError(f"margin rule '{name}' must be in (0, 1]")


def position_margin(pos: MarginPosition, rules: Optional[MarginRules] = None) -> Tuple[float, float]:
    """(initial, maintenance) requirement of one position on its own."""
    rules = rules or MarginRules()
    gross = abs(pos.value)
    if pos.quantity == 0:
        return 0.0, 0.0
    if pos.asset_class == "future":
        initial = gross * pos.margin_pct
        return initial, initial * rules.future_maintenance_ratio
    if pos.asset_class == "option":
        if pos.quantity > 0:
            return gross, gross                                    # a long option is paid for in full; it cannot be borrowed against
        u, k = float(pos.underlying_price), float(pos.strike)      # type: ignore[arg-type]
        otm = max(k - u, 0.0) if pos.right == "call" else max(u - k, 0.0)
        floor = rules.option_short_min_pct * (u if pos.right == "call" else k)
        per_unit = pos.price + max(rules.option_short_pct * u - otm, floor)
        req = per_unit * pos.multiplier * abs(pos.quantity)
        return req, req
    if pos.quantity > 0:
        return gross * rules.equity_initial_long, gross * rules.equity_maintenance_long
    return gross * rules.equity_initial_short, gross * rules.equity_maintenance_short


def rule_based_margin(positions: Sequence[MarginPosition], rules: Optional[MarginRules] = None) -> Dict[str, Any]:
    """Sum of stand-alone requirements. Conservative for hedged books: offsetting positions get no relief."""
    by_symbol = {p.symbol: position_margin(p, rules) for p in positions}
    return {"initial": sum(v[0] for v in by_symbol.values()), "maintenance": sum(v[1] for v in by_symbol.values()),
            "by_symbol": {s: {"initial": i, "maintenance": m} for s, (i, m) in by_symbol.items()}}


# ---- scenario-based ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanSettings:
    price_scan: float = 0.15            # largest ordinary price move considered, as a fraction of the underlying
    vol_scan: float = 0.25              # relative move in implied volatility, applied up and down
    steps: int = 3                      # price steps on each side: 1/3, 2/3 and the full scan range
    extreme_multiple: float = 3.0       # an extreme move of this many scan ranges ...
    extreme_cover: float = 0.35         # ... of whose loss this share is charged (deep out-of-the-money shorts)
    short_option_minimum: float = 0.005  # floor per short option, as a share of the underlying's value
    initial_ratio: float = 1.10         # initial requirement as a multiple of maintenance

    def __post_init__(self) -> None:
        if not 0 < self.price_scan < 1 or not 0 <= self.vol_scan < 1 or self.steps < 1:
            raise ValidationError("need 0 < price_scan < 1, 0 <= vol_scan < 1, steps >= 1")
        if self.extreme_multiple < 1 or not 0 < self.extreme_cover <= 1 or self.short_option_minimum < 0 or self.initial_ratio < 1:
            raise ValidationError("need extreme_multiple >= 1, 0 < extreme_cover <= 1, short_option_minimum >= 0, initial_ratio >= 1")

    def scenarios(self) -> List[Tuple[float, float, float]]:
        """(price move, volatility move, share of the loss charged) for every scenario."""
        out: List[Tuple[float, float, float]] = []
        vols = (self.vol_scan, -self.vol_scan) if self.vol_scan > 0 else (0.0,)
        for i in range(-self.steps, self.steps + 1):
            out += [(self.price_scan * i / self.steps, dv, 1.0) for dv in vols]
        ext = min(self.price_scan * self.extreme_multiple, 0.99)
        out += [(ext, 0.0, self.extreme_cover), (-ext, 0.0, self.extreme_cover)]
        return out


def _scenario_pnl(pos: MarginPosition, price_move: float, vol_move: float, rate: float) -> float:
    if pos.asset_class != "option":
        return pos.value * price_move
    if pos.volatility is None or pos.years_to_expiry is None:
        raise ValidationError(f"{pos.symbol}: scenario margin needs the option's volatility and time to expiry")
    u, k, t = float(pos.underlying_price), float(pos.strike), max(pos.years_to_expiry, 1e-6)   # type: ignore[arg-type]
    now = bs_price(u, k, t, rate, pos.volatility, pos.right or "call")
    shocked = bs_price(u * (1 + price_move), k, t, rate, max(pos.volatility * (1 + vol_move), 1e-4), pos.right or "call")
    return pos.quantity * pos.multiplier * (shocked - now)


def scenario_margin(positions: Sequence[MarginPosition], settings: Optional[ScanSettings] = None, rate: float = 0.0) -> Dict[str, Any]:
    """Worst-case loss per underlying over the scan grid, summed across underlyings.

    Within one underlying, gains on one leg offset losses on another in each scenario, so spreads, covered calls and
    delta hedges are charged for their net risk. *Across* underlyings no offset is given (no correlation is assumed),
    which is the conservative choice. Returns the requirement and, per underlying, the scenario that set it."""
    settings = settings or ScanSettings()
    scenarios = settings.scenarios()
    groups: Dict[str, List[MarginPosition]] = {}
    for p in positions:
        if p.quantity != 0:
            groups.setdefault(p.group, []).append(p)
    by_underlying: Dict[str, Dict[str, Any]] = {}
    for name, members in sorted(groups.items()):
        losses = [-sum(_scenario_pnl(p, dp, dv, rate) for p in members) * cover for dp, dv, cover in scenarios]
        worst = max(range(len(losses)), key=losses.__getitem__)
        scan_risk = max(losses[worst], 0.0)
        floor = float(sum((abs(p.quantity) * p.multiplier * float(p.underlying_price or 0.0) * settings.short_option_minimum
                           for p in members if p.asset_class == "option" and p.quantity < 0), 0.0))
        by_underlying[name] = {"margin": max(scan_risk, floor), "scan_risk": scan_risk, "short_option_minimum": floor,
                               "worst_price_move": scenarios[worst][0], "worst_vol_move": scenarios[worst][1], "positions": len(members)}
    maintenance = sum(v["margin"] for v in by_underlying.values())
    return {"maintenance": maintenance, "initial": maintenance * settings.initial_ratio, "by_underlying": by_underlying,
            "scenarios": len(scenarios)}


# ---- the account's view ---------------------------------------------------------------------------------------------------------
def margin_report(positions: Sequence[MarginPosition], equity: float, method: str = "rules", rules: Optional[MarginRules] = None,
                  settings: Optional[ScanSettings] = None, rate: float = 0.0) -> Dict[str, Any]:
    """Requirement against equity. ``excess_liquidity`` (equity minus maintenance) below zero is a margin call;
    ``available_funds`` (equity minus initial) is what new positions may still use."""
    if method not in ("rules", "scenario"):
        raise ValidationError("method must be 'rules' or 'scenario'")
    if not math.isfinite(equity):
        raise ValidationError("equity must be finite")
    req = rule_based_margin(positions, rules) if method == "rules" else scenario_margin(positions, settings, rate)
    gross = sum(abs(p.value) for p in positions if p.asset_class != "option")
    initial, maintenance = float(req["initial"]), float(req["maintenance"])
    return {"method": method, "equity": equity, "initial": initial, "maintenance": maintenance, "available_funds": equity - initial,
            "excess_liquidity": equity - maintenance, "utilisation": initial / equity if equity > 0 else math.inf,
            "margin_call": equity < maintenance, "shortfall": max(maintenance - equity, 0.0),
            "leverage": gross / equity if equity > 0 else math.inf, "detail": req}


def liquidation_price(quantity: float, entry_price: float, equity: float, maintenance_pct: float, multiplier: float = 1.0) -> float:
    """Price at which a single linear position, carried by ``equity``, hits its maintenance requirement.

    Solves ``equity + q m (P - entry) = maintenance_pct |q| m P``. Returns 0 for a long that cannot be called at any
    positive price and ``inf`` for a short that never is."""
    if quantity == 0 or multiplier <= 0 or not 0 < maintenance_pct < 1:
        raise ValidationError("need a non-zero quantity, multiplier > 0 and 0 < maintenance_pct < 1")
    q = quantity * multiplier
    denom = q - maintenance_pct * abs(q)
    price = (q * entry_price - equity) / denom
    if quantity > 0:
        return max(price, 0.0)
    return price if price > 0 else math.inf


Requirement = Callable[[Mapping[str, float], Mapping[str, float]], float]


def linear_requirement(rules: Optional[MarginRules] = None, multipliers: Optional[Mapping[str, float]] = None,
                       futures: Optional[Mapping[str, float]] = None) -> Requirement:
    """A requirement function for books of shares and futures: ``futures`` maps a symbol to its initial-margin fraction;
    everything else is treated as an equity."""
    rules, multipliers, futures = rules or MarginRules(), dict(multipliers or {}), dict(futures or {})

    def requirement(quantities: Mapping[str, float], prices: Mapping[str, float]) -> float:
        total = 0.0
        for symbol in sorted(quantities):
            qty = quantities[symbol]
            if qty == 0:
                continue
            if symbol not in prices:
                raise ValidationError(f"no price for {symbol}", details={"symbol": symbol})
            pos = MarginPosition(symbol, qty, prices[symbol], "future" if symbol in futures else "equity", multipliers.get(symbol, 1.0),
                                 futures.get(symbol, 1.0))
            total += position_margin(pos, rules)[0]
        return total
    return requirement


class MarginCheck(Check):
    """Pre-trade: reject an order that would take the initial requirement above ``max_utilisation`` of equity.

    An order that *lowers* the requirement is always allowed, even when the account is already over its limit: a margin
    check must never be the thing that stops a trader from reducing risk. If the requirement cannot be computed the
    order is rejected (fail closed)."""
    name = "margin"

    def __init__(self, requirement: Requirement, max_utilisation: float = 1.0) -> None:
        if max_utilisation <= 0:
            raise ValidationError("max_utilisation must be positive")
        self.requirement, self.max_utilisation = requirement, max_utilisation

    def check(self, order: Order, ctx: RiskContext) -> Optional[str]:
        before: Dict[str, float] = {}
        for symbol in set(ctx.positions) | set(ctx.pending):
            before[symbol] = ctx.positions.get(symbol, 0.0) + ctx.pending.get(symbol, 0.0)
        after = dict(before)
        after[order.symbol] = after.get(order.symbol, 0.0) + order.side.sign * order.quantity
        prices = dict(ctx.prices)
        if order.limit_price and order.symbol not in prices:
            prices[order.symbol] = order.limit_price
        try:
            need_after, need_before = self.requirement(after, prices), self.requirement(before, prices)
        except Exception as exc:  # noqa: BLE001 - an unknown requirement is a reason to refuse, not to guess
            return f"margin requirement could not be computed: {exc}"
        if need_after <= need_before:
            return None
        limit = ctx.equity * self.max_utilisation
        if need_after > limit:
            return f"margin requirement {need_after:,.0f} would exceed {limit:,.0f} ({self.max_utilisation:.0%} of equity)"
        return None

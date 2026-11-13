"""Strategy: the strategies the UI can run, lab options, instrument selection.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any

from ...core.exceptions import ValidationError
from ...relationships.manager import RelationshipManager
from ...selection.rankers import RANKERS
from ...selection.selector import UniverseSelector
from ...strategy.base import STRATEGIES
from ...users.manager import User

from ..common import NEEDS_DATA, SLOW
from ..session import ALERT_METRICS
from .quant import VOL_MODELS


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    @r("GET", "/api/strategies")
    def strategies(_u, _b):
        return [{"name": n, "params": STRATEGIES.get(n).PARAMS, "bounds": STRATEGIES.get(n).PARAM_BOUNDS,
                 "doc": (STRATEGIES.get(n).__doc__ or "").strip().split("\n")[0], "runnable": n not in NEEDS_DATA,
                 "slow": n in SLOW} for n in STRATEGIES.names()]

    @r("GET", "/api/lab/options")
    def options(_u, _b):
        return {"rankers": sorted(RANKERS), "vol_models": sorted(VOL_MODELS), "alert_metrics": list(ALERT_METRICS)}

    @r("POST", "/api/lab/select")
    def select(u, b):
        desk = research(u)
        md, symbols = app._data(b, u)
        if len(symbols) < 2:
            raise ValidationError("selection needs at least two instruments")
        sel = UniverseSelector().require_history(int(b.get("min_history", 130)))
        rankers = b.get("rankers") or [{"name": "momentum", "weight": 1.0}, {"name": "low_volatility", "weight": 0.5}]
        for rk in rankers:
            sel.rank_by(rk["name"], weight=float(rk.get("weight", 1.0)), ascending=bool(rk.get("ascending", False)))
        if b.get("max_correlation") not in (None, ""):
            sel.diversify(float(b["max_correlation"]))
        sel.top(int(b.get("top", 3)))
        res = desk.select_universe(sel, md)
        rel = RelationshipManager()
        summary = rel.build(md.closes(), corr_threshold=float(b.get("corr_threshold", 0.6)))
        weights = {s: 1.0 / len(res.symbols) for s in res.symbols} if res.symbols else {}
        corr = md.returns().corr()
        scores = res.scores.reset_index().rename(columns={"index": "symbol"})
        return {"selected": res.symbols, "scores": scores.to_dict("records"), "rejected": [{"symbol": s, "reason": why} for s, why in res.rejected.items()],
                "relationships": summary, "groups": [sorted(g) for g in rel.groups()],
                "pairs": [{"a": x.a, "b": x.b, "stat": x.weight, **dict(x.meta)} for x in rel.pairs()][:10],
                "leads": [{"leader": x.a, "follower": x.b, "lag": x.meta.get("lag"), "corr": x.weight} for x in rel.relations("leads")][:10],
                "concentration": rel.concentration(weights) if weights else None,
                "correlation": {s: corr.loc[s].to_dict() for s in corr.index}, "errors": rel.errors[:10]}

"""Application: the platform map (job groups, desks, dependency rules).

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any, List

from ...architecture import check_architecture, GROUPS


def register(app: Any) -> None:
    r, P = app.route, app.platform
    violations: List[List[str]] = []                              # the source does not change while the server runs: scan once

    @r("GET", "/api/structure")
    def structure(u, _b):
        if not violations:
            violations.append(check_architecture())
        return {"role": u.role, "desks": sorted(P.desks(u)),
                "groups": [{"key": g.key, "title": g.title, "job": g.job, "packages": list(g.packages), "roles": [x.value for x in g.roles]} for g in GROUPS],
                "violations": violations[0]}

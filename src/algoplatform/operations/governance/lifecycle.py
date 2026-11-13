"""Strategy lifecycle: the controlled path from an idea to live capital.

    RESEARCH  --submit + approve-->  PAPER  --submit + approve-->  LIVE  --> RETIRED
                                       ^------------- demote ---------'

Rules enforced here (and audited):

* **Evidence.** A promotion request must carry a scorecard (``analytics.evaluate_strategy``) that passes: a backtest
  scorecard to enter PAPER, a paper-trading scorecard to enter LIVE.
* **Time in paper.** A strategy must have been in PAPER for ``min_paper_days`` before it can be proposed for LIVE.
* **Four eyes.** The person who approves is never the person who submitted, and never the strategy's owner. An admin
  is not exempt: this is the control that stops one person putting untested logic into production.
* **Live gate.** ``require_live`` is what the trading layer calls before letting a strategy trade real money.

Demotion and retirement are deliberately easy (one authorised person, a reason): taking risk off must never be blocked
by process.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ...validation.analytics.evaluation import StrategyVerdict
from ...foundation.core.enums import Permission
from ...foundation.core.exceptions import GovernanceError
from ...foundation.core.logging import AuditLog
from ...access.users.manager import User, UserManager
from ...foundation.utils.helpers import json_safe


class Stage(str, Enum):
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"
    RETIRED = "RETIRED"


_NEXT = {Stage.RESEARCH: Stage.PAPER, Stage.PAPER: Stage.LIVE}


@dataclass(frozen=True)
class PromotionPolicy:
    min_paper_days: int = 20
    require_passing_scorecard: bool = True


@dataclass
class PromotionRequest:
    target: Stage
    submitted_by: str
    submitted_at: datetime
    evidence: Dict[str, Any]
    note: str = ""


@dataclass
class StrategyRecord:
    strategy_id: str
    owner: str
    description: Dict[str, Any] = field(default_factory=dict)
    stage: Stage = Stage.RESEARCH
    stage_since: Optional[datetime] = None
    pending: Optional[PromotionRequest] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class StrategyLifecycle:
    def __init__(self, audit: Optional[AuditLog] = None, policy: Optional[PromotionPolicy] = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.audit, self.policy, self._now = audit or AuditLog(), policy or PromotionPolicy(), now
        self._records: Dict[str, StrategyRecord] = {}
        self._lock = threading.RLock()

    # ---- helpers ----------------------------------------------------------------------------------------------------------
    def get(self, strategy_id: str) -> StrategyRecord:
        try:
            return self._records[strategy_id]
        except KeyError:
            raise GovernanceError(f"strategy '{strategy_id}' is not registered", code="UNKNOWN_STRATEGY") from None

    def _move(self, rec: StrategyRecord, to: Stage, actor: str, reason: str, action: str) -> None:
        now = self._now()
        rec.history.append({"ts": now.isoformat(), "actor": actor, "action": action, "from": rec.stage.value, "to": to.value, "reason": reason})
        rec.stage, rec.stage_since, rec.pending = to, now, None
        self.audit.record(actor, f"strategy.{action}", strategy=rec.strategy_id, to=to.value, reason=reason)

    # ---- lifecycle ----------------------------------------------------------------------------------------------------------
    def register(self, user: User, strategy_id: str, description: Optional[Dict[str, Any]] = None) -> StrategyRecord:
        UserManager.authorize(user, Permission.MANAGE_STRATEGY)
        with self._lock:
            if strategy_id in self._records:
                raise GovernanceError(f"strategy '{strategy_id}' is already registered", code="DUPLICATE_STRATEGY")
            rec = StrategyRecord(strategy_id, user.username, dict(description or {}), stage_since=self._now())
            self._records[strategy_id] = rec
        self.audit.record(user.username, "strategy.register", strategy=strategy_id)
        return rec

    def submit(self, user: User, strategy_id: str, verdict: StrategyVerdict, note: str = "") -> PromotionRequest:
        """Ask for promotion to the next stage, attaching the scorecard that justifies it."""
        UserManager.authorize(user, Permission.SUBMIT_STRATEGY)
        with self._lock:
            rec = self.get(strategy_id)
            target = _NEXT.get(rec.stage)
            if target is None:
                raise GovernanceError(f"a {rec.stage.value} strategy cannot be promoted", code="NO_NEXT_STAGE")
            if rec.pending is not None:
                raise GovernanceError("a promotion request is already pending", code="REQUEST_PENDING")
            if not isinstance(verdict, StrategyVerdict):
                raise GovernanceError("promotion needs a StrategyVerdict as evidence", code="NO_EVIDENCE")
            if self.policy.require_passing_scorecard and not verdict.passed:
                raise GovernanceError("the scorecard does not pass, so the strategy cannot be proposed", code="EVIDENCE_FAILS",
                                      details={"failures": verdict.failures, "grade": verdict.grade})
            if target is Stage.LIVE:
                days = (self._now() - rec.stage_since).days if rec.stage_since else 0
                if days < self.policy.min_paper_days:
                    raise GovernanceError(f"{days} days in paper trading; {self.policy.min_paper_days} required before live",
                                          code="PAPER_PERIOD", details={"days": days})
            rec.pending = PromotionRequest(target, user.username, self._now(), verdict.to_dict(), note)
        self.audit.record(user.username, "strategy.submit", strategy=strategy_id, target=target.value, grade=verdict.grade)
        return rec.pending

    def approve(self, approver: User, strategy_id: str, comment: str = "") -> StrategyRecord:
        UserManager.authorize(approver, Permission.APPROVE_STRATEGY)
        with self._lock:
            rec = self.get(strategy_id)
            if rec.pending is None:
                raise GovernanceError("nothing is pending approval", code="NOTHING_PENDING")
            if approver.username in (rec.pending.submitted_by, rec.owner):
                self.audit.record(approver.username, "strategy.self_approval_blocked", strategy=strategy_id)
                raise GovernanceError("the submitter or owner cannot approve their own strategy; a second person must",
                                      code="SELF_APPROVAL")
            self._move(rec, rec.pending.target, approver.username, comment, "approve")
        return rec

    def reject(self, approver: User, strategy_id: str, reason: str) -> StrategyRecord:
        UserManager.authorize(approver, Permission.APPROVE_STRATEGY)
        if not reason.strip():
            raise GovernanceError("a rejection needs a reason", code="REASON_REQUIRED")
        with self._lock:
            rec = self.get(strategy_id)
            if rec.pending is None:
                raise GovernanceError("nothing is pending approval", code="NOTHING_PENDING")
            rec.history.append({"ts": self._now().isoformat(), "actor": approver.username, "action": "reject", "from": rec.stage.value,
                                "to": rec.stage.value, "reason": reason})
            rec.pending = None
        self.audit.record(approver.username, "strategy.reject", strategy=strategy_id, reason=reason)
        return rec

    def demote(self, user: User, strategy_id: str, reason: str) -> StrategyRecord:
        """LIVE -> PAPER at once (risk breach, live results diverging from the backtest)."""
        if not (user.has(Permission.APPROVE_STRATEGY) or user.has(Permission.MANAGE_RISK)):
            UserManager.authorize(user, Permission.APPROVE_STRATEGY)
        if not reason.strip():
            raise GovernanceError("a demotion needs a reason", code="REASON_REQUIRED")
        with self._lock:
            rec = self.get(strategy_id)
            if rec.stage is not Stage.LIVE:
                raise GovernanceError("only a LIVE strategy can be demoted", code="NOT_LIVE")
            self._move(rec, Stage.PAPER, user.username, reason, "demote")
        return rec

    def retire(self, user: User, strategy_id: str, reason: str) -> StrategyRecord:
        with self._lock:
            rec = self.get(strategy_id)
            if user.username != rec.owner and not (user.has(Permission.APPROVE_STRATEGY) or user.has(Permission.MANAGE_RISK)):
                raise GovernanceError("only the owner, an approver or a risk manager can retire a strategy", code="FORBIDDEN")
            self._move(rec, Stage.RETIRED, user.username, reason, "retire")
        return rec

    # ---- queries ---------------------------------------------------------------------------------------------------------------
    def stage(self, strategy_id: str) -> Stage:
        return self.get(strategy_id).stage

    def can_trade_live(self, strategy_id: str) -> bool:
        rec = self._records.get(strategy_id)
        return rec is not None and rec.stage is Stage.LIVE

    def require_live(self, strategy_id: str) -> None:
        """Raise unless the strategy has been approved for live trading. Unregistered strategies are refused too."""
        if not self.can_trade_live(strategy_id):
            stage = self._records[strategy_id].stage.value if strategy_id in self._records else "UNREGISTERED"
            raise GovernanceError(f"strategy '{strategy_id}' is {stage}, not approved for live trading", code="NOT_APPROVED_FOR_LIVE",
                                  details={"stage": stage})

    def pending(self) -> List[StrategyRecord]:
        return [r for r in self._records.values() if r.pending is not None]

    def records(self) -> List[StrategyRecord]:
        return list(self._records.values())

    # ---- persistence ---------------------------------------------------------------------------------------------------------
    def export_state(self) -> List[Dict[str, Any]]:
        """Every strategy record as plain data: stage, when it got there, the pending request with its evidence, and
        the full history of decisions."""
        out = []
        with self._lock:
            for r in self._records.values():
                p = r.pending
                out.append(json_safe({
                    "strategy_id": r.strategy_id, "owner": r.owner, "description": r.description, "stage": r.stage.value,
                    "stage_since": r.stage_since, "history": r.history,
                    "pending": None if p is None else {"target": p.target.value, "submitted_by": p.submitted_by, "submitted_at": p.submitted_at,
                                                       "evidence": p.evidence, "note": p.note}}))
        return out

    def import_state(self, records: List[Dict[str, Any]]) -> int:
        """Restore records written by :meth:`export_state`. Approvals already given stay given: a strategy that was LIVE
        before the restart is LIVE after it, and a pending request is still pending for a second person to decide."""
        with self._lock:
            for d in records:
                p = d.get("pending")
                pending = None if not p else PromotionRequest(Stage(p["target"]), p["submitted_by"], datetime.fromisoformat(p["submitted_at"]),
                                                              dict(p.get("evidence") or {}), p.get("note", ""))
                since = d.get("stage_since")
                self._records[d["strategy_id"]] = StrategyRecord(d["strategy_id"], d["owner"], dict(d.get("description") or {}), Stage(d["stage"]),
                                                                 datetime.fromisoformat(since) if since else None, pending,
                                                                 list(d.get("history") or []))
        return len(records)

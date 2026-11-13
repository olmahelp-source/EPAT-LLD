"""The notification record passed between alert rules, the notifier and channels."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ...foundation.core.enums import Severity
from ...foundation.core.exceptions import ValidationError


@dataclass(frozen=True)
class Notification:
    title: str
    body: str = ""
    severity: Severity = Severity.INFO
    topic: str = "general"                       # dotted, e.g. "risk.kill_switch", "order.fill", "system.health"
    key: Optional[str] = None                    # notifications sharing a key within the dedupe window are sent once
    user_id: Optional[str] = None                # None = a platform-wide notification; otherwise private to that user
    data: Mapping[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValidationError("a notification needs a title")
        if not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity(str(self.severity).upper()))

    @property
    def dedupe_key(self) -> str:
        return self.key or f"{self.topic}|{self.title}|{self.user_id or ''}"

    def text(self) -> str:
        """Plain-text rendering used by chat and e-mail channels."""
        head = f"[{self.severity.value}] {self.title}"
        return head if not self.body else f"{head}\n{self.body}"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "ts": self.ts, "severity": self.severity.value, "topic": self.topic, "title": self.title,
                "body": self.body, "user_id": self.user_id, "key": self.key, "data": dict(self.data)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Notification":
        return cls(d["title"], d.get("body", ""), Severity(d.get("severity", "INFO")), d.get("topic", "general"), d.get("key"),
                   d.get("user_id"), dict(d.get("data") or {}), float(d.get("ts") or time.time()), d.get("id") or uuid.uuid4().hex[:12])

"""Logging setup and an append-only audit trail."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Module logger under the ``algoplatform`` namespace."""
    if not name.startswith("algoplatform"):
        name = f"algoplatform.{name}"
    return logging.getLogger(name)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Idempotent logging configuration for the whole package."""
    global _CONFIGURED
    root = logging.getLogger("algoplatform")
    if _CONFIGURED:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    if json_format:
        class _Json(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                   "level": record.levelname, "logger": record.name,
                                   "msg": record.getMessage()})
        handler.setFormatter(_Json())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _CONFIGURED = True


class AuditLog:
    """Append-only, thread-safe record of who did what.

    Entries are kept in memory and optionally mirrored to a JSON-lines file so
    state and decisions can be reconstructed after a restart.
    """

    def __init__(self, path: Optional[str] = None, max_memory: int = 100_000) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._path = path
        self._max = max_memory

    def record(self, actor: str, action: str, **details: Any) -> Dict[str, Any]:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "actor": actor,
                 "action": action, "details": details}
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                del self._entries[: len(self._entries) - self._max]
            if self._path:
                try:
                    with open(self._path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, default=str) + "\n")
                except OSError as exc:  # audit failure must never crash trading
                    get_logger(__name__).error("audit write failed: %r", exc)
        return entry

    def entries(self, action: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._entries if action is None or e["action"] == action]

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # an EMPTY log must still be truthy, otherwise `audit or AuditLog()` silently swaps a
        # shared log for a private one and entries never reach the real trail
        return True

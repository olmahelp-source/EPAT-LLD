"""Model registry: which model is in production, what it was trained on, and how it scored.

A model that trades is a production artefact like any other. The registry gives each fitted model a version, a
fingerprint of its training data, its out-of-sample metrics and a stage. Only one version per name is in production;
promoting a new one archives the old one (which stays available for rollback).
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import GovernanceError, ModelError

STAGES = ("candidate", "staging", "production", "archived")


def fingerprint(*arrays: Any) -> str:
    """Stable hash of training inputs: the same data always gives the same fingerprint, any change gives another."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


@dataclass
class ModelVersion:
    name: str
    version: int
    model: Any
    stage: str = "candidate"
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    notes: str = ""


class ModelRegistry:
    def __init__(self) -> None:
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._lock = threading.RLock()

    def register(self, name: str, model: Any, metrics: Optional[Mapping[str, float]] = None, params: Optional[Mapping[str, Any]] = None,
                 data: Any = None, created_by: str = "", notes: str = "") -> ModelVersion:
        """Record a fitted model as a new *candidate* version. ``data`` (array or tuple of arrays) is fingerprinted, not stored."""
        if not name:
            raise ModelError("a model needs a name")
        fp = "" if data is None else fingerprint(*(data if isinstance(data, (tuple, list)) else (data,)))
        with self._lock:
            versions = self._versions.setdefault(name, [])
            mv = ModelVersion(name, len(versions) + 1, model, metrics=dict(metrics or {}), params=dict(params or {}), data_fingerprint=fp,
                              created_by=created_by, notes=notes)
            versions.append(mv)
        return mv

    def versions(self, name: str) -> List[ModelVersion]:
        if name not in self._versions:
            raise ModelError(f"unknown model '{name}'", details={"known": sorted(self._versions)})
        return list(self._versions[name])

    def promote(self, name: str, version: int, stage: str) -> ModelVersion:
        """Move a version to ``stage``. Entering production requires recorded metrics (no evidence, no production) and
        archives the version currently there."""
        if stage not in STAGES:
            raise GovernanceError(f"unknown stage '{stage}'", details={"valid": list(STAGES)})
        with self._lock:
            match = [v for v in self.versions(name) if v.version == version]
            if not match:
                raise ModelError(f"model '{name}' has no version {version}")
            mv = match[0]
            if stage == "production":
                if not mv.metrics:
                    raise GovernanceError("a model without recorded out-of-sample metrics cannot go to production", code="NO_EVIDENCE")
                for other in self._versions[name]:
                    if other.stage == "production" and other is not mv:
                        other.stage = "archived"
            mv.stage = stage
        return mv

    def get(self, name: str, stage: str = "production") -> ModelVersion:
        """The latest version in ``stage``."""
        hits = [v for v in self.versions(name) if v.stage == stage]
        if not hits:
            raise ModelError(f"model '{name}' has no version in stage '{stage}'", code="NO_MODEL_IN_STAGE")
        return hits[-1]

    def rollback(self, name: str) -> ModelVersion:
        """Put the most recently archived version back into production."""
        archived = [v for v in self.versions(name) if v.stage == "archived" and v.metrics]
        if not archived:
            raise GovernanceError(f"model '{name}' has no archived version to roll back to", code="NO_ROLLBACK")
        return self.promote(name, archived[-1].version, "production")

    def data_changed(self, name: str, data: Any, stage: str = "production") -> bool:
        """True when ``data`` differs from what the staged model was trained on (a prompt to retrain or investigate)."""
        fp = fingerprint(*(data if isinstance(data, (tuple, list)) else (data,)))
        return self.get(name, stage).data_fingerprint != fp

    def to_frame(self) -> pd.DataFrame:
        rows = [{"name": v.name, "version": v.version, "stage": v.stage, "created_at": v.created_at, "created_by": v.created_by,
                 "data_fingerprint": v.data_fingerprint, **{f"metric_{k}": m for k, m in v.metrics.items()}}
                for vs in self._versions.values() for v in vs]
        return pd.DataFrame(rows)

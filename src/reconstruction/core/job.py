from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
import uuid

from .ir import ReconArtifacts

# --------- helpers ---------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _gen_job_id() -> str:
    # time-sortable-ish ID if ulid not available
    try:
        import ulid  # type: ignore
        return ulid.new().str
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]

# --------- status ---------
class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    SUCCEEDED  = "succeeded"
    FAILED     = "failed"
    CANCELLED  = "cancelled"

# --------- spec (immutable) ---------
@dataclass(frozen=True)
class JobSpec:
    adapter: str
    method: str
    pipeline: str
    input_path: str
    workdir: str
    outdir: str
    name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

# --------- record (mutable lifecycle) ---------
@dataclass
class JobRecord:
    id: str
    spec: JobSpec

    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = field(default_factory=_now_iso)

    scene_id: Optional[str] = None
    artifacts: Optional[ReconArtifacts] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    # lightweight event trail
    events: List[Dict[str, Any]] = field(default_factory=list)

    # ---- lifecycle helpers ----
    @classmethod
    def new(cls, spec: JobSpec, job_id: Optional[str] = None) -> "JobRecord":
        return cls(id=job_id or _gen_job_id(), spec=spec)

    def _bump(self, msg: str, level: str = "info") -> None:
        self.updated_at = _now_iso()
        self.events.append({"ts": self.updated_at, "level": level, "msg": msg})

    def start(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = _now_iso()
        self._bump("job started")

    def complete(self, artifacts: Optional[ReconArtifacts] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        self.status = JobStatus.SUCCEEDED
        self.completed_at = _now_iso()
        if artifacts is not None:
            self.artifacts = artifacts
        if metrics:
            self.metrics.update(metrics)
        self._bump("job completed")

    def fail(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = _now_iso()
        self._bump(f"job failed: {error}", level="error")

    def cancel(self, reason: str = "user") -> None:
        self.status = JobStatus.CANCELLED
        self.completed_at = _now_iso()
        self._bump(f"job cancelled: {reason}", level="warn")

    # ---- derived props ----
    @property
    def duration_seconds(self) -> Optional[float]:
        if not self.started_at or not self.completed_at:
            return None
        s = datetime.fromisoformat(self.started_at)
        e = datetime.fromisoformat(self.completed_at)
        return (e - s).total_seconds()

    # ---- (de)serialization ----
    def to_json_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "spec": {
                **asdict(self.spec),
            },
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "scene_id": self.scene_id,
            "artifacts": asdict(self.artifacts) if self.artifacts else None,
            "metrics": self.metrics,
            "error": self.error,
            "tags": self.tags,
            "events": list(self.events),
        }
        return d

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "JobRecord":
        spec = JobSpec(**d["spec"])
        rec = cls(
            id=d["id"], spec=spec,
            status=JobStatus(d["status"]),
            created_at=d["created_at"],
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            updated_at=d.get("updated_at", d["created_at"]),
            scene_id=d.get("scene_id"),
            artifacts=ReconArtifacts(**d["artifacts"]) if d.get("artifacts") else None,
            metrics=d.get("metrics", {}),
            error=d.get("error"),
            tags=d.get("tags", {}),
            events=d.get("events", []),
        )
        return rec

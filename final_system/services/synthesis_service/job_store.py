# services/synthesis_service/job_store.py
#
# In-memory потокобезопасное хранилище джобов синтеза.

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    queued    = "queued"
    running   = "running"
    done      = "done"
    failed    = "failed"
    cancelled = "cancelled"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id:        str
    status:        JobStatus = JobStatus.queued
    synth_path:    Optional[str] = None      # путь относительно data_root
    model_id:      Optional[str] = None
    dp_report:     Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at:    datetime = field(default_factory=_now)
    started_at:    Optional[datetime] = None
    finished_at:   Optional[datetime] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            for k, v in kwargs.items():
                setattr(rec, k, v)
            return rec


# Глобальный синглтон
job_store = JobStore()

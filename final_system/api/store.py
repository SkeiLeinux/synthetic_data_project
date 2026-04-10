# api/store.py
#
# Потокобезопасное in-memory хранилище состояния запусков.
# MVP-реализация: данные живут в памяти процесса.
# В production заменяется на Redis или PostgreSQL.

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class RunStatus(str, Enum):
    queued    = "queued"
    running   = "running"
    completed = "completed"
    failed    = "failed"
    cancelled = "cancelled"


class EvalStatus(str, Enum):
    queued    = "queued"
    running   = "running"
    completed = "completed"
    failed    = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunRecord:
    run_id:        str
    dataset_name:  str
    config_name:   str
    status:        RunStatus = RunStatus.queued
    verdict:       Optional[str] = None
    save_model:    bool = False
    webhook_url:   Optional[str] = None
    n_synth_rows:  Optional[int] = None
    model_id:      Optional[str] = None
    synth_rows:    Optional[int] = None
    synth_path:    Optional[str] = None   # абсолютный путь к CSV синтетики
    report_path:   Optional[str] = None   # абсолютный путь к JSON отчёту
    report:        Optional[Dict[str, Any]] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at:    datetime = field(default_factory=_now)
    started_at:    Optional[datetime] = None
    finished_at:   Optional[datetime] = None

    @property
    def duration_sec(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


@dataclass
class EvalRecord:
    evaluation_id: str
    eval_type:     str   # "privacy" | "utility"
    status:        EvalStatus = EvalStatus.queued
    report:        Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at:    datetime = field(default_factory=_now)
    finished_at:   Optional[datetime] = None


class RunStore:
    """Потокобезопасный реестр запусков пайплайна."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.run_id] = record

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def update(self, run_id: str, **kwargs) -> Optional[RunRecord]:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return None
            for k, v in kwargs.items():
                setattr(rec, k, v)
            return rec

    def list(
        self,
        status: Optional[str] = None,
        verdict: Optional[str] = None,
        dataset_name: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[List[RunRecord], int]:
        with self._lock:
            items = list(self._runs.values())

        # Фильтрация
        if status:
            items = [r for r in items if r.status == status]
        if verdict:
            items = [r for r in items if r.verdict == verdict]
        if dataset_name:
            items = [r for r in items if r.dataset_name == dataset_name]

        # Сортировка по дате создания (новые первые)
        items.sort(key=lambda r: r.created_at, reverse=True)

        total = len(items)
        offset = (page - 1) * per_page
        return items[offset: offset + per_page], total

    def delete(self, run_id: str) -> bool:
        with self._lock:
            if run_id not in self._runs:
                return False
            del self._runs[run_id]
            return True


class EvalStore:
    """Потокобезопасный реестр оценок."""

    def __init__(self) -> None:
        self._evals: Dict[str, EvalRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: EvalRecord) -> None:
        with self._lock:
            self._evals[record.evaluation_id] = record

    def get(self, evaluation_id: str) -> Optional[EvalRecord]:
        with self._lock:
            return self._evals.get(evaluation_id)

    def update(self, evaluation_id: str, **kwargs) -> Optional[EvalRecord]:
        with self._lock:
            rec = self._evals.get(evaluation_id)
            if rec is None:
                return None
            for k, v in kwargs.items():
                setattr(rec, k, v)
            return rec


# Глобальные синглтоны — инициализируются при импорте
run_store  = RunStore()
eval_store = EvalStore()

# api/store.py
#
# Redis-backed хранилище состояния запусков.
# RunRecord сериализуется в JSON и хранится под ключом run:{run_id}.
# Сортировка по дате создания поддерживается через ZSET runs:by_created.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

import redis


class RunStatus(str, Enum):
    queued    = "queued"
    running   = "running"
    completed = "completed"
    failed    = "failed"
    cancelled = "cancelled"


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


# ─── Serialization helpers ────────────────────────────────────────────────────

def _serialize_run(record: RunRecord) -> str:
    return json.dumps({
        "run_id":          record.run_id,
        "dataset_name":    record.dataset_name,
        "config_name":     record.config_name,
        "status":          record.status.value,
        "verdict":         record.verdict,
        "save_model":      record.save_model,
        "webhook_url":     record.webhook_url,
        "n_synth_rows":    record.n_synth_rows,
        "model_id":        record.model_id,
        "synth_rows":      record.synth_rows,
        "synth_path":      record.synth_path,
        "report_path":     record.report_path,
        "report":          record.report,
        "config_snapshot": record.config_snapshot,
        "error_message":   record.error_message,
        "created_at":      record.created_at.isoformat(),
        "started_at":      record.started_at.isoformat() if record.started_at else None,
        "finished_at":     record.finished_at.isoformat() if record.finished_at else None,
    })


def _deserialize_run(raw: str) -> RunRecord:
    d = json.loads(raw)
    return RunRecord(
        run_id=         d["run_id"],
        dataset_name=   d["dataset_name"],
        config_name=    d["config_name"],
        status=         RunStatus(d["status"]),
        verdict=        d.get("verdict"),
        save_model=     d.get("save_model", False),
        webhook_url=    d.get("webhook_url"),
        n_synth_rows=   d.get("n_synth_rows"),
        model_id=       d.get("model_id"),
        synth_rows=     d.get("synth_rows"),
        synth_path=     d.get("synth_path"),
        report_path=    d.get("report_path"),
        report=         d.get("report"),
        config_snapshot=d.get("config_snapshot"),
        error_message=  d.get("error_message"),
        created_at=     datetime.fromisoformat(d["created_at"]),
        started_at=     datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None,
        finished_at=    datetime.fromisoformat(d["finished_at"]) if d.get("finished_at") else None,
    )


_RUN_KEY_FMT = "run:{}"
_RUN_INDEX   = "runs:by_created"


class RunStore:
    """Redis-backed реестр запусков пайплайна.

    Каждый RunRecord хранится как JSON-строка под ключом run:{run_id}.
    ZSET runs:by_created (score = unix-timestamp created_at) используется
    для итерации по всем записям в порядке убывания даты создания.
    """

    def __init__(self, redis_url: str) -> None:
        self._r: redis.Redis = redis.from_url(redis_url, decode_responses=True)

    # ── public interface ──────────────────────────────────────────────────────

    def add(self, record: RunRecord) -> None:
        key = _RUN_KEY_FMT.format(record.run_id)
        score = record.created_at.timestamp()
        pipe = self._r.pipeline()
        pipe.set(key, _serialize_run(record))
        pipe.zadd(_RUN_INDEX, {record.run_id: score})
        pipe.execute()

    def get(self, run_id: str) -> Optional[RunRecord]:
        raw = self._r.get(_RUN_KEY_FMT.format(run_id))
        return _deserialize_run(raw) if raw is not None else None

    def update(self, run_id: str, **kwargs) -> Optional[RunRecord]:
        """Атомарное обновление через оптимистичную блокировку (WATCH/MULTI/EXEC).

        При конкурентном изменении повторяет попытку до 3 раз.
        """
        key = _RUN_KEY_FMT.format(run_id)
        for _ in range(3):
            with self._r.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    rec = _deserialize_run(raw)
                    for k, v in kwargs.items():
                        setattr(rec, k, v)
                    pipe.multi()
                    pipe.set(key, _serialize_run(rec))
                    pipe.execute()
                    return rec
                except redis.WatchError:
                    continue
        return None  # все попытки исчерпаны

    def expire(self, run_id: str, seconds: int) -> None:
        """Устанавливает TTL на запись. По истечении Redis удалит её автоматически."""
        self._r.expire(_RUN_KEY_FMT.format(run_id), seconds)

    def delete(self, run_id: str) -> bool:
        key = _RUN_KEY_FMT.format(run_id)
        pipe = self._r.pipeline()
        pipe.delete(key)
        pipe.zrem(_RUN_INDEX, run_id)
        results = pipe.execute()
        return bool(results[0])

    def list(
        self,
        status: Optional[str] = None,
        verdict: Optional[str] = None,
        dataset_name: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[List[RunRecord], int]:
        # Извлекаем все run_id из ZSET в порядке убывания даты создания
        run_ids: List[str] = self._r.zrevrange(_RUN_INDEX, 0, -1)

        records: List[RunRecord] = []
        for run_id in run_ids:
            raw = self._r.get(_RUN_KEY_FMT.format(run_id))
            if raw is not None:
                records.append(_deserialize_run(raw))

        # Фильтрация
        if status:
            records = [r for r in records if r.status == status]
        if verdict:
            records = [r for r in records if r.verdict == verdict]
        if dataset_name:
            records = [r for r in records if r.dataset_name == dataset_name]

        total = len(records)
        offset = (page - 1) * per_page
        return records[offset: offset + per_page], total


# Глобальные синглтоны — инициализируются при импорте
def _make_run_store() -> RunStore:
    from api.settings import get_settings
    return RunStore(get_settings().redis_url)


run_store = _make_run_store()

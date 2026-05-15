# api/schemas/runs.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl, field_validator

from api.store import RunRecord, RunStatus


class RunCreate(BaseModel):
    config_name:  str
    save_model:   bool = False
    quick_test:   bool = False
    webhook_url:  Optional[str] = None
    n_synth_rows: Optional[int] = None

    @field_validator("n_synth_rows")
    @classmethod
    def positive_rows(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("n_synth_rows должен быть > 0")
        return v


class RunSummary(BaseModel):
    run_id:       str
    status:       RunStatus
    verdict:      Optional[str]
    dataset_name: str
    config_name:  str
    created_at:   datetime
    started_at:   Optional[datetime]
    finished_at:  Optional[datetime]
    duration_sec: Optional[float]

    @classmethod
    def from_record(cls, r: RunRecord) -> "RunSummary":
        return cls(
            run_id=r.run_id,
            status=r.status,
            verdict=r.verdict,
            dataset_name=r.dataset_name,
            config_name=r.config_name,
            created_at=r.created_at,
            started_at=r.started_at,
            finished_at=r.finished_at,
            duration_sec=r.duration_sec,
        )

    @classmethod
    def from_pg_row(cls, row: Dict[str, Any]) -> "RunSummary":
        """Создаёт RunSummary из строки таблицы processes (PostgreSQL)."""
        raw_status: str = row.get("status") or "ERROR"
        if raw_status.startswith("COMPLETED_"):
            verdict: Optional[str] = raw_status[len("COMPLETED_"):]
            status = RunStatus.completed
        elif raw_status == "RUNNING":
            verdict = None
            status = RunStatus.running
        else:
            verdict = None
            status = RunStatus.failed

        src = row.get("source_data_info") or ""
        dataset_name = Path(src).stem if src else "unknown"

        cfg = row.get("config_rout") or ""
        config_name = Path(cfg).stem if cfg else "unknown"

        def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        started_at = _as_utc(row.get("start_time"))
        finished_at = _as_utc(row.get("end_time"))
        duration_sec: Optional[float] = (
            (finished_at - started_at).total_seconds()
            if started_at and finished_at else None
        )

        return cls(
            run_id=str(row["process_id"]),
            status=status,
            verdict=verdict,
            dataset_name=dataset_name,
            config_name=config_name,
            created_at=started_at or datetime.now(timezone.utc),
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
        )


class RunDetail(RunSummary):
    model_id:        Optional[str]
    synth_rows:      Optional[int]
    config_snapshot: Optional[Dict[str, Any]]
    error_message:   Optional[str]

    @classmethod
    def from_record(cls, r: RunRecord) -> "RunDetail":
        return cls(
            run_id=r.run_id,
            status=r.status,
            verdict=r.verdict,
            dataset_name=r.dataset_name,
            config_name=r.config_name,
            created_at=r.created_at,
            started_at=r.started_at,
            finished_at=r.finished_at,
            duration_sec=r.duration_sec,
            model_id=r.model_id,
            synth_rows=r.synth_rows,
            config_snapshot=r.config_snapshot,
            error_message=r.error_message,
        )


class RunListResponse(BaseModel):
    items: List[RunSummary]
    meta:  Dict[str, Any]

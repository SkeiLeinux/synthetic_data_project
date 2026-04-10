# api/schemas/runs.py

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl, field_validator

from api.store import RunRecord, RunStatus


class RunCreate(BaseModel):
    dataset_name: str
    config_name:  str
    save_model:   bool = False
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

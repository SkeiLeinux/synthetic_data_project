# shared/schemas/synthesis.py
#
# Контракт Synthesis Service: создание джобов, статус, результат.
# Используется: Synthesis Service (возвращает), Gateway (потребляет).

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued  = "queued"
    running = "running"
    done    = "done"
    failed  = "failed"


class SynthesisJobCreate(BaseModel):
    """Тело запроса POST /jobs."""
    split_id: str
    config_name: str
    n_rows: Optional[int] = None    # None = взять из конфига
    save_model: bool = False


class SynthesisJobSummary(BaseModel):
    """
    Ответ GET /jobs/{job_id}.

    synth_path и model_id заполняются после завершения (status=done).
    dp_report содержит DP-конфиг и потраченный epsilon — нужен Evaluation
    и Reporting сервисам.
    """
    job_id: str
    status: JobStatus
    model_id: Optional[str] = None
    synth_path: Optional[str] = None    # путь на Shared Volume
    dp_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

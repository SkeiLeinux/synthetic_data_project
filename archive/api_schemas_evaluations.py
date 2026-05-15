# api/schemas/evaluations.py

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from api.store import EvalStatus


class EvaluationRequest(BaseModel):
    real_dataset_name:  str
    synth_run_id:       str
    quasi_identifiers:  Optional[List[str]] = None
    sensitive_attribute: Optional[str] = None
    target_column:      Optional[str] = None


class EvaluationResult(BaseModel):
    evaluation_id: str
    eval_type:     str
    status:        EvalStatus
    created_at:    datetime
    finished_at:   Optional[datetime]
    report:        Optional[Dict[str, Any]]
    error_message: Optional[str]

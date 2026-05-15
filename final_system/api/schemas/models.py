# api/schemas/models.py

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator


class ModelSummary(BaseModel):
    model_id:        str
    name:            str
    run_id:          Optional[str]
    dataset_name:    str
    created_at:      datetime
    file_size_bytes: int
    epsilon:         Optional[float]
    epochs_completed: Optional[int]
    spent_epsilon:   Optional[float]


class ModelDetail(ModelSummary):
    dp_config:   Optional[Dict[str, Any]]
    dp_spent:    Optional[Dict[str, Any]]
    sample_size: Optional[int]


class SampleRequest(BaseModel):
    n_rows:        int
    output_format: str = "csv"

    @field_validator("n_rows")
    @classmethod
    def validate_rows(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_rows должен быть > 0")
        return v

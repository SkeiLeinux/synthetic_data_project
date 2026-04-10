# api/schemas/configs.py

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ConfigSummary(BaseModel):
    name:         str
    created_at:   datetime
    updated_at:   datetime
    epsilon:      Optional[float]
    epochs:       Optional[int]


class ConfigValidationResult(BaseModel):
    valid:    bool
    errors:   List[Dict[str, str]] = []
    warnings: List[str] = []

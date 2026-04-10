# api/schemas/datasets.py

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DatasetSummary(BaseModel):
    name:            str
    uploaded_at:     datetime
    rows:            Optional[int]
    columns:         Optional[int]
    file_size_bytes: int


class DatasetSchema(BaseModel):
    categorical:  List[str]
    continuous:   List[str]
    ignored:      List[str]
    detected_at:  datetime


class DatasetPreview(BaseModel):
    columns: List[str]
    rows:    List[Dict[str, Any]]
    stats:   Dict[str, Any]

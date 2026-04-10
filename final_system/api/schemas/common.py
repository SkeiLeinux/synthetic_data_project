# api/schemas/common.py

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    pages: int

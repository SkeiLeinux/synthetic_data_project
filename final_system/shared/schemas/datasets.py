# shared/schemas/datasets.py
#
# Контракт Data Service: метаданные датасетов и сплитов.
# Используется: Data Service (возвращает), Gateway (потребляет).

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ColumnSchema(BaseModel):
    """Типизация колонок датасета, определённая при загрузке или сплите."""
    categorical: List[str]
    continuous: List[str]
    target_column: Optional[str] = None


class DatasetMeta(BaseModel):
    """Метаданные загруженного датасета."""
    dataset_id: str
    filename: str
    rows: int
    columns: int
    file_size_bytes: int
    uploaded_at: datetime


class SplitMeta(BaseModel):
    """
    Результат POST /datasets/{id}/split.

    train_path и holdout_path — пути относительно корня Shared Volume (/data).
    Все downstream-сервисы читают файлы по этим путям.
    """
    split_id: str
    dataset_id: str
    train_rows: int
    holdout_rows: int
    train_path: str       # напр. splits/{split_id}/train.csv
    holdout_path: str     # напр. splits/{split_id}/holdout.csv
    categorical_columns: List[str]
    continuous_columns: List[str]
    target_column: Optional[str] = None
    created_at: datetime

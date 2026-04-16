# shared/schemas/datasets.py
#
# Контракт Data Service: метаданные датасетов и сплитов.
# Используется: Data Service (возвращает), Gateway (потребляет).

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


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
    minimization_report: Optional[Dict[str, Any]] = None
    created_at: datetime


class SplitRequest(BaseModel):
    """
    Тело запроса POST /datasets/{id}/split.
    Параметры предобработки и сплита передаются Gateway-ем из конфига запуска.
    """
    holdout_size: float = 0.2
    random_state: int = 42
    sample_size: int = 0                  # 0 = все строки
    na_values: List[str] = ["?"]
    exclude_columns: List[str] = []
    force_categorical: List[str] = []
    force_continuous: List[str] = []
    target_column: Optional[str] = None
    direct_identifiers: List[str] = []
    drop_high_cardinality: bool = False
    cardinality_threshold: float = 0.9

    @field_validator("holdout_size")
    @classmethod
    def valid_holdout(cls, v: float) -> float:
        if not 0.05 <= v <= 0.5:
            raise ValueError("holdout_size должен быть в диапазоне [0.05, 0.5]")
        return v

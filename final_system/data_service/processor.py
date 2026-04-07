# final_system/data_service/processor.py
#
# Preprocessing Service на архитектурной диаграмме.
# Подготовка табличных данных перед передачей в генератор.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ..logger_config import setup_logger
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger_config import setup_logger

logger = setup_logger(__name__)

# Если числовой столбец имеет не более этого числа уникальных значений —
# он считается категориальным (например, education-num имеет 16 значений).
_MAX_UNIQUE_FOR_CATEGORICAL = 15


@dataclass
class DataSchema:
    """
    Схема датасета: разбивка колонок по типам для передачи в генератор.

    Результат работы DataProcessor.detect_column_types().
    Передаётся напрямую в DPCTGANGenerator.fit() вместо хардкода в run_adult.py.

    Пример:
        schema = processor.detect_column_types()
        generator.fit(df, categorical_columns=schema.categorical,
                          continuous_columns=schema.continuous)
    """
    categorical: List[str] = field(default_factory=list)
    continuous: List[str] = field(default_factory=list)
    # Колонки, которые не попали ни в категориальные, ни в непрерывные
    # (например, id-колонки или колонки с типом datetime).
    # Генератор их не видит.
    ignored: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"categorical={len(self.categorical)}, "
            f"continuous={len(self.continuous)}, "
            f"ignored={len(self.ignored)}"
        )


class DataProcessor:
    """
    Предобработка табличных данных перед передачей в генератор.

    Методы:
        preprocess()           — очистка: дубликаты, пропуски
        detect_column_types()  — автоматическая разбивка колонок по типам
        drop_columns()         — удаляет колонки по списку
        get()                  — возвращает текущий датафрейм
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.copy()

    # ─────────────────────────────────────────────
    # Основные методы
    # ─────────────────────────────────────────────

    def preprocess(self) -> pd.DataFrame:
        """
        Базовая очистка:
          - удаление полных дубликатов
          - заполнение пропусков (числовые → median, категориальные → mode)

        Не добавляет и не удаляет колонки — датафрейм остаётся совместимым
        с любым списком categorical_columns / continuous_columns.
        """
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        dropped = before - len(self.df)
        if dropped > 0:
            logger.info(f"[Processor] Удалено дубликатов: {dropped}")

        nulls_filled = 0
        for col in self.df.columns:
            n_null = self.df[col].isnull().sum()
            if n_null == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            nulls_filled += n_null

        if nulls_filled > 0:
            logger.info(f"[Processor] Заполнено пропусков: {nulls_filled}")

        logger.info(f"[Processor] После предобработки: {self.df.shape}")
        return self.df

    def detect_column_types(
        self,
        exclude_columns: Optional[List[str]] = None,
        force_categorical: Optional[List[str]] = None,
        force_continuous: Optional[List[str]] = None,
        max_unique_for_categorical: int = _MAX_UNIQUE_FOR_CATEGORICAL,
    ) -> DataSchema:
        """
        Автоматически определяет типы колонок для передачи в генератор.

        Правила классификации (в порядке приоритета):
          1. Колонки из force_categorical → categorical
          2. Колонки из force_continuous  → continuous
          3. Строковые/object dtype       → categorical
          4. Булевые dtype                → categorical
          5. Числовые с <= max_unique_for_categorical уникальных значений → categorical
          6. Остальные числовые           → continuous
          7. datetime, timedelta и прочее → ignored (не передаются в генератор)

        Параметры:
            exclude_columns          — исключить из схемы полностью (id-поля и т.п.)
            force_categorical        — принудительно считать категориальными
            force_continuous         — принудительно считать непрерывными
            max_unique_for_categorical — порог уникальных значений для числовых колонок

        Возвращает DataSchema с тремя списками: categorical, continuous, ignored.

        Пример:
            schema = processor.detect_column_types(
                exclude_columns=["id"],
                force_categorical=["income"],   # бинарный 0/1 → явно категориальный
            )
        """
        exclude = set(exclude_columns or [])
        forced_cat = set(force_categorical or [])
        forced_cont = set(force_continuous or [])

        categorical, continuous, ignored = [], [], []

        for col in self.df.columns:
            if col in exclude:
                ignored.append(col)
                continue

            if col in forced_cat:
                categorical.append(col)
                continue

            if col in forced_cont:
                continuous.append(col)
                continue

            dtype = self.df[col].dtype

            # Строки и object
            if dtype == object or pd.api.types.is_string_dtype(dtype):
                categorical.append(col)
                continue

            # Булевые
            if pd.api.types.is_bool_dtype(dtype):
                categorical.append(col)
                continue

            # Числовые — смотрим на число уникальных значений
            if pd.api.types.is_numeric_dtype(dtype):
                n_unique = self.df[col].nunique()
                if n_unique <= max_unique_for_categorical:
                    categorical.append(col)
                else:
                    continuous.append(col)
                continue

            # datetime, timedelta и прочее — в ignored
            ignored.append(col)

        schema = DataSchema(
            categorical=categorical,
            continuous=continuous,
            ignored=ignored,
        )
        logger.info(f"[Processor] Схема колонок: {schema.summary()}")
        if schema.ignored:
            logger.info(f"[Processor] Игнорируются: {schema.ignored}")

        return schema

    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        """Удаляет указанные колонки (если они есть). Игнорирует отсутствующие."""
        existing = [c for c in columns if c in self.df.columns]
        if existing:
            self.df.drop(columns=existing, inplace=True)
            logger.info(f"[Processor] Удалены колонки: {existing}")
        return self.df

    def basic_statistics(self) -> pd.DataFrame:
        """Возвращает describe() по всем колонкам."""
        return self.df.describe(include="all")

    def get(self) -> pd.DataFrame:
        """Возвращает текущий датафрейм."""
        return self.df
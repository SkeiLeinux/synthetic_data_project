# synthesizer/base.py
#
# Базовый интерфейс всех генераторов синтетических данных.
# Все генераторы (DPCTGANGenerator, CTGANGenerator, TVAEGenerator,
# CopulaGANGenerator, DPTVAEGenerator) должны реализовывать этот контракт.

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseGenerator(ABC):
    """
    Базовый абстрактный класс для всех синтетических генераторов.

    Минимальный публичный контракт:
        fit()            -- обучение на реальных данных
        sample()         -- генерация синтетических строк
        privacy_report() -- отчёт о параметрах и DP-бюджете (если применимо)
        save() / load()  -- сериализация / десериализация модели

    Метод estimate_max_epochs() имеет дефолтную реализацию (None), т.к.
    он специфичен только для DP-генераторов. Переопределяется в DPCTGANGenerator
    и DPTVAEGenerator.
    """

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        """Обучает генератор на реальных данных."""
        ...

    @abstractmethod
    def sample(self, n_rows: int) -> pd.DataFrame:
        """Генерирует n_rows синтетических строк."""
        ...

    def privacy_report(self) -> Dict[str, Any]:
        """
        Возвращает отчёт о параметрах и расходе DP-бюджета.
        Non-DP генераторы возвращают базовую структуру без DP-полей.
        """
        return {
            "synthesizer": self.__class__.__name__,
            "status": "fitted" if getattr(self, "_is_fitted", False) else "not_fitted",
            "dp_config": None,
            "dp_spent": None,
            "dp_guarantees": None,
        }

    def estimate_max_epochs(
        self,
        data: pd.DataFrame,
        probe_epochs: int = 5,
    ) -> Optional[int]:
        """
        Оценка максимального числа эпох по DP-бюджету через dry run.
        Для non-DP генераторов всегда возвращает None (нет ограничения по бюджету).
        """
        return None

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохраняет обученный генератор на диск."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseGenerator":
        """Загружает генератор из файла."""
        ...

    # ── Вспомогательные методы для подклассов ─────────────────────────────────

    def _pickle_save(self, path: str, payload: Dict[str, Any]) -> None:
        """Сохраняет payload через pickle. Проверяет, что модель обучена."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Нельзя сохранить необученную модель.")
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            raise IOError(f"Ошибка при сохранении модели в {path}: {e}") from e

    @staticmethod
    def _pickle_load(path: str) -> Dict[str, Any]:
        """Загружает payload из pickle-файла."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Ошибка при загрузке модели из {path}: {e}") from e

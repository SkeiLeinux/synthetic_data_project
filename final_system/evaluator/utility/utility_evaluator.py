"""
utility_evaluator.py

Основной класс оценки полезности синтетических данных.
Оркестрирует вызовы statistical.py и ml_efficacy.py,
возвращает единый utility_report для дальнейшей передачи в reporter.

Использование:
    config = UtilityConfig(target_column="income", task_type="classification")
    evaluator = UtilityEvaluator(config)
    report = evaluator.evaluate(real_df, synth_df)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd

from .statistical import compute_correlation_delta, compute_marginal_stats
from .ml_efficacy import MLEfficacyConfig, run_tstr

logger = logging.getLogger(__name__)


@dataclass
class UtilityConfig:
    """
    Конфигурация оценщика полезности.
    Объединяет параметры статистических метрик и ML-оценки.
    """
    # Целевая переменная для TSTR/TRTR
    target_column: str
    task_type: Literal["classification", "regression"] = "classification"

    # Флаги: какие группы метрик считать
    compute_statistical: bool = True
    compute_correlations: bool = True
    compute_ml_efficacy: bool = True

    # Параметры ML-оценки
    n_estimators: int = 100
    max_depth: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42

    # Колонки, которые нужно исключить из признаков (ID, технические поля и т.д.)
    drop_columns: List[str] = field(default_factory=list)


class UtilityEvaluator:
    """
    Оценщик полезности синтетических данных.

    Принимает пару (real_df, synth_df) и возвращает структурированный
    utility_report, совместимый с форматом итогового отчета системы (reporter).
    """

    def __init__(self, config: UtilityConfig) -> None:
        self.config = config

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
    ) -> Dict:
        """
        Запускает все включенные группы метрик и возвращает единый отчет.

        Структура отчета:
        - metadata: размеры датасетов, время оценки
        - statistical: JSD, TVD, stats delta по колонкам
        - correlations: MAE матриц Pearson и Cramér's V
        - ml_efficacy: TRTR, TSTR, Utility Loss
        """
        if not isinstance(real_df, pd.DataFrame) or not isinstance(synth_df, pd.DataFrame):
            raise TypeError("real_df и synth_df должны быть pandas.DataFrame")

        if self.config.target_column not in real_df.columns:
            raise ValueError(
                f"Целевая колонка '{self.config.target_column}' не найдена в real_df"
            )

        logger.info(
            f"[UtilityEvaluator] Старт оценки. "
            f"real={real_df.shape}, synth={synth_df.shape}, "
            f"target='{self.config.target_column}'"
        )

        eval_start = time.monotonic()
        report: Dict = {
            "metadata": {
                "real_rows": len(real_df),
                "synth_rows": len(synth_df),
                "real_columns": len(real_df.columns),
                "target_column": self.config.target_column,
                "task_type": self.config.task_type,
            },
            "statistical": None,
            "correlations": None,
            "ml_efficacy": None,
        }
        # Убираем целевую переменную из статистики
        exclude_from_stats = set(self.config.drop_columns + [self.config.target_column])
        real_features = real_df.drop(columns=[c for c in exclude_from_stats if c in real_df.columns])
        synth_features = synth_df.drop(columns=[c for c in exclude_from_stats if c in synth_df.columns])

        # Статистические метрики по колонкам
        if self.config.compute_statistical:
            logger.info("[UtilityEvaluator] Считаем маргинальные распределения...")
            report["statistical"] = compute_marginal_stats(real_features, synth_features)

        # Сравнение матриц корреляций
        if self.config.compute_correlations:
            logger.info("[UtilityEvaluator] Считаем матрицы корреляций...")
            report["correlations"] = compute_correlation_delta(real_features, synth_features)

        # ML-оценка (TSTR / TRTR)
        if self.config.compute_ml_efficacy:
            logger.info("[UtilityEvaluator] Запускаем TRTR / TSTR...")
            ml_config = MLEfficacyConfig(
                target_column=self.config.target_column,
                task_type=self.config.task_type,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                drop_columns=self.config.drop_columns,
            )
            report["ml_efficacy"] = run_tstr(real_df, synth_df, ml_config)

        report["metadata"]["eval_duration_sec"] = round(time.monotonic() - eval_start, 2)
        logger.info(
            f"[UtilityEvaluator] Готово за {report['metadata']['eval_duration_sec']}с"
        )
        return report


# ─────────────────────────────────────────────
# Тестовый блок
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout  # ← синхронизируем с print()
    )

    print("=== ТЕСТ UTILITY EVALUATOR ===\n")

    # Генерируем тестовые данные, имитирующие реальный и синтетический датасеты.
    # Синтетика намеренно немного зашумлена, чтобы метрики были ненулевыми.
    np.random.seed(42)
    N = 1000

    real_df = pd.DataFrame({
        "age": np.random.randint(18, 70, N),
        "income": np.random.normal(50000, 15000, N).round(0),
        "city": np.random.choice(["Moscow", "SPb", "Kazan", "Novosibirsk"], N),
        "education": np.random.choice(["high", "middle", "low"], N),
        "target": np.random.choice([0, 1], N, p=[0.7, 0.3]),
    })

    # Синтетика: немного смещенные распределения + добавленный шум
    synth_df = pd.DataFrame({
        "age": np.random.randint(20, 75, N),
        "income": np.random.normal(52000, 17000, N).round(0),
        "city": np.random.choice(["Moscow", "SPb", "Kazan", "Novosibirsk"], N, p=[0.5, 0.2, 0.2, 0.1]),
        "education": np.random.choice(["high", "middle", "low"], N, p=[0.5, 0.3, 0.2]),
        "target": np.random.choice([0, 1], N, p=[0.65, 0.35]),
    })

    config = UtilityConfig(
        target_column="target",
        task_type="classification",
        n_estimators=50,   # уменьшаем для быстрого теста
        random_state=42,
    )

    evaluator = UtilityEvaluator(config)
    report = evaluator.evaluate(real_df, synth_df)

    # Вывод отчета
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print("\n=== ТЕСТ ЗАВЕРШЁН ===")

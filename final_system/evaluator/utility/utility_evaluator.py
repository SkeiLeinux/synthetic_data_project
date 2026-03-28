"""
utility_evaluator.py

Основной класс оценки полезности синтетических данных.
Оркестрирует вызовы statistical.py и ml_efficacy.py,
возвращает единый utility_report для дальнейшей передачи в reporter.

Разделение реальных данных на train/test выполняется в run_pipeline()
до обучения генератора. evaluate() принимает готовые части, не делает
split самостоятельно. Это обеспечивает методологическую корректность:
real_test_df — отложенная выборка, которую генератор не видел ни разу.

Использование:
    config = UtilityConfig(target_column="income", task_type="classification")
    evaluator = UtilityEvaluator(config)
    report = evaluator.evaluate(real_train_df, synth_df, real_test_df)
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
    random_state: int = 42

    # Колонки, которые нужно исключить из признаков (ID, технические поля и т.д.)
    drop_columns: List[str] = field(default_factory=list)


class UtilityEvaluator:
    """
    Оценщик полезности синтетических данных.

    Принимает тройку (real_train_df, synth_df, real_test_df) и возвращает
    структурированный utility_report, совместимый с форматом итогового
    отчета системы (reporter).

    Разделение на train/test выполняется снаружи (в run_pipeline),
    чтобы гарантировать, что holdout не виден генератору.
    """

    def __init__(self, config: UtilityConfig) -> None:
        self.config = config

    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
    ) -> Dict:
        """
        Запускает все включенные группы метрик и возвращает единый отчет.

        Параметры:
            real_train_df — часть реальных данных, на которой обучался генератор
            synth_df      — сгенерированные синтетические данные
            real_test_df  — отложенная тестовая выборка (генератор её НЕ видел)

        Статистические метрики (JSD, TVD, корреляции) считаются между
        real_train_df и synth_df — сравниваем синтетику с тем, чему учился генератор.

        ML-метрики (TSTR/TRTR) используют real_test_df как единый holdout для
        честного сравнения качества моделей, обученных на реальных и синтетических данных.

        Структура отчета:
        - metadata:    размеры датасетов, время оценки
        - statistical: JSD, TVD, stats delta по колонкам
        - correlations: MAE матриц Pearson и Cramér's V
        - ml_efficacy:  TRTR, TSTR, Utility Loss
        """
        for name, df in [("real_train_df", real_train_df), ("synth_df", synth_df), ("real_test_df", real_test_df)]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} должен быть pandas.DataFrame")

        if self.config.target_column not in real_train_df.columns:
            raise ValueError(
                f"Целевая колонка '{self.config.target_column}' не найдена в real_train_df"
            )

        logger.info(
            f"[UtilityEvaluator] Старт оценки. "
            f"real_train={real_train_df.shape}, synth={synth_df.shape}, "
            f"real_test={real_test_df.shape}, target='{self.config.target_column}'"
        )

        eval_start = time.monotonic()
        report: Dict = {
            "metadata": {
                "real_train_rows": len(real_train_df),
                "synth_rows": len(synth_df),
                "real_test_rows": len(real_test_df),
                "real_columns": len(real_train_df.columns),
                "target_column": self.config.target_column,
                "task_type": self.config.task_type,
            },
            "statistical": None,
            "correlations": None,
            "ml_efficacy": None,
        }

        # Статистику считаем между train и synth: сравниваем с тем,
        # на чём обучался генератор, а не с holdout.
        exclude_from_stats = set(self.config.drop_columns + [self.config.target_column])
        real_features = real_train_df.drop(
            columns=[c for c in exclude_from_stats if c in real_train_df.columns]
        )
        synth_features = synth_df.drop(
            columns=[c for c in exclude_from_stats if c in synth_df.columns]
        )

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
                random_state=self.config.random_state,
                drop_columns=self.config.drop_columns,
            )
            report["ml_efficacy"] = run_tstr(
                real_train_df=real_train_df,
                synth_df=synth_df,
                real_test_df=real_test_df,
                config=ml_config,
            )

        report["metadata"]["eval_duration_sec"] = round(time.monotonic() - eval_start, 2)
        logger.info(
            f"[UtilityEvaluator] Готово за {report['metadata']['eval_duration_sec']}с"
        )
        return report
"""
privacy_evaluator.py

Основной класс оценки приватности синтетических данных.
Оркестрирует classical.py, distance_metrics.py и attack_simulation.py.

Структура выходного privacy_report:
- dp_guarantees:   формальные гарантии (из отчета генератора)
- empirical_risk:  DCR/NNDR + MIA (эмпирические метрики)
- diagnostic:      k/l/t-анонимность (диагностика структуры таблицы)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .classical import compute_classical_metrics
from .distance_metrics import compute_distance_metrics
from .attack_simulation import run_membership_inference

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """
    Конфигурация оценщика приватности.
    Позволяет включать/отключать группы метрик для гибкости при экспериментах.
    """
    # Колонки для классических метрик k/l/t
    quasi_identifiers: List[str] = field(default_factory=list)
    sensitive_attribute: Optional[str] = None

    # Флаги: какие группы метрик считать
    compute_classical: bool = True
    compute_distance: bool = True
    compute_mia: bool = True

    # Параметры DCR/NNDR и MIA
    distance_sample_size: int = 2000
    mia_sample_size: int = 1000
    mia_n_estimators: int = 100

    # Доля реальных данных для holdout (для DCR и MIA)
    holdout_size: float = 0.2
    random_state: int = 42


class PrivacyEvaluator:
    """
    Оценщик приватности синтетических данных.

    Принимает real_df, synth_df и опционально dp_report от генератора.
    Возвращает полный privacy_report для передачи в reporter.

    Ключевое разграничение в отчете (важно для диплома и ПНСТ):
    - dp_guarantees: "DP сказал, что epsilon=X" (математическая гарантия)
    - empirical_risk: "мы проверили атакой и расстояниями" (практическая проверка)
    - diagnostic: "k/l/t — структурные свойства таблицы" (диагностика)
    """

    def __init__(self, config: PrivacyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        dp_report: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Запускает все включенные группы метрик и возвращает единый отчет.

        dp_report — словарь из DPCTGANGenerator.privacy_report().
        Если передан, формальные DP-гарантии включаются в отчет.
        """
        if not isinstance(real_df, pd.DataFrame) or not isinstance(synth_df, pd.DataFrame):
            raise TypeError("real_df и synth_df должны быть pandas.DataFrame")

        logger.info(
            f"[PrivacyEvaluator] Старт оценки. "
            f"real={real_df.shape}, synth={synth_df.shape}"
        )

        eval_start = time.monotonic()

        # Разбиваем реальные данные на train и holdout.
        # train — то, на чём обучался генератор.
        # holdout — данные, которые генератор НЕ видел (нужны для DCR и MIA).
        real_train, real_holdout = train_test_split(
            real_df,
            test_size=self.config.holdout_size,
            random_state=self.config.random_state,
        )

        report: Dict = {
            "metadata": {
                "real_rows": len(real_df),
                "synth_rows": len(synth_df),
                "real_train_rows": len(real_train),
                "real_holdout_rows": len(real_holdout),
            },
            # Формальные гарантии: передаются напрямую из генератора
            "dp_guarantees": self._extract_dp_guarantees(dp_report),
            "empirical_risk": {},
            "diagnostic": {},
        }

        # Метрики расстояний (DCR, NNDR)
        if self.config.compute_distance:
            logger.info("[PrivacyEvaluator] Считаем DCR и NNDR...")
            report["empirical_risk"]["distance_metrics"] = compute_distance_metrics(
                real_train_df=real_train,
                real_holdout_df=real_holdout,
                synth_df=synth_df,
                sample_size=self.config.distance_sample_size,
            )

        # Membership Inference Attack
        if self.config.compute_mia:
            logger.info("[PrivacyEvaluator] Запускаем MIA...")
            report["empirical_risk"]["membership_inference"] = run_membership_inference(
                real_train_df=real_train,
                real_holdout_df=real_holdout,
                synth_df=synth_df,
                n_estimators=self.config.mia_n_estimators,
                random_state=self.config.random_state,
                sample_size=self.config.mia_sample_size,
            )

        # Классические диагностические метрики (k/l/t)
        if self.config.compute_classical:
            if not self.config.quasi_identifiers or not self.config.sensitive_attribute:
                logger.warning(
                    "[PrivacyEvaluator] quasi_identifiers или sensitive_attribute не заданы. "
                    "Классические метрики пропущены."
                )
            else:
                logger.info("[PrivacyEvaluator] Считаем k/l/t-анонимность...")
                report["diagnostic"]["classical"] = compute_classical_metrics(
                    synth_df=synth_df,
                    real_df=real_df,
                    quasi_identifiers=self.config.quasi_identifiers,
                    sensitive_attribute=self.config.sensitive_attribute,
                )

        report["metadata"]["eval_duration_sec"] = round(time.monotonic() - eval_start, 2)
        logger.info(
            f"[PrivacyEvaluator] Готово за {report['metadata']['eval_duration_sec']}с"
        )
        return report

    @staticmethod
    def _extract_dp_guarantees(dp_report: Optional[Dict]) -> Dict:
        """
        Извлекает ключевые DP-поля из отчета генератора.
        Если dp_report не передан, возвращает пустую структуру.
        """
        if dp_report is None:
            return {"available": False}

        dp_spent = dp_report.get("dp_spent", {})
        dp_config = dp_report.get("dp_config", {})

        return {
            "available": True,
            "is_dp_enabled": dp_config.get("is_dp_enabled"),
            "epsilon_initial": dp_config.get("epsilon_initial"),
            "delta": dp_config.get("delta"),
            "spent_epsilon_final": dp_spent.get("spent_epsilon_final"),
            "epochs_completed": dp_spent.get("epochs_completed"),
            "epochs_requested": dp_spent.get("epochs_requested"),
        }


# ─────────────────────────────────────────────
# Тестовый блок
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    print("=== ТЕСТ PRIVACY EVALUATOR ===\n")

    np.random.seed(42)
    N = 1000

    real_df = pd.DataFrame({
        "age": np.random.randint(18, 70, N),
        "income": np.random.normal(50000, 15000, N).round(0),
        "city": np.random.choice(["Moscow", "SPb", "Kazan", "Novosibirsk"], N),
        "education": np.random.choice(["high", "middle", "low"], N),
        "income_level": np.random.choice(["low", "medium", "high"], N),
    })

    # Синтетика с небольшим шумом
    synth_df = pd.DataFrame({
        "age": np.random.randint(20, 75, N),
        "income": np.random.normal(52000, 17000, N).round(0),
        "city": np.random.choice(["Moscow", "SPb", "Kazan", "Novosibirsk"], N, p=[0.5, 0.2, 0.2, 0.1]),
        "education": np.random.choice(["high", "middle", "low"], N, p=[0.5, 0.3, 0.2]),
        "income_level": np.random.choice(["low", "medium", "high"], N),
    })

    # Имитируем dp_report от генератора
    mock_dp_report = {
        "dp_config": {
            "is_dp_enabled": True,
            "epsilon_initial": 3.0,
            "delta": 4.47e-05,
        },
        "dp_spent": {
            "spent_epsilon_final": 2.48,
            "epochs_requested": 100,
            "epochs_completed": 87,
        },
    }

    config = PrivacyConfig(
        quasi_identifiers=["age", "city", "education"],
        sensitive_attribute="income_level",
        compute_classical=True,
        compute_distance=True,
        compute_mia=True,
        distance_sample_size=500,
        mia_sample_size=300,
    )

    evaluator = PrivacyEvaluator(config)
    report = evaluator.evaluate(real_df, synth_df, dp_report=mock_dp_report)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print("\n=== ТЕСТ ЗАВЕРШЁН ===")

"""
privacy_evaluator.py

Основной класс оценки приватности синтетических данных.
Оркестрирует classical.py, distance_metrics.py и attack_simulation.py.

Структура выходного privacy_report:
- dp_guarantees:   формальные гарантии (из отчета генератора)
- empirical_risk:  DCR/NNDR + MIA (эмпирические метрики)
- diagnostic:      k/l/t-анонимность (диагностика структуры таблицы)

Разделение реальных данных на train/holdout выполняется в run_pipeline()
до обучения генератора. evaluate() принимает готовые части — это обеспечивает
методологическую корректность оценки меморизации:
  - DCR_synth vs DCR_holdout: сравниваем, насколько синтетика «близка» к train,
    используя holdout как контрольную группу.
  - MIA: атакующий классификатор учится различать train-записи (label=1)
    и holdout-записи (label=0) по их расстоянию до синтетики.
    Если генератор не меморизировал данные, расстояния будут неразличимы.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .classical import compute_classical_metrics
from .distance_metrics import compute_distance_metrics
from .attack_simulation import evaluate_membership_inference

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

    random_state: int = 42


class PrivacyEvaluator:
    """
    Оценщик приватности синтетических данных.

    Принимает real_train_df, real_holdout_df, synth_df и опционально
    dp_report от генератора. Возвращает полный privacy_report для reporter.

    Разделение реальных данных на train/holdout выполняется снаружи
    (в run_pipeline), чтобы гарантировать: генератор обучается только
    на train, holdout остаётся «невидимым» эталоном для оценки меморизации.

    Ключевое разграничение в отчете (важно для диплома и ПНСТ):
    - dp_guarantees: "DP сказал, что epsilon=X" (математическая гарантия)
    - empirical_risk: "мы проверили атакой и расстояниями" (практическая проверка)
    - diagnostic: "k/l/t — структурные свойства таблицы" (диагностика)
    """

    def __init__(self, config: PrivacyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_holdout_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        dp_report: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Запускает все включенные группы метрик и возвращает единый отчет.

        Параметры:
            real_train_df   — данные, на которых обучался генератор
            real_holdout_df — данные, которые генератор НЕ видел (отложенная выборка)
            synth_df        — синтетические данные от генератора
            dp_report       — словарь из DPCTGANGenerator.privacy_report();
                              если передан, формальные DP-гарантии включаются в отчет
        """
        for name, df in [
            ("real_train_df", real_train_df),
            ("real_holdout_df", real_holdout_df),
            ("synth_df", synth_df),
        ]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} должен быть pandas.DataFrame")

        logger.info(
            f"[PrivacyEvaluator] Старт оценки. "
            f"real_train={real_train_df.shape}, "
            f"real_holdout={real_holdout_df.shape}, "
            f"synth={synth_df.shape}"
        )

        eval_start = time.monotonic()

        report: Dict = {
            "metadata": {
                "real_train_rows": len(real_train_df),
                "real_holdout_rows": len(real_holdout_df),
                "synth_rows": len(synth_df),
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
                real_train_df=real_train_df,
                real_holdout_df=real_holdout_df,
                synth_df=synth_df,
                sample_size=self.config.distance_sample_size,
            )

        # Proxy Membership Inference Attack (distance-based)
        if self.config.compute_mia:
            logger.info("[PrivacyEvaluator] Запускаем proxy MIA...")
            report["empirical_risk"]["membership_inference"] = evaluate_membership_inference(
                real_train_df=real_train_df,
                real_holdout_df=real_holdout_df,
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
                # Для k/l/t используем полный реальный датасет (train + holdout)
                # как эталон распределения при расчёте t-близости.
                real_full = pd.concat([real_train_df, real_holdout_df], ignore_index=True)
                report["diagnostic"]["classical"] = compute_classical_metrics(
                    synth_df=synth_df,
                    real_df=real_full,
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
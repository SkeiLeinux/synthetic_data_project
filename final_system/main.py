# final_system/main.py

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from synthesizer.dp_ctgan import DPCTGANConfig, DPCTGANGenerator
from evaluator.privacy.privacy_evaluator import PrivacyConfig, PrivacyEvaluator
from evaluator.utility.utility_evaluator import UtilityConfig, UtilityEvaluator
from reporter.reporter import Reporter, VerdictThresholds
from processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    real_df: pd.DataFrame,
    synth_config: DPCTGANConfig,
    privacy_config: PrivacyConfig,
    utility_config: UtilityConfig,
    categorical_columns: List[str],
    continuous_columns: List[str],
    n_synth_rows: int,
    dataset_name: str = "dataset",
    output_dir: str = "reporter/reports",
    thresholds: Optional[VerdictThresholds] = None,
    run_preprocessing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Сквозной пайплайн: предобработка → генерация → оценка → отчёт.

    Параметры:
        run_preprocessing  — если True, запускает DataProcessor.preprocess()
                             перед обучением генератора.
        thresholds         — пороги вердикта; None = дефолтные из VerdictThresholds.

    Возвращает:
        synth_df           — сгенерированный датафрейм
        report             — итоговый отчёт (dict)
    """

    # 1. Предобработка
    if run_preprocessing:
        logger.info("[Pipeline] Предобработка данных...")
        processor = DataProcessor(real_df)
        real_df = processor.preprocess()
        logger.info(f"[Pipeline] После предобработки: {real_df.shape}")

    # 2. Генерация
    logger.info("[Pipeline] Обучение генератора...")
    generator = DPCTGANGenerator(synth_config)
    generator.fit(
        real_df,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
    )
    synth_df = generator.sample(n_synth_rows)
    dp_report = generator.privacy_report()
    logger.info(f"[Pipeline] Сгенерировано {len(synth_df)} строк.")

    # 3. Оценка приватности
    logger.info("[Pipeline] Оценка приватности...")
    privacy_evaluator = PrivacyEvaluator(privacy_config)
    privacy_report = privacy_evaluator.evaluate(real_df, synth_df, dp_report=dp_report)

    # 4. Оценка полезности
    logger.info("[Pipeline] Оценка полезности...")
    utility_evaluator = UtilityEvaluator(utility_config)
    utility_report = utility_evaluator.evaluate(real_df, synth_df)

    # 5. Сборка и сохранение отчёта
    reporter = Reporter(thresholds=thresholds or VerdictThresholds())
    report = reporter.build(
        dp_report=dp_report,
        utility_report=utility_report,
        privacy_report=privacy_report,
        dataset_name=dataset_name,
        generator_type="dpctgan",
    )
    filepath = reporter.save(report, output_dir=output_dir)

    verdict = report["verdict"]
    logger.info(f"[Pipeline] Вердикт: {verdict['overall']}")
    if verdict["issues"]:
        logger.warning(f"[Pipeline] Проблемы: {verdict['issues']}")
    logger.info(f"[Pipeline] Отчёт сохранён: {filepath}")

    return synth_df, report

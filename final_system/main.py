# final_system/main.py

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split

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
    holdout_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Сквозной пайплайн: предобработка → разделение → генерация → оценка → отчёт.

    ВАЖНО — порядок шагов не случаен:
        1. Предобработка (очистка данных)
        2. Разделение на real_train / real_holdout  ← до обучения генератора
        3. Генератор обучается только на real_train
        4. Метрики приватности (DCR, NNDR, MIA) вычисляются с использованием
           real_holdout как контрольной группы — данные, которые генератор не видел
        5. Метрики полезности (TSTR/TRTR) тестируются на real_holdout

    Такой порядок обеспечивает методологическую корректность:
    если генератор обучен на всём датасете, а holdout выделяется позже —
    оценка меморизации некорректна, т.к. генератор «видел» тестовые данные.

    Параметры:
        holdout_size       — доля данных для отложенной выборки (по умолчанию 0.2)
        random_state       — фиксирует разбиение для воспроизводимости
        run_preprocessing  — если True, запускает DataProcessor.preprocess()
                             перед разделением и обучением генератора
        thresholds         — пороги вердикта; None = дефолтные из VerdictThresholds

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

    # 2. Разделение на train и holdout — ДО обучения генератора.
    #    Стратификация по целевому признаку сохраняет баланс классов в обеих частях.
    #    Генератор получит только real_train и никогда не увидит real_holdout.
    target_col = utility_config.target_column
    stratify = real_df[target_col] if target_col in real_df.columns else None

    real_train, real_holdout = train_test_split(
        real_df,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify,
    )
    logger.info(
        f"[Pipeline] Разделение: real_train={len(real_train)}, "
        f"real_holdout={len(real_holdout)} (holdout_size={holdout_size})"
    )

    # 3. Генерация — обучение только на real_train
    logger.info("[Pipeline] Обучение генератора на real_train...")
    generator = DPCTGANGenerator(synth_config)
    generator.fit(
        real_train,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
    )
    synth_df = generator.sample(n_synth_rows)
    dp_report = generator.privacy_report()
    logger.info(f"[Pipeline] Сгенерировано {len(synth_df)} строк.")

    # 4. Оценка приватности
    #    real_holdout передаётся как контрольная группа для DCR и MIA:
    #    проверяем, не «ближе» ли синтетика к обучающим данным, чем независимые данные.
    logger.info("[Pipeline] Оценка приватности...")
    privacy_evaluator = PrivacyEvaluator(privacy_config)
    privacy_report = privacy_evaluator.evaluate(
        real_train_df=real_train,
        real_holdout_df=real_holdout,
        synth_df=synth_df,
        dp_report=dp_report,
    )

    # 5. Оценка полезности
    #    real_holdout — единый тестовый набор для TRTR и TSTR.
    #    Оба эксперимента тестируются на одних и тех же данных → честное сравнение.
    logger.info("[Pipeline] Оценка полезности...")
    utility_evaluator = UtilityEvaluator(utility_config)
    utility_report = utility_evaluator.evaluate(
        real_train_df=real_train,
        synth_df=synth_df,
        real_test_df=real_holdout,
    )

    # 6. Сборка и сохранение отчёта
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
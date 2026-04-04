# final_system/main.py
#
# Orchestrator / Pipeline Service на архитектурной диаграмме.
# Сквозной пайплайн: предобработка → разделение → генерация → оценка → отчёт.

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from data_service.processor import DataProcessor
from synthesizer.dp_ctgan import DPCTGANConfig, DPCTGANGenerator
from evaluator.privacy.privacy_evaluator import PrivacyConfig, PrivacyEvaluator
from evaluator.utility.utility_evaluator import UtilityConfig, UtilityEvaluator
from reporter.reporter import Reporter, VerdictThresholds
from registry.process_registry import ProcessRegistry

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
    source_info: str = "unknown",
    registry: Optional[ProcessRegistry] = None,
    log_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Сквозной пайплайн: предобработка → разделение → генерация → оценка → отчёт.

    ВАЖНО — порядок шагов не случаен:
        1. Регистрация процесса в реестре (если registry передан)
        2. Предобработка данных
        3. Разделение на real_train / real_holdout  <- до обучения генератора
        4. Генератор обучается только на real_train
        5. Метрики приватности (DCR, NNDR, MIA) используют real_holdout
           как контрольную группу — генератор эти данные не видел
        6. Метрики полезности (TSTR/TRTR) тестируются на real_holdout
        7. Сборка отчёта -> вердикт PASS / FAIL / PARTIAL
        8. Сохранение результатов в реестр (если registry передан)

    Параметры:
        source_info   — описание источника данных для записи в реестр
        registry      — экземпляр ProcessRegistry; если None — БД не используется
        log_path      — путь к лог-файлу для сохранения в реестр после завершения

    Возвращает:
        synth_df — сгенерированный датафрейм
        report   — итоговый отчёт (dict) с вердиктом и всеми метриками
    """
    process_id = str(uuid.uuid4())

    # 1. Регистрация процесса
    if registry is not None:
        registry.start_process(
            process_id=process_id,
            source_info=source_info,
        )
        registry.save_source_info(process_id, {
            "num_rows": len(real_df),
            "columns": list(real_df.columns),
            "source": source_info,
        })

    try:
        # 2. Предобработка
        if run_preprocessing:
            logger.info("[Pipeline] Предобработка данных...")
            processor = DataProcessor(real_df)
            real_df = processor.preprocess()
            logger.info(f"[Pipeline] После предобработки: {real_df.shape}")

        # 3. Разделение на train и holdout — ДО обучения генератора.
        #    Стратификация по целевому признаку сохраняет баланс классов.
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

        # 4. Генерация — обучение только на real_train
        logger.info("[Pipeline] Обучение генератора на real_train...")
        all_schema_cols = set(categorical_columns) | set(continuous_columns)
        extra_cols = [c for c in real_train.columns if c not in all_schema_cols]
        if extra_cols:
            logger.info(f"[Pipeline] Колонки вне схемы, удаляются перед генератором: {extra_cols}")
            real_train = real_train.drop(columns=extra_cols)

        generator = DPCTGANGenerator(synth_config)
        generator.fit(
            real_train,
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
        )
        synth_df = generator.sample(n_synth_rows)
        dp_report = generator.privacy_report()
        logger.info(f"[Pipeline] Сгенерировано {len(synth_df)} строк.")

        # Приводим типы синтетики к типам реальных данных
        for col in synth_df.columns:
            if col in real_train.columns:
                try:
                    synth_df[col] = synth_df[col].astype(real_train[col].dtype)
                except (ValueError, TypeError):
                    pass

        if registry is not None:
            registry.save_generator_config(process_id, dp_report.get("dp_config", {}))

        # 5. Оценка приватности
        logger.info("[Pipeline] Оценка приватности...")
        privacy_evaluator = PrivacyEvaluator(privacy_config)
        privacy_report = privacy_evaluator.evaluate(
            real_train_df=real_train,
            real_holdout_df=real_holdout,
            synth_df=synth_df,
            dp_report=dp_report,
        )
        if registry is not None:
            registry.save_privacy_report(process_id, _flatten_for_db(privacy_report))

        # 6. Оценка полезности
        logger.info("[Pipeline] Оценка полезности...")
        utility_evaluator = UtilityEvaluator(utility_config)
        utility_report = utility_evaluator.evaluate(
            real_train_df=real_train,
            synth_df=synth_df,
            real_test_df=real_holdout,
        )
        if registry is not None:
            registry.save_utility_report(process_id, _flatten_for_db(utility_report))

        # 7. Сборка отчёта
        reporter = Reporter(thresholds=thresholds or VerdictThresholds())
        report = reporter.build(
            dp_report=dp_report,
            utility_report=utility_report,
            privacy_report=privacy_report,
            dataset_name=dataset_name,
            generator_type="dpctgan",
            process_id=process_id,
        )
        filepath = reporter.save(report, output_dir=output_dir)

        verdict = report["verdict"]
        logger.info(f"[Pipeline] Вердикт: {verdict['overall']}")
        if verdict["issues"]:
            logger.warning(f"[Pipeline] Проблемы: {verdict['issues']}")
        logger.info(f"[Pipeline] Отчёт сохранён: {filepath}")

        # 8. Финализация в реестре
        if registry is not None:
            registry.finish_process(
                process_id=process_id,
                status=f"COMPLETED_{verdict['overall']}",
                report_location=filepath,
            )
            if log_path:
                registry.save_log_from_file(process_id, log_path)

        return synth_df, report

    except Exception as e:
        logger.exception(f"[Pipeline] Критическая ошибка: {e}")
        if registry is not None:
            registry.finish_process(process_id=process_id, status="ERROR")
            if log_path:
                registry.save_log_from_file(process_id, log_path)
        raise


def _flatten_for_db(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Упрощает вложенный отчёт до плоского словаря для хранения в БД.
    Оставляет только ключевые числовые метрики без вложенных историй и списков.
    """
    result = {}
    try:
        emp = report.get("empirical_risk") or {}
        mia = emp.get("membership_inference") or {}
        dcr = (emp.get("distance_metrics") or {}).get("dcr") or {}
        result["mia_auc"]               = mia.get("attack_auc")
        result["dcr_privacy_preserved"] = dcr.get("privacy_preserved")
        result["dcr_synth_median"]      = (dcr.get("synth_to_real") or {}).get("median")
        result["dcr_holdout_median"]    = (dcr.get("holdout_to_real") or {}).get("median")

        dp = report.get("dp_guarantees") or {}
        result["spent_epsilon"]    = dp.get("spent_epsilon_final")
        result["epochs_completed"] = dp.get("epochs_completed")

        klt = (report.get("diagnostic") or {}).get("classical") or {}
        result["k_anonymity"] = klt.get("k_anonymity")
        result["l_diversity"] = klt.get("l_diversity")
        result["t_closeness"] = klt.get("t_closeness")

        ml = report.get("ml_efficacy") or {}
        result["trtr_f1"]      = (ml.get("trtr") or {}).get("f1_weighted")
        result["tstr_f1"]      = (ml.get("tstr") or {}).get("f1_weighted")
        result["utility_loss"] = (ml.get("utility_loss") or {}).get("value")

        stat = report.get("statistical") or {}
        result["mean_jsd"] = (stat.get("summary") or {}).get("mean_jsd")
        result["mean_tvd"] = (stat.get("summary") or {}).get("mean_tvd")
    except Exception:
        pass
    return result
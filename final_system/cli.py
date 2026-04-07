# final_system/cli.py
#
# Универсальная точка входа для запуска пайплайна.
# Заменяет run_adult.py — все параметры берутся из configs/adult.yaml.
#
# Использование:
#   python cli.py                                  # запуск с configs/adult.yaml
#   python cli.py --config configs/bank.yaml       # другой датасет
#   python cli.py --quick-test                     # быстрый тест (5k строк, 50 эпох)
#   python cli.py --data path/to/data.csv          # переопределить путь к данным
#   python cli.py --check                          # только проверить конфиг и выйти

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config_loader import load_config, apply_quick_test
from data_service.data_io import DataIO
from data_service.processor import DataProcessor
from pipeline import run_pipeline

from registry.process_registry import ProcessRegistry
from logger_config import setup_logger, reconfigure_file_handler
logger = setup_logger(__name__)


# ==============================================================================
# Аргументы командной строки
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сервис генерации конфиденциальных синтетических данных (DP-CTGAN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python cli.py                              # запуск с configs/adult.yaml
  python cli.py --config configs/bank.yaml   # другой датасет
  python cli.py --quick-test                 # быстрый тест (5k строк, 50 эпох)
  python cli.py --data data/custom.csv       # переопределить путь к данным
  python cli.py --check                      # только проверить конфиг
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/adult.yaml",
        metavar="PATH",
        help="Путь к YAML-конфигу (по умолчанию: configs/adult.yaml)",
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        metavar="PATH",
        help="Путь к CSV-файлу с исходными данными (переопределяет config)",
    )
    parser.add_argument(
        "--quick-test", "-q",
        action="store_true",
        help="Режим быстрого теста: 5k строк, 50 эпох, без GPU (~2-3 мин)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Только проверить конфиг и выйти без запуска пайплайна",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Запустить без ProcessRegistry (игнорировать секцию [database])",
    )
    return parser.parse_args()


# ==============================================================================
# Главная функция
# ==============================================================================

def main() -> None:
    args = _parse_args()

    # ── Загрузка и валидация конфига ──────────────────────────────────────────
    logger.info(f"Загрузка конфига: {args.config}")
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    if args.quick_test:
        cfg = apply_quick_test(cfg)
        logger.info("Режим: QUICK TEST (5k строк, 50 эпох, без GPU)")

    if args.data:
        cfg.pipeline.data_path = args.data
        logger.info(f"Путь к данным переопределён: {args.data}")

    if args.check:
        logger.info("Конфиг валиден. Запуск не производился (--check).")
        _print_config_summary(cfg)
        sys.exit(0)

    # ── Перенастройка пути к лог-файлу из конфига ────────────────────────────
    log_path = str(
        Path(os.path.dirname(os.path.abspath(__file__))) / cfg.paths.logs
    )
    reconfigure_file_handler(log_path)

    # ── Загрузка данных ───────────────────────────────────────────────────────
    data_path = Path(cfg.pipeline.data_path)
    if not data_path.exists():
        logger.error(f"Файл данных не найден: {data_path.resolve()}")
        sys.exit(1)

    io = DataIO(app_config=cfg)

    if cfg.pipeline.data_source == "db":
        logger.info("Загрузка данных из БД (data_source)...")
        df = io.load_from_db(cfg.pipeline.db_query)
    else:
        logger.info(f"Загрузка данных из CSV: {data_path}")
        df = io.load_from_csv(str(data_path), na_values=["?"])

    df.dropna(inplace=True)

    # ── Сэмплирование ─────────────────────────────────────────────────────────
    if cfg.pipeline.sample_size > 0:
        df = df.sample(cfg.pipeline.sample_size, random_state=cfg.pipeline.random_state)
        df = df.reset_index(drop=True)
        logger.info(f"Сэмплировано: {len(df)} строк")
    else:
        logger.info(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")

    # ── Определение схемы колонок ─────────────────────────────────────────────
    if cfg.data_schema.is_auto:
        # Автодетекция: DataProcessor определяет типы колонок по dtype
        logger.info("Схема колонок: автодетекция")
        processor = DataProcessor(df)
        schema = processor.detect_column_types(
            exclude_columns=cfg.data_schema.exclude or None,
            force_categorical=[cfg.utility.target_column],
        )
        categorical_cols = schema.categorical
        continuous_cols = schema.continuous
    else:
        # Явная схема из конфига
        logger.info("Схема колонок: из конфига")
        categorical_cols = cfg.data_schema.categorical
        continuous_cols = cfg.data_schema.continuous

    logger.info(f"categorical ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"continuous  ({len(continuous_cols)}): {continuous_cols}")

    # ── ProcessRegistry (опционально) ─────────────────────────────────────────
    registry = None
    if not args.no_db and cfg.database is not None:
        try:
            registry = ProcessRegistry(app_config=cfg)
            if not registry.test_connection():
                logger.warning("БД недоступна, запускаем без реестра процессов")
                registry = None
        except Exception as e:
            logger.warning(f"ProcessRegistry не инициализирован: {e}")
            registry = None

    # ── Вычисляем n_synth_rows ────────────────────────────────────────────────
    # Размер train ≈ len(df) * (1 - holdout_size)
    n_train_approx = int(len(df) * (1 - cfg.pipeline.holdout_size))
    n_synth_rows = cfg.get_n_synth_rows(n_train_approx)
    logger.info(
        f"Разделение: train ~{n_train_approx}, "
        f"holdout ~{len(df) - n_train_approx}, "
        f"n_synth_rows={n_synth_rows}"
    )

    output_dir = str(
        Path(os.path.dirname(os.path.abspath(__file__))) / cfg.paths.output_dir
    )

    # ── Пути вывода ───────────────────────────────────────────────────────────
    synth_out = data_path.parent / f"{data_path.stem}_synth.csv"

    # ── Запуск пайплайна ──────────────────────────────────────────────────────
    try:
        synth_df, report = run_pipeline(
            real_df=df,
            synth_config=cfg.get_generator_config(),
            privacy_config=cfg.get_privacy_config(),
            utility_config=cfg.get_utility_config(),
            categorical_columns=categorical_cols,
            continuous_columns=continuous_cols,
            n_synth_rows=n_synth_rows,
            dataset_name=cfg.pipeline.dataset_name,
            output_dir=output_dir,
            thresholds=cfg.get_thresholds(),
            run_preprocessing=cfg.pipeline.run_preprocessing,
            holdout_size=cfg.pipeline.holdout_size,
            random_state=cfg.pipeline.random_state,
            source_info=str(data_path),
            registry=registry,
            log_path=log_path if registry else None,
            config_path=args.config,
            synth_output_path=str(synth_out),
        )
    finally:
        io.close()
        if registry is not None:
            registry.close()

    # ── Сохранение синтетики ──────────────────────────────────────────────────
    synth_df.to_csv(synth_out, index=False)

    # ── Итоговый вывод ────────────────────────────────────────────────────────
    _print_verdict(report, synth_out, output_dir)


# ==============================================================================
# Вспомогательные функции вывода
# ==============================================================================

def _print_config_summary(cfg) -> None:
    print("\n=== КОНФИГ ===")
    print(f"  dataset:      {cfg.pipeline.dataset_name}")
    print(f"  data_path:    {cfg.pipeline.data_path}")
    print(f"  epsilon:      {cfg.generator.epsilon}")
    print(f"  epochs:       {cfg.generator.epochs}")
    print(f"  target:       {cfg.utility.target_column}")
    schema = cfg.data_schema
    if schema.is_auto:
        print("  schema:       автодетекция")
    else:
        print(f"  categorical:  {len(schema.categorical)} колонок")
        print(f"  continuous:   {len(schema.continuous)} колонок")


def _print_verdict(report: dict, synth_out: Path, output_dir: str) -> None:
    verdict = report["verdict"]

    print("\n" + "=" * 60)
    print(f"  Вердикт: {verdict['overall']}")
    print("=" * 60)

    dp = report.get("generator", {}).get("dp_spent", {})
    print(
        f"  DP:       spent_ε = {dp.get('spent_epsilon_final', 'n/a')}, "
        f"epochs = {dp.get('epochs_completed', 'n/a')}/{dp.get('epochs_requested', 'n/a')}"
    )

    ml = (report.get("utility") or {}).get("ml_efficacy") or {}
    trtr = ml.get("trtr", {})
    tstr = ml.get("tstr", {})
    loss = ml.get("utility_loss", {})
    print(
        f"  Utility:  TRTR F1 = {trtr.get('f1_weighted', 'n/a')}, "
        f"TSTR F1 = {tstr.get('f1_weighted', 'n/a')}, "
        f"loss = {loss.get('value', 'n/a')}"
    )

    emp = (report.get("privacy") or {}).get("empirical_risk") or {}
    mia = emp.get("membership_inference") or {}
    dcr = (emp.get("distance_metrics") or {}).get("dcr") or {}
    print(
        f"  Privacy:  MIA AUC = {mia.get('attack_auc', 'n/a')}, "
        f"DCR ok = {dcr.get('privacy_preserved', 'n/a')}"
    )

    klt = ((report.get("privacy") or {}).get("diagnostic") or {}).get("classical") or {}
    print(
        f"  k/l/t:    k={klt.get('k_anonymity', 'n/a')}, "
        f"l={klt.get('l_diversity', 'n/a')}, "
        f"t={klt.get('t_closeness', 'n/a')}"
    )

    if verdict["issues"]:
        print(f"\n  ⚠ Проблемы:")
        for issue in verdict["issues"]:
            print(f"    - {issue}")

    print(f"\n  Синтетика: {synth_out}")
    print(f"  Отчёт:     {output_dir}/")
    print("=" * 60)


# ==============================================================================

if __name__ == "__main__":
    main()
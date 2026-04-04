# final_system/run_adult.py
#
# Точка входа для тестового запуска пайплайна на датасете Adult Census.
# Запускать из корня репозитория:
#   python final_system/run_adult.py
#
# QUICK_TEST = True  -> 5k строк, 50 эпох, ~2-3 минуты
# QUICK_TEST = False -> полный датасет, 300 эпох

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from main import run_pipeline
from data_service.processor import DataProcessor
from synthesizer.dp_ctgan import DPCTGANConfig
from evaluator.privacy.privacy_evaluator import PrivacyConfig
from evaluator.utility.utility_evaluator import UtilityConfig
from reporter.reporter import VerdictThresholds

# Раскомментировать когда БД настроена:
# from registry.process_registry import ProcessRegistry

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Режим запуска
# ──────────────────────────────────────────────────────────────────────────────
QUICK_TEST = True


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка и подготовка данных
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult.csv")

df = pd.read_csv(DATA_PATH, skipinitialspace=True, na_values=["?"])
df.dropna(inplace=True)

# income -> бинарный числовой признак
df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})

if QUICK_TEST:
    df = df.sample(5000, random_state=42).reset_index(drop=True)

logger.info(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")


# ──────────────────────────────────────────────────────────────────────────────
# Автодетекция колонок через DataProcessor
# income принудительно в categorical: бинарный 0/1 -> категориальный тип
# fnlwgt в ignored: технический вес, не нужен генератору
# ──────────────────────────────────────────────────────────────────────────────
processor = DataProcessor(df)
schema = processor.detect_column_types(
    force_categorical=["income"],
    exclude_columns=["fnlwgt"],
)
logger.info(f"categorical: {schema.categorical}")
logger.info(f"continuous:  {schema.continuous}")


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурации
# ──────────────────────────────────────────────────────────────────────────────
if QUICK_TEST:
    synth_config = DPCTGANConfig(
        epsilon=5.0,
        preprocessor_eps=0.5,
        epochs=50,
        batch_size=500,
        cuda=False,
        verbose=True,
        disabled_dp=False,
        random_seed=42,
    )
    n_synth_rows = 4000
else:
    synth_config = DPCTGANConfig(
        epsilon=3.0,
        preprocessor_eps=0.5,
        epochs=300,
        batch_size=500,
        sigma=5.0,
        cuda=True,
        verbose=True,
        disabled_dp=False,
        random_seed=42,
    )
    n_synth_rows = 26000

privacy_config = PrivacyConfig(
    quasi_identifiers=["age", "education", "occupation", "sex", "race"],
    sensitive_attribute="income",
    compute_classical=True,
    compute_distance=True,
    compute_mia=True,
    distance_sample_size=1000 if QUICK_TEST else 2000,
    mia_sample_size=500 if QUICK_TEST else 1000,
    random_state=42,
)

utility_config = UtilityConfig(
    target_column="income",
    task_type="classification",
    drop_columns=["fnlwgt"],
    random_state=42,
)

thresholds = VerdictThresholds(
    max_utility_loss=0.25,
    max_mean_jsd=0.40,
    max_mia_auc=0.60,
    require_dp_enabled=True,
    max_spent_epsilon=None,
)


# ──────────────────────────────────────────────────────────────────────────────
# ProcessRegistry (опционально — раскомментировать когда БД готова)
# ──────────────────────────────────────────────────────────────────────────────
registry = None
# registry = ProcessRegistry()
# if not registry.test_connection():
#     logger.warning("БД недоступна, запускаем без реестра процессов")
#     registry = None

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "app.log")


# ──────────────────────────────────────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────────────────────────────────────
logger.info(f"Режим: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
logger.info(f"Датасет: {len(df)} строк -> train ~{int(len(df)*0.8)}, holdout ~{int(len(df)*0.2)}")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporter", "reports")

synth_df, report = run_pipeline(
    real_df=df,
    synth_config=synth_config,
    privacy_config=privacy_config,
    utility_config=utility_config,
    categorical_columns=schema.categorical,
    continuous_columns=schema.continuous,
    n_synth_rows=n_synth_rows,
    dataset_name="adult_census",
    output_dir=output_dir,
    thresholds=thresholds,
    run_preprocessing=True,
    holdout_size=0.2,
    random_state=42,
    source_info=DATA_PATH,
    registry=registry,
    log_path=LOG_PATH if registry else None,
)

# Сохранение синтетики
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult_synth.csv")
synth_df.to_csv(output_csv, index=False)

# Итоговый вывод
verdict = report["verdict"]
print("\n" + "=" * 60)
print(f"  Вердикт: {verdict['overall']}")
print("=" * 60)

dp = report.get("generator", {}).get("dp_spent", {})
print(f"  DP:       spent_e = {dp.get('spent_epsilon_final', 'n/a')}, "
      f"epochs = {dp.get('epochs_completed', 'n/a')}/{dp.get('epochs_requested', 'n/a')}")

ml   = (report.get("utility") or {}).get("ml_efficacy") or {}
trtr = ml.get("trtr", {})
tstr = ml.get("tstr", {})
loss = ml.get("utility_loss", {})
print(f"  Utility:  TRTR F1 = {trtr.get('f1_weighted', 'n/a')}, "
      f"TSTR F1 = {tstr.get('f1_weighted', 'n/a')}, "
      f"loss = {loss.get('value', 'n/a')}")

emp = (report.get("privacy") or {}).get("empirical_risk") or {}
mia = emp.get("membership_inference") or {}
dcr = (emp.get("distance_metrics") or {}).get("dcr") or {}
print(f"  Privacy:  MIA AUC = {mia.get('attack_auc', 'n/a')}, "
      f"DCR ok = {dcr.get('privacy_preserved', 'n/a')}")

klt = ((report.get("privacy") or {}).get("diagnostic") or {}).get("classical") or {}
print(f"  k/l/t:    k={klt.get('k_anonymity', 'n/a')}, "
      f"l={klt.get('l_diversity', 'n/a')}, "
      f"t={klt.get('t_closeness', 'n/a')}")

if verdict["issues"]:
    print(f"  Проблемы: {verdict['issues']}")

print(f"\n  Синтетика: {output_csv} ({len(synth_df)} строк)")
print(f"  Отчёт:    {output_dir}/")
print("=" * 60)
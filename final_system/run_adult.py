# final_system/run_adult.py
#
# Тестовый запуск полного пайплайна на датасете Adult Census.
# Запускать из корня репозитория:
#   python final_system/run_adult.py
#
# QUICK_TEST = True  → 5k строк, 50 эпох, ~2-3 минуты — проверить что всё работает
# QUICK_TEST = False → полный датасет, 300 эпох — финальный запуск для диплома

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from main import run_pipeline
from synthesizer.dp_ctgan import DPCTGANConfig
from evaluator.privacy.privacy_evaluator import PrivacyConfig
from evaluator.utility.utility_evaluator import UtilityConfig
from reporter.reporter import VerdictThresholds

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Режим запуска
# ──────────────────────────────────────────────────────────────────────────────
QUICK_TEST = False


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка и подготовка данных
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult.csv")

df = pd.read_csv(DATA_PATH, skipinitialspace=True, na_values=["?"])
df.dropna(inplace=True)

# income → бинарный числовой признак (нужно до передачи в пайплайн,
# иначе смешанный тип строка/число сломает генератор)
df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})

if QUICK_TEST:
    df = df.sample(5000, random_state=42).reset_index(drop=True)

logger.info(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")


# ──────────────────────────────────────────────────────────────────────────────
# Колонки по типам
# income не входит в CAT/CONT явно — SmartNoise обрабатывает его отдельно
# ──────────────────────────────────────────────────────────────────────────────
CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country", "income",
]
CONT_COLS = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурации
# ──────────────────────────────────────────────────────────────────────────────
if QUICK_TEST:
    synth_config = DPCTGANConfig(
        epsilon=5.0,           # высокий ε — быстрее сходится, меньше шума
        preprocessor_eps=0.5,
        epochs=50,
        batch_size=500,
        cuda=True,
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
        batch_size=500,        # меньше батч → точнее privacy accountant
        # sigma=2.0,
        cuda=True,
        verbose=True,
        disabled_dp=False,
        random_seed=42,
    )
    n_synth_rows = 26000       # ~= размеру train-части

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
    drop_columns=["fnlwgt"],   # технический вес, не несёт смысла для ML
    random_state=42,
)

# Пороги адаптированы под Adult + DP:
# DP неизбежно снижает качество, поэтому пороги мягче "идеальных"
thresholds = VerdictThresholds(
    max_utility_loss=0.25,     # до 25% потери F1 — приемлемо при DP
    max_mean_jsd=0.15,         # Adult с DP даёт JSD ~0.05-0.12 по числовым
    max_mia_auc=0.60,          # стандартная граница "атака работает"
    require_dp_enabled=True,
    max_spent_epsilon=None,
)


# ──────────────────────────────────────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────────────────────────────────────
logger.info(f"Режим: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
logger.info(f"Датасет: {len(df)} строк → train ~{int(len(df)*0.8)}, holdout ~{int(len(df)*0.2)}")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporter", "reports")

synth_df, report = run_pipeline(
    real_df=df,
    synth_config=synth_config,
    privacy_config=privacy_config,
    utility_config=utility_config,
    categorical_columns=CAT_COLS,
    continuous_columns=CONT_COLS,
    n_synth_rows=n_synth_rows,
    dataset_name="adult_census",
    output_dir=output_dir,
    thresholds=thresholds,
    run_preprocessing=True,
    holdout_size=0.2,
    random_state=42,
)

print(synth_df["income"].dtype, synth_df["income"].unique())
print(df["income"].dtype, df["income"].unique())

# Сохранение синтетики
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult_synth.csv")
synth_df.to_csv(output_csv, index=False)

# Итоговый вывод
verdict = report["verdict"]
print("\n" + "="*60)
print(f"  Вердикт: {verdict['overall']}")
print("="*60)

dp = report.get("generator", {}).get("dp_spent", {})
print(f"  DP:       spent_ε = {dp.get('spent_epsilon_final', 'n/a')}, "
      f"epochs = {dp.get('epochs_completed', 'n/a')}/{dp.get('epochs_requested', 'n/a')}")

ml  = (report.get("utility") or {}).get("ml_efficacy") or {}
trtr = ml.get("trtr", {})
tstr = ml.get("tstr", {})
loss = ml.get("utility_loss", {})
print(f"  Utility:  TRTR F1 = {trtr.get('f1_weighted', 'n/a')}, "
      f"TSTR F1 = {tstr.get('f1_weighted', 'n/a')}, "
      f"loss = {loss.get('value', 'n/a')}")

emp  = (report.get("privacy") or {}).get("empirical_risk") or {}
mia  = emp.get("membership_inference") or {}
dcr  = (emp.get("distance_metrics") or {}).get("dcr") or {}
print(f"  Privacy:  MIA AUC = {mia.get('attack_auc', 'n/a')}, "
      f"DCR ok = {dcr.get('privacy_preserved', 'n/a')}")

klt  = ((report.get("privacy") or {}).get("diagnostic") or {}).get("classical") or {}
print(f"  k/l/t:    k={klt.get('k_anonymity','n/a')}, "
      f"l={klt.get('l_diversity','n/a')}, "
      f"t={klt.get('t_closeness','n/a')}")

if verdict["issues"]:
    print(f"  Проблемы: {verdict['issues']}")

print(f"\n  Синтетика: {output_csv} ({len(synth_df)} строк)")
print(f"  Отчёт:    {output_dir}/")
print("="*60)
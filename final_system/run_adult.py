# final_system/run_adult.py

import os
import sys
import configparser, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from main import run_pipeline
from synthesizer.dp_ctgan import DPCTGANConfig
from evaluator.privacy.privacy_evaluator import PrivacyConfig
from evaluator.utility.utility_evaluator import UtilityConfig


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read(Path(__file__).parent / "config.ini")

gen_cfg = json.loads(cfg["GENERATOR"]["config_json"])
utility_cfg = json.loads(cfg["UTILITY"]["config_json"])
privacy_cfg = json.loads(cfg["PRIVACY"]["config_json"])
schema_cfg = cfg["DATA_SCHEMA"]



# ──────────────────────────────────────────────────────────────────────────────
# Загрузка
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "final_system/data/adult.csv",
    skipinitialspace=True,   # убирает пробелы перед значениями
    na_values=["?"],
)
df.dropna(inplace=True)
df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})
# df = df.sample(3000, random_state=42).reset_index(drop=True)

print(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")
print(df.dtypes)

# ──────────────────────────────────────────────────────────────────────────────
# Колонки
# ──────────────────────────────────────────────────────────────────────────────
CAT_COLS = json.loads(schema_cfg["columns_categorical"])
CONT_COLS = json.loads(schema_cfg["columns_continuous"])
# CAT_COLS = [
#     "workclass", "education", "marital-status", "occupation",
#     "relationship", "race", "sex", "native-country", "income",
# ]
# CONT_COLS = [
#     "age", "fnlwgt", "education-num",
#     "capital-gain", "capital-loss", "hours-per-week",
# ]
# income — target, не передаётся в CAT/CONT явно,
# SmartNoise обработает его как отдельную колонку автоматически

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурации
# ──────────────────────────────────────────────────────────────────────────────
synth_config = DPCTGANConfig(
    epsilon=3.0,
    preprocessor_eps=0.5,
    epochs=300,
    batch_size=100,
    cuda=True,
    verbose=True,
    random_seed=42,
)

privacy_config = PrivacyConfig(
    quasi_identifiers=["age", "education", "occupation", "sex", "race"],
    sensitive_attribute="income",
    compute_classical=True,
    compute_distance=True,
    compute_mia=True,
    distance_sample_size=1000,
    mia_sample_size=500,
)

utility_config = UtilityConfig(
    target_column="income",
    task_type="classification",
    drop_columns=["fnlwgt"],
    random_state=42,
)

# ──────────────────────────────────────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────────────────────────────────────

synth_df, report = run_pipeline(
    real_df=df,
    synth_config=synth_config,
    privacy_config=privacy_config,
    utility_config=utility_config,
    categorical_columns=CAT_COLS,
    continuous_columns=CONT_COLS,
    n_synth_rows=30000,
    dataset_name="adult_census",
    output_dir="final_system/reporter/reports",
)

synth_df.to_csv("final_system/data/adult_synth.csv", index=False)
print(f"\nСинтетика сохранена: final_system/data/adult_synth.csv ({len(synth_df)} строк)")

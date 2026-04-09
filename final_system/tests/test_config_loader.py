# final_system/tests/test_config_loader.py
#
# Unit-тесты для валидации конфигурации (config_loader.py)
# Запуск: python -m pytest final_system/tests/test_config_loader.py -v

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError
from config_loader import (
    AppConfig, PipelineConfig, GeneratorYamlConfig,
    DataSchemaYamlConfig, load_config, apply_quick_test,
)


# ─────────────────────────────────────────────
# Минимальный валидный raw-конфиг для тестов
# ─────────────────────────────────────────────

def _base_raw():
    return {
        "pipeline": {
            "dataset_name": "test",
            "data_source": "csv",
            "data_path": "data/test.csv",
        },
        "utility": {
            "target_column": "label",
        },
    }


# ─────────────────────────────────────────────
# Успешная загрузка
# ─────────────────────────────────────────────

def test_valid_minimal_config_loads():
    cfg = AppConfig.model_validate(_base_raw())
    assert cfg.pipeline.dataset_name == "test"
    assert cfg.pipeline.data_source == "csv"
    assert cfg.utility.target_column == "label"


def test_defaults_applied():
    cfg = AppConfig.model_validate(_base_raw())
    assert cfg.generator.epsilon == 3.0
    assert cfg.pipeline.holdout_size == 0.2
    assert cfg.pipeline.random_state == 42
    assert cfg.data_schema.drop_high_cardinality is False
    assert cfg.data_schema.cardinality_threshold == 0.9
    assert cfg.data_schema.direct_identifiers == []


def test_load_real_adult_config():
    """Проверяем, что реальный конфиг adult.yaml загружается без ошибок."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "adult.yaml"
    )
    cfg = load_config(config_path)
    assert cfg.pipeline.dataset_name == "adult_census"
    assert cfg.generator.epsilon > 0
    assert cfg.utility.target_column == "income"
    assert cfg.data_schema.direct_identifiers == []


# ─────────────────────────────────────────────
# Валидация epsilon
# ─────────────────────────────────────────────

def test_epsilon_zero_raises():
    raw = _base_raw()
    raw["generator"] = {"epsilon": 0.0}
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_epsilon_negative_raises():
    raw = _base_raw()
    raw["generator"] = {"epsilon": -1.0}
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_preprocessor_eps_exceeds_epsilon_raises():
    """preprocessor_eps должен быть < epsilon."""
    raw = _base_raw()
    raw["generator"] = {"epsilon": 1.0, "preprocessor_eps": 1.0}
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_preprocessor_eps_valid():
    raw = _base_raw()
    raw["generator"] = {"epsilon": 3.0, "preprocessor_eps": 0.5}
    cfg = AppConfig.model_validate(raw)
    assert cfg.generator.preprocessor_eps == 0.5


# ─────────────────────────────────────────────
# Валидация holdout_size
# ─────────────────────────────────────────────

def test_holdout_size_zero_raises():
    raw = _base_raw()
    raw["pipeline"]["holdout_size"] = 0.0
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_holdout_size_one_raises():
    raw = _base_raw()
    raw["pipeline"]["holdout_size"] = 1.0
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_holdout_size_valid():
    raw = _base_raw()
    raw["pipeline"]["holdout_size"] = 0.3
    cfg = AppConfig.model_validate(raw)
    assert cfg.pipeline.holdout_size == 0.3


# ─────────────────────────────────────────────
# Валидация data_source
# ─────────────────────────────────────────────

def test_unknown_data_source_raises():
    raw = _base_raw()
    raw["pipeline"]["data_source"] = "ftp"
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_csv_without_data_path_raises():
    raw = _base_raw()
    raw["pipeline"]["data_path"] = ""
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


def test_db_source_without_query_raises():
    raw = _base_raw()
    raw["pipeline"]["data_source"] = "db"
    raw["pipeline"]["data_path"] = ""
    # db_query не задан
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate(raw)


# ─────────────────────────────────────────────
# DataSchemaYamlConfig
# ─────────────────────────────────────────────

def test_schema_is_auto_when_empty():
    schema = DataSchemaYamlConfig()
    assert schema.is_auto is True


def test_schema_is_not_auto_when_set():
    schema = DataSchemaYamlConfig(categorical=["col1"], continuous=["col2"])
    assert schema.is_auto is False


def test_schema_minimization_defaults():
    schema = DataSchemaYamlConfig()
    assert schema.direct_identifiers == []
    assert schema.drop_high_cardinality is False
    assert schema.cardinality_threshold == 0.9


# ─────────────────────────────────────────────
# apply_quick_test
# ─────────────────────────────────────────────

def test_apply_quick_test_overrides_params():
    cfg = AppConfig.model_validate(_base_raw())
    quick = apply_quick_test(cfg)
    assert quick.pipeline.sample_size == 5000
    assert quick.generator.epochs == 50
    assert quick.generator.cuda is False
    assert quick.privacy.distance_sample_size == 500
    assert quick.privacy.mia_sample_size == 250
    # Пороги ослаблены для smoke-test режима
    assert quick.thresholds.max_utility_loss == pytest.approx(0.40)
    assert quick.thresholds.max_mean_jsd == pytest.approx(0.40)
    assert quick.thresholds.max_mia_auc == pytest.approx(0.65)
    assert quick.thresholds.max_spent_epsilon is None


def test_apply_quick_test_does_not_mutate_original():
    cfg = AppConfig.model_validate(_base_raw())
    original_epochs = cfg.generator.epochs
    _ = apply_quick_test(cfg)
    assert cfg.generator.epochs == original_epochs


# ─────────────────────────────────────────────
# load_config — файловые ошибки
# ─────────────────────────────────────────────

def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/path/config.yaml")

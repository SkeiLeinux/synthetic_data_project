# final_system/tests/test_processor_core.py
#
# Unit-тесты для DataProcessor.preprocess() и detect_column_types()
# Запуск: python -m pytest final_system/tests/test_processor_core.py -v

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from data_processor.processor import DataProcessor, DataSchema


# ─────────────────────────────────────────────
# Фикстуры
# ─────────────────────────────────────────────

@pytest.fixture
def mixed_df():
    """DataFrame со смешанными типами, дубликатами и пропусками."""
    return pd.DataFrame({
        "age":        [25, 30, 30, None, 45],      # числовой, один пропуск, один дубликат
        "income":     ["low", "high", "high", "low", "high"],  # object (дубликат в строке 2)
        "score":      [0.1, 0.5, 0.5, 0.3, None],  # float, пропуск
        "flag":       [True, False, False, True, True],  # bool
    })


# ─────────────────────────────────────────────
# preprocess()
# ─────────────────────────────────────────────

def test_preprocess_removes_duplicate_rows(mixed_df):
    processor = DataProcessor(mixed_df)
    result = processor.preprocess()
    # Строки 1 и 2 идентичны (30, high, 0.5, False) → одна удаляется
    assert len(result) == 4


def test_preprocess_fills_numeric_nulls_with_median(mixed_df):
    processor = DataProcessor(mixed_df)
    result = processor.preprocess()
    assert result["age"].isnull().sum() == 0


def test_preprocess_fills_categorical_nulls_with_mode():
    # Добавляем второй столбец, чтобы строки не были дубликатами после заполнения
    df = pd.DataFrame({
        "city":  ["Moscow", "Kazan", None, "SPb", "Moscow"],
        "index": [1,        2,       3,    4,      5],        # уникальный, не даёт дублей строк
    })
    processor = DataProcessor(df)
    result = processor.preprocess()
    # Пропуск должен исчезнуть — неважно каким значением (mode = "Moscow")
    assert result["city"].isnull().sum() == 0
    assert len(result) == len(df)  # строки не удаляются (нет полных дубликатов)


def test_preprocess_does_not_change_columns(mixed_df):
    processor = DataProcessor(mixed_df)
    cols_before = list(mixed_df.columns)
    result = processor.preprocess()
    assert list(result.columns) == cols_before


def test_preprocess_no_duplicates_no_nulls_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    processor = DataProcessor(df)
    result = processor.preprocess()
    assert len(result) == 3
    assert result.isnull().sum().sum() == 0


# ─────────────────────────────────────────────
# detect_column_types()
# ─────────────────────────────────────────────

@pytest.fixture
def schema_df():
    n = 20  # > _MAX_UNIQUE_FOR_CATEGORICAL=15, чтобы числовые столбцы попали в continuous
    return pd.DataFrame({
        "name":        [f"Person_{i}" for i in range(n)],          # object → categorical
        "age":         list(range(n)),                              # int, n уникальных > 15 → continuous
        "education":   [i % 5 for i in range(n)],                  # int, 5 уникальных < 15 → categorical
        "salary":      [float(i) * 1000.5 for i in range(n)],      # float, n уникальных > 15 → continuous
        "is_adult":    [i % 2 == 0 for i in range(n)],             # bool → categorical
        "created_at":  pd.to_datetime([f"202{i % 5}-01-01" for i in range(n)]),  # datetime → ignored
        "id":          list(range(101, 101 + n)),                   # будет в exclude
    })


def test_detect_object_columns_are_categorical(schema_df):
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(exclude_columns=["id"])
    assert "name" in schema.categorical


def test_detect_bool_columns_are_categorical(schema_df):
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(exclude_columns=["id"])
    assert "is_adult" in schema.categorical


def test_detect_high_unique_numeric_is_continuous(schema_df):
    processor = DataProcessor(schema_df)
    # salary: 20 уникальных float-значений > _MAX_UNIQUE_FOR_CATEGORICAL=15 → continuous
    schema = processor.detect_column_types(exclude_columns=["id"])
    assert "salary" in schema.continuous


def test_detect_datetime_is_ignored(schema_df):
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(exclude_columns=["id"])
    assert "created_at" in schema.ignored


def test_detect_exclude_columns_go_to_ignored(schema_df):
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(exclude_columns=["id"])
    assert "id" in schema.ignored
    assert "id" not in schema.categorical
    assert "id" not in schema.continuous


def test_detect_force_categorical_overrides(schema_df):
    """Числовой столбец можно принудительно сделать категориальным."""
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(
        exclude_columns=["id"],
        force_categorical=["salary"],
    )
    assert "salary" in schema.categorical
    assert "salary" not in schema.continuous


def test_detect_force_continuous_overrides(schema_df):
    """Object-столбец можно принудительно сделать continuous."""
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(
        exclude_columns=["id"],
        force_continuous=["name"],
    )
    assert "name" in schema.continuous
    assert "name" not in schema.categorical


def test_detect_all_columns_classified(schema_df):
    """Каждая колонка должна оказаться ровно в одном списке."""
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types(exclude_columns=["id"])
    all_cols = set(schema.categorical) | set(schema.continuous) | set(schema.ignored)
    expected = set(schema_df.columns)
    assert all_cols == expected


def test_detect_no_column_in_two_lists(schema_df):
    """Колонка не может быть одновременно в двух списках."""
    processor = DataProcessor(schema_df)
    schema = processor.detect_column_types()
    cat = set(schema.categorical)
    cont = set(schema.continuous)
    ign = set(schema.ignored)
    assert cat & cont == set()
    assert cat & ign == set()
    assert cont & ign == set()


# ─────────────────────────────────────────────
# get() и drop_columns()
# ─────────────────────────────────────────────

def test_get_returns_current_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    processor = DataProcessor(df)
    assert processor.get().equals(processor.df)


def test_drop_columns_removes_existing():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    processor = DataProcessor(df)
    result = processor.drop_columns(["a", "b"])
    assert "a" not in result.columns
    assert "b" not in result.columns
    assert "c" in result.columns


def test_drop_columns_ignores_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    processor = DataProcessor(df)
    # Не должен падать на несуществующей колонке
    result = processor.drop_columns(["a", "nonexistent"])
    assert "a" not in result.columns
    assert "b" in result.columns

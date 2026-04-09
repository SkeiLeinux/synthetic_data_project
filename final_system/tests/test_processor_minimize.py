# final_system/tests/test_processor_minimize.py
#
# Unit-тесты для DataProcessor.minimize()
# Запуск: python -m pytest final_system/tests/test_processor_minimize.py -v

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from data_service.processor import DataProcessor


# ─────────────────────────────────────────────
# Фикстуры
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":         [1, 2, 3, 4, 5],
        "email":      ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        "name":       ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age":        [25, 30, 35, 40, 45],
        "income":     ["low", "high", "low", "high", "low"],
        "city":       ["Moscow", "Moscow", "SPb", "SPb", "Moscow"],
    })


# ─────────────────────────────────────────────
# Тесты
# ─────────────────────────────────────────────

def test_minimize_removes_direct_identifiers(sample_df):
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(direct_identifiers=["id", "email"])

    assert "id" not in df_result.columns
    assert "email" not in df_result.columns
    assert "age" in df_result.columns
    assert "income" in df_result.columns

    assert report["removed_direct_identifiers"] == ["id", "email"]
    assert report["removed_high_cardinality"] == []
    assert report["columns_before"] == 6
    assert report["columns_after"] == 4
    assert report["rows_unchanged"] is True


def test_minimize_rows_never_change(sample_df):
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(
        direct_identifiers=["id", "email", "name"],
        drop_high_cardinality=True,
        cardinality_threshold=0.5,
    )
    assert len(df_result) == len(sample_df)
    assert report["rows_unchanged"] is True


def test_minimize_missing_columns_does_not_raise(sample_df):
    """Колонки, которых нет в DataFrame, должны молча пропускаться."""
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(
        direct_identifiers=["id", "nonexistent_column", "another_missing"]
    )
    assert "id" not in df_result.columns
    # Несуществующие — не попали в removed
    assert "nonexistent_column" not in report["removed_direct_identifiers"]
    assert "another_missing" not in report["removed_direct_identifiers"]
    assert report["removed_direct_identifiers"] == ["id"]


def test_minimize_high_cardinality_removes_object_columns(sample_df):
    """
    email и name — каждый имеет 5 уникальных значений из 5 строк (100%),
    что превышает порог 0.9. Должны быть удалены.
    city — 2 уникальных из 5 (40%), не должна удаляться.
    """
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(
        direct_identifiers=[],
        drop_high_cardinality=True,
        cardinality_threshold=0.9,
    )
    assert "email" not in df_result.columns
    assert "name" not in df_result.columns
    assert "city" in df_result.columns
    assert set(report["removed_high_cardinality"]) == {"email", "name"}


def test_minimize_high_cardinality_skips_numeric(sample_df):
    """Числовые колонки не трогаются даже при высокой кардинальности."""
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(
        direct_identifiers=[],
        drop_high_cardinality=True,
        cardinality_threshold=0.1,  # очень низкий порог
    )
    # age — числовой, не должен удаляться несмотря на низкий порог
    assert "age" in df_result.columns
    assert "age" not in report["removed_high_cardinality"]


def test_minimize_empty_identifiers_no_changes(sample_df):
    """Пустой список и drop_high_cardinality=False → ничего не удаляется."""
    processor = DataProcessor(sample_df)
    df_result, report = processor.minimize(direct_identifiers=[])

    assert list(df_result.columns) == list(sample_df.columns)
    assert report["removed_direct_identifiers"] == []
    assert report["removed_high_cardinality"] == []
    assert report["columns_before"] == report["columns_after"]


def test_minimize_report_structure(sample_df):
    """Проверяем, что отчёт содержит все обязательные ключи."""
    processor = DataProcessor(sample_df)
    _, report = processor.minimize(direct_identifiers=["id"])

    required_keys = {
        "removed_direct_identifiers",
        "removed_high_cardinality",
        "columns_before",
        "columns_after",
        "rows_unchanged",
    }
    assert required_keys.issubset(report.keys())

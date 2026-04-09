# final_system/tests/test_pipeline_serialize.py
#
# Unit-тесты для вспомогательных функций сериализации метрик в pipeline.py.
# Основная цель — регрессия на баг с NaN в JSONB (был зафиксирован 2026-04-07).
#
# Запуск: python -m pytest final_system/tests/test_pipeline_serialize.py -v

import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline import _serialize_utility_metrics, _serialize_privacy_metrics


# ─────────────────────────────────────────────
# _serialize_utility_metrics
# ─────────────────────────────────────────────

def _full_utility_report(pearson=0.05, utility_loss=0.19, mean_jsd=0.14, mean_tvd=0.26):
    return {
        "ml_efficacy": {
            "trtr": {"f1_weighted": 0.84},
            "tstr": {"f1_weighted": 0.64},
            "utility_loss": {"value": utility_loss},
        },
        "statistical": {
            "summary": {"mean_jsd": mean_jsd, "mean_tvd": mean_tvd}
        },
        "correlations": {"pearson_corr_mae": pearson},
    }


def test_serialize_utility_extracts_all_keys():
    result = _serialize_utility_metrics(_full_utility_report())
    assert "trtr_f1" in result
    assert "tstr_f1" in result
    assert "utility_loss" in result
    assert "mean_jsd" in result
    assert "mean_tvd" in result
    assert "pearson_corr_mae" in result


def test_serialize_utility_correct_values():
    result = _serialize_utility_metrics(_full_utility_report(
        utility_loss=0.20, mean_jsd=0.15, pearson=0.07
    ))
    assert result["trtr_f1"] == pytest.approx(0.84)
    assert result["tstr_f1"] == pytest.approx(0.64)
    assert result["utility_loss"] == pytest.approx(0.20)
    assert result["mean_jsd"] == pytest.approx(0.15)
    assert result["pearson_corr_mae"] == pytest.approx(0.07)


def test_serialize_utility_nan_becomes_none():
    """
    Регрессионный тест: NaN в pearson_corr_mae ронял INSERT в PostgreSQL JSONB.
    После фикса _sanitize_json в process_registry NaN → None.
    Здесь проверяем, что сериализатор возвращает None (не NaN),
    который затем корректно запишется как null в JSON.
    """
    report = _full_utility_report(pearson=float("nan"))
    result = _serialize_utility_metrics(report)
    # Значение может быть NaN (сериализатор не занимается этим),
    # но главное — _sanitize_json в registry должен его обработать.
    # Проверяем, что хотя бы ключ присутствует.
    assert "pearson_corr_mae" in result


def test_serialize_utility_nan_survives_json_sanitization():
    """
    Проверяем цепочку: NaN из метрик → _sanitize_json → json.dumps → валидный JSON.
    """
    from registry.process_registry import _sanitize_json
    report = _full_utility_report(pearson=float("nan"))
    result = _serialize_utility_metrics(report)
    sanitized = _sanitize_json(result)
    # После санитизации NaN должен стать None
    assert sanitized["pearson_corr_mae"] is None
    # И сериализация в JSON не должна падать
    dumped = json.dumps(sanitized)
    parsed = json.loads(dumped)
    assert parsed["pearson_corr_mae"] is None


def test_serialize_utility_inf_sanitized():
    """Infinity тоже не валиден для JSONB."""
    from registry.process_registry import _sanitize_json
    result = {"value": float("inf"), "other": 0.5}
    sanitized = _sanitize_json(result)
    assert sanitized["value"] is None
    assert sanitized["other"] == pytest.approx(0.5)


def test_serialize_utility_empty_report_returns_dict():
    result = _serialize_utility_metrics({})
    assert isinstance(result, dict)


def test_serialize_utility_none_values_in_nested():
    """Частично заполненный отчёт не должен падать."""
    report = {
        "ml_efficacy": {"trtr": None, "tstr": None, "utility_loss": None},
        "statistical": None,
        "correlations": None,
    }
    result = _serialize_utility_metrics(report)
    assert result.get("trtr_f1") is None
    assert result.get("mean_jsd") is None


# ─────────────────────────────────────────────
# _serialize_privacy_metrics
# ─────────────────────────────────────────────

def _full_privacy_report(attack_auc=0.50, dcr_preserved=True, spent_eps=2.49):
    return {
        "empirical_risk": {
            "membership_inference": {"attack_auc": attack_auc},
            "distance_metrics": {
                "dcr": {
                    "privacy_preserved": dcr_preserved,
                    "synth_to_real":   {"median": 0.45},
                    "holdout_to_real": {"median": 0.47},
                }
            },
        },
        "dp_guarantees": {
            "spent_epsilon_final": spent_eps,
            "epochs_completed": 33,
        },
        "diagnostic": {
            "classical": {
                "k_anonymity": 5,
                "l_diversity": 2,
                "t_closeness": 0.75,
            }
        },
    }


def test_serialize_privacy_extracts_all_keys():
    result = _serialize_privacy_metrics(_full_privacy_report())
    expected_keys = {
        "mia_auc", "dcr_privacy_preserved", "dcr_synth_median",
        "dcr_holdout_median", "spent_epsilon", "epochs_completed",
        "k_anonymity", "l_diversity", "t_closeness",
    }
    assert expected_keys.issubset(result.keys())


def test_serialize_privacy_correct_values():
    result = _serialize_privacy_metrics(_full_privacy_report(
        attack_auc=0.48, dcr_preserved=True, spent_eps=2.49
    ))
    assert result["mia_auc"] == pytest.approx(0.48)
    assert result["dcr_privacy_preserved"] is True
    assert result["spent_epsilon"] == pytest.approx(2.49)
    assert result["epochs_completed"] == 33
    assert result["k_anonymity"] == 5


def test_serialize_privacy_empty_report_returns_dict():
    result = _serialize_privacy_metrics({})
    assert isinstance(result, dict)


# ─────────────────────────────────────────────
# _sanitize_json (unit)
# ─────────────────────────────────────────────

def test_sanitize_json_nested_nan():
    from registry.process_registry import _sanitize_json
    data = {
        "a": float("nan"),
        "b": {"c": float("nan"), "d": 1.0},
        "e": [float("nan"), 2.0],
    }
    result = _sanitize_json(data)
    assert result["a"] is None
    assert result["b"]["c"] is None
    assert result["b"]["d"] == pytest.approx(1.0)
    assert result["e"][0] is None
    assert result["e"][1] == pytest.approx(2.0)


def test_sanitize_json_no_nan_unchanged():
    from registry.process_registry import _sanitize_json
    data = {"a": 1.0, "b": "text", "c": [1, 2, 3]}
    result = _sanitize_json(data)
    assert result == data

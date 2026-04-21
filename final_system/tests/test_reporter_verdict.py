# final_system/tests/test_reporter_verdict.py
#
# Unit-тесты для логики вердикта Reporter._compute_verdict()
# Запуск: python -m pytest final_system/tests/test_reporter_verdict.py -v

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from services.reporting_service.reporter import Reporter, VerdictThresholds


# ─────────────────────────────────────────────
# Вспомогательные фикстуры с "хорошими" отчётами
# ─────────────────────────────────────────────

def _good_dp_report(is_dp_enabled=True, spent_eps=2.5):
    return {
        "dp_config": {"is_dp_enabled": is_dp_enabled},
        "dp_spent":  {"spent_epsilon_final": spent_eps, "epochs_completed": 33},
    }

def _good_utility_report(utility_loss=0.10, mean_jsd=0.15):
    return {
        "ml_efficacy": {
            "trtr": {"f1_weighted": 0.84},
            "tstr": {"f1_weighted": 0.75},
            "utility_loss": {"value": utility_loss},
        },
        "statistical": {"summary": {"mean_jsd": mean_jsd, "mean_tvd": 0.20}},
        "correlations": {"pearson_corr_mae": 0.05},
    }

def _good_privacy_report(attack_auc=0.51, dcr_privacy_preserved=True):
    return {
        "empirical_risk": {
            "membership_inference": {"attack_auc": attack_auc},
            "distance_metrics": {
                "dcr": {"privacy_preserved": dcr_privacy_preserved}
            },
        },
        "dp_guarantees": {"spent_epsilon_final": 2.5, "epochs_completed": 33},
        "diagnostic": {"classical": {"k_anonymity": 5, "l_diversity": 2}},
    }

@pytest.fixture
def reporter():
    return Reporter(thresholds=VerdictThresholds(
        max_utility_loss=0.25,
        max_mean_jsd=0.40,
        max_mia_auc=0.60,
        require_dcr_privacy_preserved=True,
        require_dp_enabled=True,
        max_spent_epsilon=None,
    ))


# ─────────────────────────────────────────────
# PASS
# ─────────────────────────────────────────────

def test_verdict_pass_all_good(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "PASS"
    assert report["verdict"]["issues"] == []
    assert report["verdict"]["utility_ok"] is True
    assert report["verdict"]["privacy_ok"] is True
    assert report["verdict"]["dp_ok"] is True


# ─────────────────────────────────────────────
# FAIL — utility
# ─────────────────────────────────────────────

def test_verdict_fail_utility_loss_too_high(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(utility_loss=0.50),  # > 0.25
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert report["verdict"]["utility_ok"] is False
    assert any("Utility Loss" in issue for issue in report["verdict"]["issues"])


def test_verdict_fail_mean_jsd_too_high(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(mean_jsd=0.50),  # > 0.40
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert any("JSD" in issue for issue in report["verdict"]["issues"])


# ─────────────────────────────────────────────
# FAIL — privacy
# ─────────────────────────────────────────────

def test_verdict_fail_mia_auc_too_high(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(attack_auc=0.75),  # > 0.60
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert report["verdict"]["privacy_ok"] is False
    assert any("MIA AUC" in issue for issue in report["verdict"]["issues"])


def test_verdict_fail_dcr_not_preserved(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(dcr_privacy_preserved=False),
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert any("DCR" in issue for issue in report["verdict"]["issues"])


# ─────────────────────────────────────────────
# FAIL — DP
# ─────────────────────────────────────────────

def test_verdict_fail_dp_disabled(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(is_dp_enabled=False),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert report["verdict"]["dp_ok"] is False
    assert any("DP" in issue for issue in report["verdict"]["issues"])


def test_verdict_fail_spent_epsilon_exceeded():
    reporter = Reporter(thresholds=VerdictThresholds(
        require_dp_enabled=True,
        max_spent_epsilon=2.0,  # жёсткий лимит
    ))
    report = reporter.build(
        dp_report=_good_dp_report(spent_eps=3.5),  # > 2.0
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "FAIL"
    assert any("epsilon" in issue.lower() for issue in report["verdict"]["issues"])


# ─────────────────────────────────────────────
# PARTIAL
# ─────────────────────────────────────────────

def test_verdict_partial_missing_privacy(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=None,
    )
    assert report["verdict"]["overall"] == "PARTIAL"
    assert report["verdict"]["privacy_ok"] is None


def test_verdict_partial_missing_utility(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=None,
        privacy_report=_good_privacy_report(),
    )
    assert report["verdict"]["overall"] == "PARTIAL"
    assert report["verdict"]["utility_ok"] is None


def test_verdict_partial_all_none(reporter):
    report = reporter.build(
        dp_report=None,
        utility_report=None,
        privacy_report=None,
    )
    assert report["verdict"]["overall"] == "PARTIAL"


# ─────────────────────────────────────────────
# Структура отчёта
# ─────────────────────────────────────────────

def test_report_contains_data_processing_section(reporter):
    from copy import deepcopy
    minimization = {
        "removed_direct_identifiers": ["id"],
        "removed_high_cardinality": [],
        "columns_before": 10,
        "columns_after": 9,
        "rows_unchanged": True,
    }
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(),
        minimization_report=minimization,
    )
    assert "data_processing" in report
    assert report["data_processing"]["minimization"] == minimization


def test_report_data_processing_empty_when_no_minimization(reporter):
    report = reporter.build(
        dp_report=_good_dp_report(),
        utility_report=_good_utility_report(),
        privacy_report=_good_privacy_report(),
        minimization_report=None,
    )
    assert report["data_processing"]["minimization"] == {}

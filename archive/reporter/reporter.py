"""
reporter.py

Финальная точка сборки результатов пайплайна синтетических данных.
Принимает отчеты от генератора, utility_evaluator и privacy_evaluator,
склеивает их в единый документ, выносит агрегированный вердикт
и сохраняет результат на диск и/или в БД через DataManager.

Не считает метрики самостоятельно — только оркестрирует готовые отчеты.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Пороговые значения для вердикта
# ─────────────────────────────────────────────

@dataclass
class VerdictThresholds:
    """
    Пороги для агрегированного вердикта PASS/FAIL.
    Все значения можно менять под конкретный датасет или требования ПНСТ.

    Логика:
    - utility: чем меньше utility_loss и mean_jsd, тем лучше
    - privacy: чем ниже attack_auc и выше dcr, тем лучше
    - dp:      наличие включенного DP и spent_epsilon в допустимом диапазоне
    """
    # Utility: максимально допустимая потеря ML-метрики (TRTR - TSTR)
    # ПНСТ раздел 4: < 25 % от TRTR F1
    max_utility_loss: float = 0.25

    # Utility: максимально допустимое среднее JSD по числовым колонкам
    # ПНСТ раздел 4: < 0.40
    max_mean_jsd: float = 0.40

    # Privacy: максимально допустимый AUC атаки MIA
    # ПНСТ раздел 4: ≤ 0.55 ("атака хуже или равна случайному угадыванию")
    max_mia_auc: float = 0.55

    # Privacy: синтетика должна быть не ближе к train, чем holdout (DCR)
    require_dcr_privacy_preserved: bool = True

    # DP: требовать ли включенный DP для вердикта PASS
    require_dp_enabled: bool = True

    # DP: максимально допустимый потраченный epsilon
    # None = не проверять
    max_spent_epsilon: Optional[float] = None


# ─────────────────────────────────────────────
# Репортер
# ─────────────────────────────────────────────

class Reporter:
    """
    Собирает финальный отчет пайплайна и сохраняет его.

    Использование:
        reporter = Reporter(thresholds=VerdictThresholds())
        report = reporter.build(
            dp_report=generator.privacy_report(),
            utility_report=utility_evaluator.evaluate(...),
            privacy_report=privacy_evaluator.evaluate(...),
            dataset_name="adult_census",
            generator_type="dpctgan",
        )
        reporter.save(report, output_dir="reports/")
    """

    def __init__(
        self,
        thresholds: Optional[VerdictThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or VerdictThresholds()

    # ──────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────

    def build(
        self,
        dp_report: Optional[Dict[str, Any]],
        utility_report: Optional[Dict[str, Any]],
        privacy_report: Optional[Dict[str, Any]],
        minimization_report: Optional[Dict[str, Any]] = None,
        dataset_name: str = "unknown",
        generator_type: str = "unknown",
        process_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Собирает финальный отчет из трех компонентов.

        Все три отчета опциональны: если модуль не запускался,
        соответствующий раздел будет None, а вердикт — PARTIAL.
        process_id — связующий ключ с записью в таблице processes DataManager.
        """
        report_id = str(uuid.uuid4())
        generated_at = datetime.now().isoformat()

        logger.info(
            f"[Reporter] Сборка отчета. "
            f"dataset='{dataset_name}', generator='{generator_type}', "
            f"report_id={report_id}"
        )

        verdict = self._compute_verdict(dp_report, utility_report, privacy_report)

        report = {
            "report_id": report_id,
            "process_id": process_id,
            "generated_at": generated_at,
            "dataset_name": dataset_name,
            "generator_type": generator_type,
            "verdict": verdict,
            "data_processing": {
                "minimization": minimization_report or {},
            },
            "generator": dp_report,
            "utility": utility_report,
            "privacy": privacy_report,
        }

        logger.info(
            f"[Reporter] Вердикт: {verdict['overall']}. "
            f"Проблемы: {verdict['issues'] or 'нет'}"
        )

        return report

    def save(
        self,
        report: Dict[str, Any],
        output_dir: str = "reports",
    ) -> str:
        """
        Сохраняет отчет на диск как JSON.
        Возвращает путь к сохраненному файлу.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Имя файла: dataset_generator_timestamp.json
        dataset = report.get("dataset_name", "unknown")
        generator = report.get("generator_type", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}__{generator}__{ts}.json"
        filepath = output_path / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[Reporter] Отчет сохранен: {filepath}")
        except Exception as e:
            raise IOError(f"Ошибка при сохранении отчета: {e}") from e

        return str(filepath)

    # ──────────────────────────────────────────
    # Внутренняя логика вердикта
    # ──────────────────────────────────────────

    def _compute_verdict(
        self,
        dp_report: Optional[Dict],
        utility_report: Optional[Dict],
        privacy_report: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Проверяет отчеты против порогов и выносит агрегированный вердикт.

        PASS    — все активные проверки пройдены
        FAIL    — одна или несколько проверок провалены
        PARTIAL — часть отчетов отсутствует, вердикт неполный
        """
        issues: List[str] = []
        checks: Dict[str, Optional[bool]] = {
            "utility_ok": None,
            "privacy_ok": None,
            "dp_ok": None,
        }

        # ── Проверка Utility ──────────────────
        if utility_report is not None:
            utility_issues = []

            ml = utility_report.get("ml_efficacy") or {}
            utility_loss_val = (ml.get("utility_loss") or {}).get("value")
            if utility_loss_val is not None:
                if utility_loss_val > self.thresholds.max_utility_loss:
                    utility_issues.append(
                        f"Utility Loss {utility_loss_val:.3f} > "
                        f"порога {self.thresholds.max_utility_loss}"
                    )

            stat = utility_report.get("statistical") or {}
            mean_jsd = (stat.get("summary") or {}).get("mean_jsd")
            if mean_jsd is not None:
                if mean_jsd > self.thresholds.max_mean_jsd:
                    utility_issues.append(
                        f"Mean JSD {mean_jsd:.4f} > порога {self.thresholds.max_mean_jsd}"
                    )

            checks["utility_ok"] = len(utility_issues) == 0
            issues.extend(utility_issues)

        # ── Проверка Privacy (эмпирическая) ──
        if privacy_report is not None:
            privacy_issues = []

            mia = (
                (privacy_report.get("empirical_risk") or {})
                .get("membership_inference") or {}
            )
            attack_auc = mia.get("attack_auc")
            if attack_auc is not None:
                if attack_auc > self.thresholds.max_mia_auc:
                    privacy_issues.append(
                        f"MIA AUC {attack_auc:.3f} > "
                        f"порога {self.thresholds.max_mia_auc} (риск утечки)"
                    )

            dcr = (
                (privacy_report.get("empirical_risk") or {})
                .get("distance_metrics") or {}
            ).get("dcr") or {}
            if self.thresholds.require_dcr_privacy_preserved:
                if dcr.get("privacy_preserved") is False:
                    privacy_issues.append(
                        "DCR: синтетика аномально близка к обучающим данным"
                    )

            checks["privacy_ok"] = len(privacy_issues) == 0
            issues.extend(privacy_issues)

        # ── Проверка DP-гарантий ──────────────
        if dp_report is not None:
            dp_issues = []

            # Ищем dp_config в двух возможных структурах:
            # 1) напрямую в dp_report (если передан dp_report от генератора)
            # 2) внутри privacy_report["dp_guarantees"] (если передан privacy_report)
            dp_config = dp_report.get("dp_config") or {}
            dp_spent = dp_report.get("dp_spent") or {}
            is_dp_enabled = dp_config.get("is_dp_enabled")

            if self.thresholds.require_dp_enabled and is_dp_enabled is False:
                dp_issues.append("DP отключен (disabled_dp=True)")

            spent_eps = dp_spent.get("spent_epsilon_final")
            if (
                self.thresholds.max_spent_epsilon is not None
                and spent_eps is not None
                and spent_eps > self.thresholds.max_spent_epsilon
            ):
                dp_issues.append(
                    f"Spent epsilon {spent_eps:.3f} > "
                    f"допустимого {self.thresholds.max_spent_epsilon}"
                )

            checks["dp_ok"] = len(dp_issues) == 0
            issues.extend(dp_issues)

        # ── Агрегированный результат ──────────
        # PARTIAL если хотя бы один модуль не запускался
        has_none = any(v is None for v in checks.values())
        all_passed = all(v is True for v in checks.values() if v is not None)

        if has_none and all_passed:
            overall = "PARTIAL"
        elif all_passed:
            overall = "PASS"
        else:
            overall = "FAIL"

        return {
            "overall": overall,
            **checks,
            "issues": issues,
        }


# ─────────────────────────────────────────────
# Тестовый блок
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    print("=== ТЕСТ REPORTER ===\n")

    # Имитируем готовые отчеты от всех трёх модулей
    mock_dp_report = {
        "synthesizer": "smartnoise_dpctgan",
        "status": "fitted",
        "dp_config": {
            "is_dp_enabled": True,
            "epsilon_initial": 3.0,
            "delta": 4.47e-05,
            "sigma": 5.0,
            "max_grad_norm": 1.0,
            "batch_size": 500,
        },
        "dp_spent": {
            "epsilon_target_after_preprocess": 1.5,
            "spent_epsilon_final": 2.48,
            "epochs_requested": 100,
            "epochs_completed": 87,
        },
        "reproducibility": {"random_seed": 42},
    }

    mock_utility_report = {
        "metadata": {"real_rows": 1000, "synth_rows": 1000, "eval_duration_sec": 0.28},
        "statistical": {
            "numerical": {
                "age": {"jsd": 0.063, "mean_delta": -3.97},
                "income": {"jsd": 0.016, "mean_delta": -1221.0},
            },
            "categorical": {
                "city": {"tvd": 0.232},
                "education": {"tvd": 0.129},
            },
            "summary": {"mean_jsd": 0.039, "mean_tvd": 0.18},
        },
        "correlations": {"pearson_corr_mae": 0.019, "cramers_v_mae": 0.082},
        "ml_efficacy": {
            "trtr": {"f1_weighted": 0.58, "roc_auc": 0.49},
            "tstr": {"f1_weighted": 0.57, "roc_auc": 0.50},
            "utility_loss": {
                "metric": "f1_weighted",
                "value": 0.0116,
                "interpretation": "ok (loss < 10%)",
            },
        },
    }

    mock_privacy_report = {
        "metadata": {"real_rows": 1000, "synth_rows": 1000, "eval_duration_sec": 0.66},
        "dp_guarantees": {
            "available": True,
            "is_dp_enabled": True,
            "spent_epsilon_final": 2.48,
            "epochs_completed": 87,
        },
        "empirical_risk": {
            "distance_metrics": {
                "dcr": {
                    "privacy_preserved": True,
                    "interpretation": "ok: синтетика не ближе к обучающим данным, чем holdout",
                },
                "nndr": {"synth_mean": 0.679, "share_below_0.1": 0.01},
            },
            "membership_inference": {
                "attack_auc": 0.516,
                "interpretation": "protected: атака не лучше случайного угадывания",
            },
        },
        "diagnostic": {
            "classical": {
                "k_anonymity": 12,
                "l_diversity": 3,
                "t_closeness": 0.12,
            }
        },
    }

    # Сценарий 1: всё хорошо → ожидаем PASS
    print("[1] Сценарий: нормальный запуск (ожидаем PASS)")
    reporter = Reporter(thresholds=VerdictThresholds())
    report = reporter.build(
        dp_report=mock_dp_report,
        utility_report=mock_utility_report,
        privacy_report=mock_privacy_report,
        dataset_name="adult_census",
        generator_type="dpctgan",
        process_id="test-process-001",
    )
    filepath = reporter.save(report, output_dir="reports")
    print(f"Файл: {filepath}")
    print(f"Вердикт: {report['verdict']}\n")

    # Сценарий 2: DP отключен → ожидаем FAIL
    print("[2] Сценарий: DP отключен (ожидаем FAIL)")
    bad_dp = dict(mock_dp_report)
    bad_dp["dp_config"] = {**mock_dp_report["dp_config"], "is_dp_enabled": False}
    report_fail = reporter.build(
        dp_report=bad_dp,
        utility_report=mock_utility_report,
        privacy_report=mock_privacy_report,
        dataset_name="adult_census",
        generator_type="ctgan_nodp",
    )
    print(f"Вердикт: {report_fail['verdict']}\n")

    # Сценарий 3: privacy_evaluator не запускался → ожидаем PARTIAL
    print("[3] Сценарий: privacy_evaluator пропущен (ожидаем PARTIAL)")
    report_partial = reporter.build(
        dp_report=mock_dp_report,
        utility_report=mock_utility_report,
        privacy_report=None,
        dataset_name="adult_census",
        generator_type="dpctgan",
    )
    print(f"Вердикт: {report_partial['verdict']}\n")

    print("=== ТЕСТ ЗАВЕРШЁН ===")

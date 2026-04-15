# shared/schemas/reporting.py
#
# Контракт Reporting Service: сборка финального отчёта.
# Используется: Reporting Service (потребляет), Gateway (отправляет).

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class VerdictThresholdsRequest(BaseModel):
    """
    Сериализуемая версия reporter.VerdictThresholds.
    Передаётся Gateway-ем при вызове POST /reports, чтобы Reporting Service
    применил пороги из конфига конкретного запуска.
    """
    max_utility_loss: float = 0.25
    max_mean_jsd: float = 0.40
    max_mia_auc: float = 0.6
    require_dcr_privacy_preserved: bool = True
    require_dp_enabled: bool = True
    max_spent_epsilon: Optional[float] = None


class ReportRequest(BaseModel):
    """
    Тело запроса POST /reports.

    Все три sub-отчёта опциональны: отсутствующий раздел даёт вердикт PARTIAL.
    """
    run_id: str
    dataset_name: str
    generator_type: str
    dp_report: Optional[Dict[str, Any]] = None
    utility_report: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    minimization_report: Optional[Dict[str, Any]] = None
    thresholds: VerdictThresholdsRequest = VerdictThresholdsRequest()

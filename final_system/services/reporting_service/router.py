# services/reporting_service/router.py

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # -> final_system/
from reporter.reporter import Reporter, VerdictThresholds
from shared.schemas.reporting import ReportRequest, ReportResponse
from services.reporting_service.settings import Settings, get_settings

router = APIRouter()


def _make_thresholds(req: ReportRequest) -> VerdictThresholds:
    t = req.thresholds
    return VerdictThresholds(
        max_utility_loss=t.max_utility_loss,
        max_mean_jsd=t.max_mean_jsd,
        max_mia_auc=t.max_mia_auc,
        require_dcr_privacy_preserved=t.require_dcr_privacy_preserved,
        require_dp_enabled=t.require_dp_enabled,
        max_spent_epsilon=t.max_spent_epsilon,
    )


# ── POST /reports ──────────────────────────────────────────────────────────────

@router.post(
    "/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Собрать финальный отчёт и вынести вердикт",
)
def create_report(
    body: ReportRequest,
    settings: Settings = Depends(get_settings),
) -> ReportResponse:
    thresholds = _make_thresholds(body)
    reporter = Reporter(thresholds=thresholds)

    report = reporter.build(
        dp_report=body.dp_report,
        utility_report=body.utility_report,
        privacy_report=body.privacy_report,
        minimization_report=body.minimization_report,
        dataset_name=body.dataset_name,
        generator_type=body.generator_type,
        process_id=body.run_id,
    )

    try:
        report_path = reporter.save(report, output_dir=str(settings.reports_dir))
    except IOError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "SAVE_ERROR", "message": str(e)},
        )

    return ReportResponse(report_path=report_path, report=report)

# api/routers/evaluations.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.dependencies import require_auth
from api.schemas.evaluations import EvaluationRequest, EvaluationResult
from api.settings import Settings, get_settings
from api.store import EvalRecord, EvalStatus, eval_store, run_store

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


# ──────────────────────────────────────────────────────────────────────────────
# POST /evaluations/privacy
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/privacy", response_model=EvaluationResult, status_code=status.HTTP_202_ACCEPTED)
def evaluate_privacy(
    body:             EvaluationRequest,
    background_tasks: BackgroundTasks,
    settings:         Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> EvaluationResult:
    _validate_eval_request(body, settings)
    evaluation_id = str(uuid.uuid4())
    record = EvalRecord(evaluation_id=evaluation_id, eval_type="privacy")
    eval_store.add(record)
    background_tasks.add_task(_run_privacy_eval, evaluation_id, body, settings)
    return _to_response(record)


# ──────────────────────────────────────────────────────────────────────────────
# POST /evaluations/utility
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/utility", response_model=EvaluationResult, status_code=status.HTTP_202_ACCEPTED)
def evaluate_utility(
    body:             EvaluationRequest,
    background_tasks: BackgroundTasks,
    settings:         Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> EvaluationResult:
    if not body.target_column:
        raise HTTPException(
            status_code=400,
            detail={"code": "VALIDATION_ERROR", "message": "target_column обязателен для utility-оценки"},
        )
    _validate_eval_request(body, settings)
    evaluation_id = str(uuid.uuid4())
    record = EvalRecord(evaluation_id=evaluation_id, eval_type="utility")
    eval_store.add(record)
    background_tasks.add_task(_run_utility_eval, evaluation_id, body, settings)
    return _to_response(record)


# ──────────────────────────────────────────────────────────────────────────────
# GET /evaluations/{evaluation_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{evaluation_id}", response_model=EvaluationResult)
def get_evaluation(
    evaluation_id: str,
    _: None = Depends(require_auth),
) -> EvaluationResult:
    record = eval_store.get(evaluation_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Оценка не найдена"})
    return _to_response(record)


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _validate_eval_request(body: EvaluationRequest, settings: Settings) -> None:
    if not (settings.data_dir / f"{body.real_dataset_name}.csv").exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Датасет '{body.real_dataset_name}' не найден"})
    run = run_store.get(body.synth_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Запуск '{body.synth_run_id}' не найден"})
    if not run.synth_path:
        raise HTTPException(status_code=409, detail={"code": "CONFLICT", "message": "Синтетика для этого запуска недоступна"})


def _to_response(record: EvalRecord) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=record.evaluation_id,
        eval_type=record.eval_type,
        status=record.status,
        created_at=record.created_at,
        finished_at=record.finished_at,
        report=record.report,
        error_message=record.error_message,
    )


def _run_privacy_eval(evaluation_id: str, body: EvaluationRequest, settings: Settings) -> None:
    eval_store.update(evaluation_id, status=EvalStatus.running)
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from evaluator.privacy.privacy_evaluator import PrivacyConfig, PrivacyEvaluator

        real_df = pd.read_csv(settings.data_dir / f"{body.real_dataset_name}.csv", na_values=["?"])
        run = run_store.get(body.synth_run_id)
        synth_df = pd.read_csv(run.synth_path)

        real_train, real_holdout = train_test_split(real_df, test_size=0.2, random_state=42)

        config = PrivacyConfig(
            quasi_identifiers=body.quasi_identifiers or [],
            sensitive_attribute=body.sensitive_attribute,
        )
        evaluator = PrivacyEvaluator(config)
        report = evaluator.evaluate(
            real_train_df=real_train,
            real_holdout_df=real_holdout,
            synth_df=synth_df,
            dp_report={},
        )
        eval_store.update(evaluation_id, status=EvalStatus.completed, report=report, finished_at=datetime.now(timezone.utc))
    except Exception as e:
        eval_store.update(evaluation_id, status=EvalStatus.failed, error_message=str(e), finished_at=datetime.now(timezone.utc))


def _run_utility_eval(evaluation_id: str, body: EvaluationRequest, settings: Settings) -> None:
    eval_store.update(evaluation_id, status=EvalStatus.running)
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from evaluator.utility.utility_evaluator import UtilityConfig, UtilityEvaluator

        real_df = pd.read_csv(settings.data_dir / f"{body.real_dataset_name}.csv", na_values=["?"])
        run = run_store.get(body.synth_run_id)
        synth_df = pd.read_csv(run.synth_path)

        real_train, real_holdout = train_test_split(real_df, test_size=0.2, random_state=42)

        config = UtilityConfig(target_column=body.target_column)
        evaluator = UtilityEvaluator(config)
        report = evaluator.evaluate(
            real_train_df=real_train,
            synth_df=synth_df,
            real_test_df=real_holdout,
        )
        eval_store.update(evaluation_id, status=EvalStatus.completed, report=report, finished_at=datetime.now(timezone.utc))
    except Exception as e:
        eval_store.update(evaluation_id, status=EvalStatus.failed, error_message=str(e), finished_at=datetime.now(timezone.utc))

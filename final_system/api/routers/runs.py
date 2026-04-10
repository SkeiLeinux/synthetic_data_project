# api/routers/runs.py

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from api.dependencies import require_auth
from api.schemas.runs import RunCreate, RunDetail, RunListResponse, RunSummary
from api.settings import Settings, get_settings
from api.store import RunRecord, RunStatus, run_store

router = APIRouter(prefix="/runs", tags=["runs"])


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=RunListResponse)
def list_runs(
    page:         int = Query(1, ge=1),
    per_page:     int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    verdict:      Optional[str] = Query(None),
    dataset_name: Optional[str] = Query(None),
    _: None = Depends(require_auth),
) -> RunListResponse:
    items, total = run_store.list(
        status=status_filter,
        verdict=verdict,
        dataset_name=dataset_name,
        page=page,
        per_page=per_page,
    )
    return RunListResponse(
        items=[RunSummary.from_record(r) for r in items],
        meta={
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": math.ceil(total / per_page) if total else 0,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /runs
# ──────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=RunSummary, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    body:             RunCreate,
    background_tasks: BackgroundTasks,
    settings:         Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> RunSummary:
    # Проверяем что датасет существует
    dataset_path = settings.data_dir / f"{body.dataset_name}.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Датасет '{body.dataset_name}' не найден. Загрузите его через POST /datasets",
            },
        )

    # Проверяем что конфиг существует
    config_path = settings.configs_dir / f"{body.config_name}.yaml"
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Конфиг '{body.config_name}' не найден. Загрузите его через POST /configs",
            },
        )

    run_id = str(uuid.uuid4())
    record = RunRecord(
        run_id=run_id,
        dataset_name=body.dataset_name,
        config_name=body.config_name,
        save_model=body.save_model,
        webhook_url=body.webhook_url,
        n_synth_rows=body.n_synth_rows,
    )
    run_store.add(record)

    background_tasks.add_task(
        _execute_pipeline,
        run_id=run_id,
        dataset_path=str(dataset_path),
        config_path=str(config_path),
        settings=settings,
    )

    return RunSummary.from_record(record)


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    _: None = Depends(require_auth),
) -> RunDetail:
    record = _get_or_404(run_id)
    return RunDetail.from_record(record)


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /runs/{run_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(
    run_id: str,
    _: None = Depends(require_auth),
) -> None:
    record = _get_or_404(run_id)

    if record.status == RunStatus.running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": "Нельзя удалить запуск со статусом running"},
        )

    # Помечаем как cancelled если ещё в очереди
    if record.status == RunStatus.queued:
        run_store.update(run_id, status=RunStatus.cancelled)

    run_store.delete(run_id)


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/report
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{run_id}/report")
def get_run_report(
    run_id: str,
    _: None = Depends(require_auth),
) -> Dict[str, Any]:
    record = _get_or_404(run_id)
    _require_finished(record)

    if record.report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Отчёт не найден"},
        )
    return record.report


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/logs
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{run_id}/logs")
def get_run_logs(
    run_id: str,
    tail:   Optional[int] = Query(None, ge=1, le=10000),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Any:
    _get_or_404(run_id)

    log_path = settings.base_dir / settings.log_path
    if not log_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Лог-файл не найден"},
        )

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if tail:
        lines = lines[-tail:]

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/synthetic
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{run_id}/synthetic")
def get_run_synthetic(
    run_id:  str,
    format:  str = Query("csv", pattern="^(csv|json)$"),
    _: None = Depends(require_auth),
) -> Any:
    record = _get_or_404(run_id)
    _require_finished(record)

    if not record.synth_path or not Path(record.synth_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Файл синтетики не найден"},
        )

    if format == "json":
        import pandas as pd
        from fastapi.responses import JSONResponse
        df = pd.read_csv(record.synth_path)
        return JSONResponse(content=df.to_dict(orient="records"))

    filename = f"{record.dataset_name}__synth__{record.run_id[:8]}.csv"
    return FileResponse(
        path=record.synth_path,
        media_type="text/csv",
        filename=filename,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /runs/{run_id}/synthetic  — догенерация из модели запуска
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{run_id}/synthetic")
def generate_more_synthetic(
    run_id:   str,
    body:     Dict[str, Any],
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Any:
    record = _get_or_404(run_id)
    _require_finished(record)

    if not record.model_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "MODEL_NOT_SAVED",
                "message": "Модель не сохранена. Повторите запуск с save_model: true",
            },
        )

    n_rows = body.get("n_rows")
    if not n_rows or n_rows <= 0:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": "n_rows должен быть > 0"})

    model_path = settings.models_dir / f"{record.model_id}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Файл модели не найден"})

    from synthesizer.dp_ctgan import DPCTGANGenerator
    generator = DPCTGANGenerator.load(str(model_path))
    synth_df = generator.sample(n_rows)

    import io
    buf = io.StringIO()
    synth_df.to_csv(buf, index=False)
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{record.dataset_name}_extra_{n_rows}.csv"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _get_or_404(run_id: str) -> RunRecord:
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Запуск {run_id!r} не найден"},
        )
    return record


def _require_finished(record: RunRecord) -> None:
    if record.status not in (RunStatus.completed, RunStatus.failed):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "RUN_NOT_FINISHED",
                "message": f"Ресурс недоступен — запуск ещё выполняется (status: {record.status})",
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Фоновая задача: выполнение пайплайна
# ──────────────────────────────────────────────────────────────────────────────

def _execute_pipeline(
    run_id:       str,
    dataset_path: str,
    config_path:  str,
    settings:     Settings,
) -> None:
    """
    Выполняется в фоновом потоке FastAPI BackgroundTasks.
    Обновляет RunRecord в run_store по мере выполнения.
    """
    import sys, os
    sys.path.insert(0, str(settings.base_dir))

    run_store.update(
        run_id,
        status=RunStatus.running,
        started_at=datetime.now(timezone.utc),
    )

    try:
        import pandas as pd
        from config_loader import load_config
        from data_service.data_io import DataIO
        from data_service.processor import DataProcessor
        from pipeline import run_pipeline

        record = run_store.get(run_id)

        # Загружаем конфиг
        cfg = load_config(config_path)
        cfg_raw = yaml.safe_load(open(config_path, encoding="utf-8"))
        run_store.update(run_id, config_snapshot=cfg_raw)

        # Загружаем датасет
        df = pd.read_csv(dataset_path, na_values=["?"])
        df.dropna(inplace=True)

        if cfg.pipeline.sample_size > 0:
            df = df.sample(cfg.pipeline.sample_size, random_state=cfg.pipeline.random_state).reset_index(drop=True)

        # Схема колонок
        if cfg.data_schema.is_auto:
            processor = DataProcessor(df)
            schema = processor.detect_column_types(
                exclude_columns=cfg.data_schema.exclude or None,
                force_categorical=[cfg.utility.target_column],
            )
            categorical_cols = schema.categorical
            continuous_cols  = schema.continuous
        else:
            categorical_cols = cfg.data_schema.categorical
            continuous_cols  = cfg.data_schema.continuous

        n_train_approx = int(len(df) * (1 - cfg.pipeline.holdout_size))
        n_synth_rows = record.n_synth_rows or cfg.get_n_synth_rows(n_train_approx)

        # Пути вывода
        synth_path = str(settings.data_dir / f"{record.dataset_name}_synth_{run_id[:8]}.csv")
        model_save_path = None
        model_id = None
        if record.save_model:
            model_id = str(uuid.uuid4())
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = str(settings.models_dir / f"{model_id}.pkl")

        output_dir = str(settings.reports_dir)
        log_path = str(settings.base_dir / settings.log_path)

        synth_df, report = run_pipeline(
            real_df=df,
            synth_config=cfg.get_generator_config(),
            privacy_config=cfg.get_privacy_config(),
            utility_config=cfg.get_utility_config(),
            categorical_columns=categorical_cols,
            continuous_columns=continuous_cols,
            n_synth_rows=n_synth_rows,
            dataset_name=cfg.pipeline.dataset_name,
            output_dir=output_dir,
            thresholds=cfg.get_thresholds(),
            run_preprocessing=cfg.pipeline.run_preprocessing,
            holdout_size=cfg.pipeline.holdout_size,
            random_state=cfg.pipeline.random_state,
            source_info=dataset_path,
            config_path=config_path,
            synth_output_path=synth_path,
            model_save_path=model_save_path,
            direct_identifiers=cfg.data_schema.direct_identifiers,
            drop_high_cardinality=cfg.data_schema.drop_high_cardinality,
            cardinality_threshold=cfg.data_schema.cardinality_threshold,
        )

        synth_df.to_csv(synth_path, index=False)

        verdict = report.get("verdict", {}).get("overall")

        run_store.update(
            run_id,
            status=RunStatus.completed,
            verdict=verdict,
            synth_path=synth_path,
            synth_rows=len(synth_df),
            report=report,
            report_path=output_dir,
            model_id=model_id,
            finished_at=datetime.now(timezone.utc),
        )

        # Webhook-уведомление
        record = run_store.get(run_id)
        if record and record.webhook_url:
            _send_webhook(record)

    except Exception as exc:
        run_store.update(
            run_id,
            status=RunStatus.failed,
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )


def _send_webhook(record: RunRecord) -> None:
    """POST-уведомление на webhook_url при завершении запуска."""
    try:
        import urllib.request, json
        payload = {
            "run_id": record.run_id,
            "status": record.status.value,
            "verdict": record.verdict,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            record.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # webhook — best-effort, не роняем пайплайн

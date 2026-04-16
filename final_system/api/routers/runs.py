# api/routers/runs.py

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from api.dependencies import require_auth
from api.schemas.runs import RunCreate, RunDetail, RunListResponse, RunSummary
from api.settings import Settings, get_settings
from api.store import RunRecord, RunStatus, run_store

router = APIRouter(prefix="/runs", tags=["runs"])
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GET /runs/active  — только Redis (активные и недавние запуски)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/active", response_model=RunListResponse)
def list_active_runs(
    page:          int = Query(1, ge=1),
    per_page:      int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    verdict:       Optional[str] = Query(None),
    dataset_name:  Optional[str] = Query(None),
    _: None = Depends(require_auth),
) -> RunListResponse:
    """Возвращает только запуски из Redis (активные + завершённые в пределах TTL 1ч)."""
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
# GET /runs  — Redis + PostgreSQL (полная история)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=RunListResponse)
def list_all_runs(
    page:          int = Query(1, ge=1),
    per_page:      int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    verdict:       Optional[str] = Query(None),
    dataset_name:  Optional[str] = Query(None),
    settings:      Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> RunListResponse:
    """Возвращает все запуски: Redis (активные) + PostgreSQL (история).

    Если один и тот же run_id присутствует в обоих источниках,
    берётся запись из Redis (содержит больше деталей).
    """
    # 1. Все записи из Redis
    redis_items, _ = run_store.list(page=1, per_page=10_000)
    redis_by_id: Dict[str, RunSummary] = {
        r.run_id: RunSummary.from_record(r) for r in redis_items
    }

    # 2. Записи из PostgreSQL (только если БД включена)
    pg_items: list[RunSummary] = []
    if not settings.db_disabled:
        pg_items = _list_pg_runs(settings)

    # 3. Объединение: Redis имеет приоритет
    merged: Dict[str, RunSummary] = {}
    for item in pg_items:
        merged[item.run_id] = item
    merged.update(redis_by_id)  # Redis перезаписывает PG при совпадении run_id

    all_items = sorted(merged.values(), key=lambda r: r.created_at, reverse=True)

    # 4. Фильтрация
    if status_filter:
        all_items = [r for r in all_items if r.status.value == status_filter]
    if verdict:
        all_items = [r for r in all_items if r.verdict == verdict]
    if dataset_name:
        all_items = [r for r in all_items if r.dataset_name == dataset_name]

    # 5. Пагинация
    total = len(all_items)
    offset = (page - 1) * per_page
    page_items = all_items[offset: offset + per_page]

    return RunListResponse(
        items=page_items,
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
        quick_test=body.quick_test,
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

    if record.status in (RunStatus.queued, RunStatus.running):
        # Помечаем как cancelled — запись остаётся, клиент видит финальный статус
        run_store.update(run_id, status=RunStatus.cancelled,
                         finished_at=datetime.now(timezone.utc))
        return

    # Для завершённых (completed / failed / cancelled) — удаляем физически
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

    import io

    if settings.microservices_enabled:
        # Делегируем генерацию в Synthesis Service
        from api.clients import ServiceClient
        synth_cli = ServiceClient(settings.synthesis_service_url, timeout=300)
        result = synth_cli.post(f"/api/v1/models/{record.model_id}/sample", json={"n_rows": n_rows})
        synth_path = Path("/data") / result["synth_path"]
    else:
        # Монолитный fallback
        model_path = settings.models_dir / f"{record.model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Файл модели не найден"})
        from synthesizer.loader import load_generator
        import pandas as pd
        generator = load_generator(str(model_path))
        synth_df = generator.sample(n_rows)
        buf = io.StringIO()
        synth_df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{record.dataset_name}_extra_{n_rows}.csv"'},
        )

    if not synth_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Файл синтетики не найден"})

    import pandas as pd
    synth_df = pd.read_csv(synth_path)
    buf = io.StringIO()
    synth_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{record.dataset_name}_extra_{n_rows}.csv"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _list_pg_runs(settings: Settings) -> list[RunSummary]:
    """Читает все записи из таблицы processes в PostgreSQL и конвертирует в RunSummary."""
    try:
        from sqlalchemy import create_engine, text as sa_text
        engine = create_engine(
            f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
        schema = settings.db_schema
        sql = sa_text(f"""
            SELECT process_id, start_time, end_time, status,
                   source_data_info, config_rout
            FROM {schema}.processes
            ORDER BY start_time DESC
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()
        engine.dispose()
        return [RunSummary.from_pg_row(dict(row)) for row in rows]
    except Exception as e:
        logger.warning("_list_pg_runs: DB query failed: %s", e)
        return []  # БД недоступна — возвращаем пустой список, не роняем API

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
    quick_test:   bool = False,
) -> None:
    """
    Выполняется в фоновом потоке FastAPI BackgroundTasks.
    Если все URL микросервисов заданы — оркестрирует через HTTP.
    Иначе — запускает монолитный пайплайн напрямую (fallback).
    """
    if settings.microservices_enabled:
        _execute_pipeline_microservices(run_id, dataset_path, config_path, settings, quick_test)
    else:
        _execute_pipeline_monolith(run_id, dataset_path, config_path, settings, quick_test)


def _execute_pipeline_microservices(
    run_id:       str,
    dataset_path: str,
    config_path:  str,
    settings:     Settings,
    quick_test:   bool = False,
) -> None:
    """Оркестрирует пайплайн через HTTP-вызовы к микросервисам."""
    import sys
    sys.path.insert(0, str(settings.base_dir))

    from api.clients import ServiceClient, poll_synthesis_job
    from config_loader import load_config, apply_quick_test

    run_store.update(run_id, status=RunStatus.running, started_at=datetime.now(timezone.utc))
    logger.info("[run %s] Pipeline started: dataset=%s config=%s", run_id, dataset_path, config_path)

    try:
        record = run_store.get(run_id)

        cfg = load_config(config_path)
        if quick_test:
            cfg = apply_quick_test(cfg)
        cfg_raw = yaml.safe_load(open(config_path, encoding="utf-8"))
        run_store.update(run_id, config_snapshot=cfg_raw)

        data_cli  = ServiceClient(settings.data_service_url,      timeout=60)
        synth_cli = ServiceClient(settings.synthesis_service_url,  timeout=60)
        eval_cli  = ServiceClient(settings.evaluation_service_url, timeout=300)
        rep_cli   = ServiceClient(settings.reporting_service_url,  timeout=60)

        # 1. Загрузка датасета в Data Service
        logger.info("[run %s] Step 1/7: uploading dataset", run_id)
        dataset_meta = data_cli.post_file("/api/v1/datasets", dataset_path)
        dataset_id = dataset_meta["dataset_id"]
        logger.info("[run %s] Step 1/7 done: dataset_id=%s rows=%s", run_id, dataset_id, dataset_meta.get("rows"))

        # 2. Предобработка + holdout split
        logger.info("[run %s] Step 2/7: preprocessing + holdout split", run_id)
        force_cat = list(cfg.data_schema.categorical)
        if cfg.utility.target_column and cfg.utility.target_column not in force_cat:
            force_cat.append(cfg.utility.target_column)

        split_meta = data_cli.post(f"/api/v1/datasets/{dataset_id}/split", json={
            "holdout_size":          cfg.pipeline.holdout_size,
            "random_state":          cfg.pipeline.random_state,
            "sample_size":           cfg.pipeline.sample_size,
            "target_column":         cfg.utility.target_column,
            "force_categorical":     force_cat,
            "force_continuous":      cfg.data_schema.continuous,
            "exclude_columns":       cfg.data_schema.exclude,
            "direct_identifiers":    cfg.data_schema.direct_identifiers,
            "drop_high_cardinality": cfg.data_schema.drop_high_cardinality,
            "cardinality_threshold": cfg.data_schema.cardinality_threshold,
            "na_values":             ["?"],
        })
        split_id = split_meta["split_id"]
        logger.info("[run %s] Step 2/7 done: split_id=%s train=%s holdout=%s",
                    run_id, split_id, split_meta.get("train_rows"), split_meta.get("holdout_rows"))

        # 3. Запуск джоба синтеза
        logger.info("[run %s] Step 3/7: starting synthesis job", run_id)
        config_name = record.config_name
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        job = synth_cli.post("/api/v1/jobs", json={
            "split_id":   split_id,
            "config_name": config_name,
            "n_rows":     record.n_synth_rows,
            "save_model": record.save_model,
        })
        job_id = job["job_id"]
        logger.info("[run %s] Step 3/7 done: job_id=%s", run_id, job_id)

        # 4. Ожидание завершения синтеза (polling)
        poll_interval = 5 if quick_test else 10
        logger.info("[run %s] Step 4/7: waiting for synthesis (polling every %ds)...", run_id, poll_interval)
        job = poll_synthesis_job(synth_cli, job_id, poll_interval=poll_interval, timeout=7200)

        synth_path = job["synth_path"]      # относительный путь на shared volume
        dp_report  = job.get("dp_report")
        model_id   = job.get("model_id")
        logger.info("[run %s] Step 4/7 done: synth_path=%s", run_id, synth_path)

        # 5. Оценка приватности
        logger.info("[run %s] Step 5/7: privacy evaluation", run_id)
        privacy_report = eval_cli.post("/api/v1/evaluate/privacy", json={
            "split_id":           split_id,
            "synth_path":         synth_path,
            "dp_report":          dp_report,
            "quasi_identifiers":  cfg.privacy.quasi_identifiers,
            "sensitive_attribute": cfg.privacy.sensitive_attribute,
        })
        logger.info("[run %s] Step 5/7 done", run_id)

        # 6. Оценка полезности
        logger.info("[run %s] Step 6/7: utility evaluation", run_id)
        utility_report = eval_cli.post("/api/v1/evaluate/utility", json={
            "split_id":           split_id,
            "synth_path":         synth_path,
            "target_column":      cfg.utility.target_column,
            "categorical_columns": split_meta["categorical_columns"],
            "continuous_columns":  split_meta["continuous_columns"],
        })
        logger.info("[run %s] Step 6/7 done", run_id)

        # 7. Финальный отчёт
        logger.info("[run %s] Step 7/7: building report", run_id)
        thresholds = cfg.get_thresholds()
        rep_resp = rep_cli.post("/api/v1/reports", json={
            "run_id":          run_id,
            "dataset_name":    cfg.pipeline.dataset_name,
            "generator_type":  cfg.generator.generator_type,
            "dp_report":       dp_report,
            "utility_report":  utility_report,
            "privacy_report":  privacy_report,
            "thresholds": {
                "max_utility_loss":             thresholds.max_utility_loss,
                "max_mean_jsd":                 thresholds.max_mean_jsd,
                "max_mia_auc":                  thresholds.max_mia_auc,
                "require_dcr_privacy_preserved": thresholds.require_dcr_privacy_preserved,
                "require_dp_enabled":            thresholds.require_dp_enabled,
                "max_spent_epsilon":             thresholds.max_spent_epsilon,
            },
        })
        report      = rep_resp["report"]
        report_path = rep_resp["report_path"]
        verdict     = report.get("verdict", {}).get("overall", "?")
        logger.info("[run %s] Step 7/7 done: verdict=%s report=%s", run_id, verdict, report_path)

        # synth_path — относительный ("synth/{job_id}/synthetic.csv"),
        # Gateway смонтировал shared_data:/data, поэтому абсолютный путь /data/...
        abs_synth = str(Path("/data") / synth_path)

        verdict = report.get("verdict", {}).get("overall")
        run_store.update(
            run_id,
            status=RunStatus.completed,
            verdict=verdict,
            synth_path=abs_synth,
            synth_rows=split_meta.get("train_rows"),
            report=report,
            report_path=report_path,
            model_id=model_id,
            finished_at=datetime.now(timezone.utc),
        )

        record = run_store.get(run_id)
        if record and record.webhook_url:
            _send_webhook(record)

    except Exception as exc:
        logger.error("[run %s] Pipeline failed: %s", run_id, exc, exc_info=True)
        run_store.update(
            run_id,
            status=RunStatus.failed,
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )
        run_store.expire(run_id, 3600)


def _execute_pipeline_monolith(
    run_id:       str,
    dataset_path: str,
    config_path:  str,
    settings:     Settings,
    quick_test:   bool = False,
) -> None:
    """Fallback: запускает пайплайн локально (старый монолитный режим)."""
    import sys
    sys.path.insert(0, str(settings.base_dir))

    run_store.update(
        run_id,
        status=RunStatus.running,
        started_at=datetime.now(timezone.utc),
    )

    registry = None
    try:
        import pandas as pd
        from config_loader import load_config, apply_quick_test
        from data_service.processor import DataProcessor
        from pipeline import run_pipeline

        record = run_store.get(run_id)

        cfg = load_config(config_path)
        if quick_test:
            cfg = apply_quick_test(cfg)
        cfg_raw = yaml.safe_load(open(config_path, encoding="utf-8"))

        if not settings.db_disabled and cfg.database is not None:
            from registry.process_registry import ProcessRegistry
            cfg.database.host = settings.db_host
            try:
                registry = ProcessRegistry(cfg)
                if not registry.test_connection():
                    registry = None
            except Exception:
                registry = None
        run_store.update(run_id, config_snapshot=cfg_raw)

        df = pd.read_csv(dataset_path, na_values=["?"])
        df.dropna(inplace=True)

        if cfg.pipeline.sample_size > 0:
            df = df.sample(cfg.pipeline.sample_size, random_state=cfg.pipeline.random_state).reset_index(drop=True)

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

        synth_path = str(settings.data_dir / f"{record.dataset_name}_synth_{run_id[:8]}.csv")
        model_save_path = None
        model_id = None
        if record.save_model:
            model_id = str(uuid.uuid4())
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = str(settings.models_dir / f"{model_id}.pkl")

        output_dir = str(settings.reports_dir)

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
            registry=registry,
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

        record = run_store.get(run_id)
        if record and record.webhook_url:
            _send_webhook(record)

        if registry is not None:
            run_store.expire(run_id, 3600)

    except Exception as exc:
        run_store.update(
            run_id,
            status=RunStatus.failed,
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )
        if registry is not None:
            run_store.expire(run_id, 3600)


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
    except Exception as e:
        logger.warning("[run %s] Webhook delivery failed: %s", record.run_id, e)

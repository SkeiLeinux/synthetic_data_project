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
from shared.log_context import set_run_id

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
    run_id:   str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> None:
    record = _get_or_404(run_id)

    if record.status in (RunStatus.queued, RunStatus.running):
        # Останавливаем активный джоб синтеза (если уже запущен)
        if record.current_job_id:
            try:
                from api.clients import ServiceClient
                synth_cli = ServiceClient(settings.synthesis_service_url, timeout=10)
                synth_cli.delete(f"/api/v1/jobs/{record.current_job_id}")
                logger.info("Synthesis job %s cancel requested", record.current_job_id)
            except Exception as e:
                logger.warning("Could not cancel synthesis job: %s", e)
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

    # Собираем лог-файлы: gateway (base_dir/log_path) + все сервисы (/data/logs/)
    log_files = [settings.base_dir / settings.log_path]
    service_log_dir = settings.data_root / "logs"
    for name in ("data_service", "synthesis_service", "evaluation_service", "reporting_service"):
        log_files.append(service_log_dir / f"{name}.log")

    all_lines: list[str] = []
    for log_path in log_files:
        if not log_path.exists():
            continue
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                all_lines.extend(f.readlines())
        except OSError:
            pass

    if not all_lines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Лог-файлы не найдены"},
        )

    # Сортируем все строки по временной метке (формат: [YYYY-MM-DD HH:MM:SS])
    def _ts(line: str) -> str:
        # Извлекаем первый токен вида [2024-01-01 00:00:00] для сортировки
        if line.startswith("[") and "] " in line:
            return line[1: line.index("]")]
        return ""

    all_lines.sort(key=_ts)

    # Фильтруем только строки, относящиеся к данному run_id.
    # Трейсбеки (строки с отступом) прикрепляем к предыдущей строке-с-run_id.
    filtered: list[str] = []
    inside_run = False
    for line in all_lines:
        if run_id in line:
            filtered.append(line)
            inside_run = True
        elif inside_run and line.startswith((" ", "\t")):
            filtered.append(line)
        else:
            inside_run = False

    if tail:
        filtered = filtered[-tail:]

    return PlainTextResponse("".join(filtered))


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

    from api.clients import ServiceClient
    synth_cli = ServiceClient(settings.synthesis_service_url, timeout=300)
    result = synth_cli.post(f"/api/v1/models/{record.model_id}/sample", json={"n_rows": n_rows})
    synth_path = Path("/data") / result["synth_path"]

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
    """Оркестрирует пайплайн через HTTP-вызовы к микросервисам."""
    import sys
    sys.path.insert(0, str(settings.base_dir))

    from api.clients import ServiceClient, poll_synthesis_job
    from config_loader import load_config, apply_quick_test

    set_run_id(run_id)
    run_store.update(run_id, status=RunStatus.running, started_at=datetime.now(timezone.utc))
    logger.info("Pipeline started: dataset=%s config=%s", dataset_path, config_path)

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
        logger.info("Step 1/7: uploading dataset")
        dataset_meta = data_cli.post_file("/api/v1/datasets", dataset_path)
        dataset_id = dataset_meta["dataset_id"]
        logger.info("Step 1/7 done: dataset_id=%s rows=%s", dataset_id, dataset_meta.get("rows"))

        # 2. Предобработка + holdout split
        logger.info("Step 2/7: preprocessing + holdout split")
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
            "run_id":                run_id,
        })
        split_id = split_meta["split_id"]
        logger.info("Step 2/7 done: split_id=%s train=%s holdout=%s",
                    split_id, split_meta.get("train_rows"), split_meta.get("holdout_rows"))

        config_name = record.config_name
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

        max_iterations = getattr(cfg.pipeline, "max_iterations", 1)
        poll_interval  = 5 if quick_test else 10
        thresholds     = cfg.thresholds  # ThresholdsYamlConfig напрямую — без lazy-импорта reporter.reporter

        report = None
        report_path = None
        synth_path = None
        dp_report = None
        model_id = None

        for iteration in range(1, max_iterations + 1):
            iter_tag = f" (iteration {iteration}/{max_iterations})" if max_iterations > 1 else ""

            # 3. Запуск джоба синтеза
            logger.info("Step 3/7: starting synthesis job%s", iter_tag)
            job = synth_cli.post("/api/v1/jobs", json={
                "split_id":    split_id,
                "config_name": config_name,
                "n_rows":      record.n_synth_rows,
                "save_model":  record.save_model,
                "run_id":      run_id,
                "dataset_name": record.dataset_name,
            })
            job_id = job["job_id"]
            run_store.update(run_id, current_job_id=job_id)
            logger.info("Step 3/7 done: job_id=%s", job_id)

            # 4. Ожидание завершения синтеза (polling)
            logger.info("Step 4/7: waiting for synthesis (polling every %ds)%s...", poll_interval, iter_tag)
            job = poll_synthesis_job(synth_cli, job_id, poll_interval=poll_interval, timeout=7200)

            synth_path = job["synth_path"]
            dp_report  = job.get("dp_report")
            model_id   = job.get("model_id")
            logger.info("Step 4/7 done: synth_path=%s", synth_path)

            # 5. Оценка приватности
            logger.info("Step 5/7: privacy evaluation%s", iter_tag)
            privacy_report = eval_cli.post("/api/v1/evaluate/privacy", json={
                "split_id":           split_id,
                "synth_path":         synth_path,
                "dp_report":          dp_report,
                "quasi_identifiers":  cfg.privacy.quasi_identifiers,
                "sensitive_attribute": cfg.privacy.sensitive_attribute,
                "run_id":             run_id,
            })
            logger.info("Step 5/7 done")

            # 6. Оценка полезности
            logger.info("Step 6/7: utility evaluation%s", iter_tag)
            utility_report = eval_cli.post("/api/v1/evaluate/utility", json={
                "split_id":           split_id,
                "synth_path":         synth_path,
                "target_column":      cfg.utility.target_column,
                "categorical_columns": split_meta["categorical_columns"],
                "continuous_columns":  split_meta["continuous_columns"],
                "run_id":             run_id,
            })
            logger.info("Step 6/7 done")

            # 7. Финальный отчёт
            logger.info("Step 7/7: building report%s", iter_tag)
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
            logger.info("Step 7/7 done: verdict=%s report=%s%s", verdict, report_path, iter_tag)

            if verdict != "FAIL" or iteration >= max_iterations:
                break

            logger.info("Verdict is FAIL — retrying synthesis (iteration %d/%d)", iteration + 1, max_iterations)

        # FR-08.5: финализируем синтетику — переименовываем pending → final.
        pending_path = Path("/data") / synth_path   # .../synthetic_pending.csv
        final_rel    = synth_path.replace("synthetic_pending.csv", "synthetic.csv")
        final_path   = Path("/data") / final_rel
        try:
            pending_path.rename(final_path)
            synth_path = final_rel
            logger.info("Synth finalized: %s", final_rel)
        except OSError as e:
            logger.warning("Could not finalize synth file: %s", e)

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
        record = run_store.get(run_id)
        if record and record.status == RunStatus.cancelled:
            logger.info("Pipeline stopped: run was cancelled")
        else:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            run_store.update(
                run_id,
                status=RunStatus.failed,
                error_message=str(exc),
                finished_at=datetime.now(timezone.utc),
            )
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
        logger.warning("Webhook delivery failed: %s", e)

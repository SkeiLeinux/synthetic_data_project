# services/synthesis_service/router.py

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException, status

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # -> final_system/
from config_loader import GeneratorYamlConfig
from shared.schemas.datasets import SplitMeta
from shared.schemas.synthesis import SampleRequest, SynthesisJobCreate, SynthesisJobSummary
from services.synthesis_service.job_store import JobRecord, JobStatus, JobStore, job_store
from services.synthesis_service.settings import Settings, get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_generator(gen_yaml: GeneratorYamlConfig):
    """Dispatch GeneratorYamlConfig → конкретный генератор."""
    from synthesizer.dp_ctgan import DPCTGANGenerator
    from synthesizer.dp_tvae import DPTVAEGenerator
    from synthesizer.sdv_generators import CTGANGenerator, TVAEGenerator, CopulaGANGenerator

    t = gen_yaml.generator_type
    if t == "dpctgan":
        return DPCTGANGenerator(gen_yaml.to_dpctgan_config())
    elif t == "dptvae":
        return DPTVAEGenerator(gen_yaml.to_dptvae_config())
    elif t == "ctgan":
        return CTGANGenerator(gen_yaml.to_ctgan_config())
    elif t == "tvae":
        return TVAEGenerator(gen_yaml.to_tvae_config())
    elif t == "copulagan":
        return CopulaGANGenerator(gen_yaml.to_copulagan_config())
    else:
        raise ValueError(f"Неизвестный generator_type: {t!r}")


def _load_gen_yaml(config_path: Path) -> GeneratorYamlConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GeneratorYamlConfig.model_validate(raw.get("generator", {}))


def _resolve_config(settings: Settings, config_name: str) -> Path:
    """Поддерживает абсолютный путь, путь от configs_dir, или имя файла."""
    p = Path(config_name)
    if p.is_absolute() and p.exists():
        return p
    candidate = settings.configs_dir / config_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Конфиг не найден: {config_name} (искали в {settings.configs_dir})")


def _load_split_meta(settings: Settings, split_id: str) -> SplitMeta:
    meta_path = settings.splits_dir / split_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"SplitMeta не найдена: {meta_path}")
    return SplitMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def _job_to_summary(rec: JobRecord) -> SynthesisJobSummary:
    return SynthesisJobSummary(
        job_id=rec.job_id,
        status=rec.status,
        model_id=rec.model_id,
        synth_path=rec.synth_path,
        dp_report=rec.dp_report,
        error_message=rec.error_message,
        created_at=rec.created_at,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
    )


# ── background worker ─────────────────────────────────────────────────────────

def _run_job(job_id: str, body: SynthesisJobCreate, settings: Settings) -> None:
    store = job_store
    store.update(job_id, status=JobStatus.running, started_at=datetime.now(timezone.utc))
    t0 = time.time()
    logger.info("[job %s] Started: split_id=%s config=%s n_rows=%s", job_id, body.split_id, body.config_name, body.n_rows)
    try:
        # 1. Загрузка метаданных сплита
        meta = _load_split_meta(settings, body.split_id)
        train_df = pd.read_csv(settings.splits_dir / body.split_id / "train.csv")
        logger.info("[job %s] Loaded train set: %d rows, %d columns", job_id, len(train_df), len(train_df.columns))

        # 2. Загрузка конфига генератора
        config_path = _resolve_config(settings, body.config_name)
        gen_yaml = _load_gen_yaml(config_path)
        generator = _build_generator(gen_yaml)
        logger.info("[job %s] Generator: %s", job_id, gen_yaml.generator_type)

        # 3. Фильтрация колонок (на случай если после preprocessing их нет в train_df)
        cat_cols = [c for c in meta.categorical_columns if c in train_df.columns]
        cont_cols = [c for c in meta.continuous_columns if c in train_df.columns]

        # Дропаем неклассифицированные колонки — SDV отвергает любые колонки,
        # не указанные как categorical/continuous (зеркалит поведение pipeline.py)
        all_classified = set(cat_cols + cont_cols)
        train_df = train_df[[c for c in train_df.columns if c in all_classified]]
        logger.info("[job %s] Columns: cat=%d cont=%d total=%d", job_id, len(cat_cols), len(cont_cols), len(train_df.columns))

        # 4. Обучение
        logger.info("[job %s] Fitting generator...", job_id)
        t_fit = time.time()
        generator.fit(train_df, categorical_columns=cat_cols, continuous_columns=cont_cols)
        logger.info("[job %s] Fit done in %.1fs", job_id, time.time() - t_fit)

        # 5. Генерация
        n_rows = body.n_rows or len(train_df)
        logger.info("[job %s] Sampling %d rows...", job_id, n_rows)
        synth_df = generator.sample(n_rows)
        logger.info("[job %s] Sampled %d rows", job_id, len(synth_df))

        # 6. Сохранение синтетики на shared volume
        synth_dir = settings.synth_dir / job_id
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_rel = f"synth/{job_id}/synthetic.csv"
        synth_df.to_csv(settings.data_root / synth_rel, index=False)
        logger.info("[job %s] Synth saved: %s", job_id, synth_rel)

        # 7. Опционально: сохранение модели
        model_id: Optional[str] = None
        if body.save_model:
            model_id = str(uuid.uuid4())
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            generator.save(str(settings.models_dir / f"{model_id}.pkl"))
            logger.info("[job %s] Model saved: %s", job_id, model_id)

        store.update(
            job_id,
            status=JobStatus.done,
            synth_path=synth_rel,
            model_id=model_id,
            dp_report=generator.privacy_report(),
            finished_at=datetime.now(timezone.utc),
        )
        logger.info("[job %s] Done in %.1fs total", job_id, time.time() - t0)

    except Exception as exc:
        logger.error("[job %s] Failed after %.1fs: %s", job_id, time.time() - t0, exc, exc_info=True)
        store.update(
            job_id,
            status=JobStatus.failed,
            error_message=str(exc),
            finished_at=datetime.now(timezone.utc),
        )


# ── POST /jobs ────────────────────────────────────────────────────────────────

@router.post(
    "/jobs",
    response_model=SynthesisJobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить обучение генератора (фоновый поток)",
)
def create_job(
    body: SynthesisJobCreate,
    settings: Settings = Depends(get_settings),
) -> SynthesisJobSummary:
    job_id = str(uuid.uuid4())
    rec = JobRecord(job_id=job_id)
    job_store.add(rec)

    t = threading.Thread(target=_run_job, args=(job_id, body, settings), daemon=True)
    t.start()

    return _job_to_summary(rec)


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}",
    response_model=SynthesisJobSummary,
    summary="Статус джоба: queued / running / done / failed",
)
def get_job(job_id: str) -> SynthesisJobSummary:
    rec = job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Джоб не найден"})
    return _job_to_summary(rec)


# ── GET /jobs/{job_id}/dp_report ──────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/dp_report",
    summary="DP-отчёт обученной модели (доступен после status=done)",
)
def get_dp_report(job_id: str) -> Dict[str, Any]:
    rec = job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Джоб не найден"})
    if rec.status != JobStatus.done:
        raise HTTPException(status_code=409, detail={"code": "NOT_READY", "message": f"Джоб ещё не завершён: {rec.status}"})
    return rec.dp_report or {}


# ── POST /models/{model_id}/sample ────────────────────────────────────────────

@router.post(
    "/models/{model_id}/sample",
    summary="Генерация из сохранённой модели (без повторного обучения)",
)
def sample_from_model(
    model_id: str,
    body: SampleRequest,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    from synthesizer.loader import load_generator

    model_path = settings.models_dir / f"{model_id}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Модель не найдена: {model_id}"})

    try:
        generator = load_generator(str(model_path))
        synth_df = generator.sample(body.n_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "SAMPLE_ERROR", "message": str(e)})

    # Сохраняем результат
    out_job_id = body.job_id or str(uuid.uuid4())
    synth_dir = settings.synth_dir / out_job_id
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_rel = f"synth/{out_job_id}/synthetic.csv"
    synth_df.to_csv(settings.data_root / synth_rel, index=False)

    return {"synth_path": synth_rel, "rows": len(synth_df), "model_id": model_id}

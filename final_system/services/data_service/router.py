# services/data_service/router.py

from __future__ import annotations

import io
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → final_system/
from data_service.processor import DataProcessor
from shared.schemas.datasets import DatasetMeta, SplitMeta, SplitRequest
from services.data_service.settings import Settings, get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dataset_dir(settings: Settings, dataset_id: str) -> Path:
    return settings.datasets_dir / dataset_id


def _load_dataset_meta(dataset_dir: Path) -> DatasetMeta:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Датасет не найден"})
    return DatasetMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def _load_split_meta(splits_dir: Path, split_id: str) -> SplitMeta:
    meta_path = splits_dir / split_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Сплит не найден"})
    return SplitMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


# ── POST /datasets ─────────────────────────────────────────────────────────────

@router.post("/datasets", response_model=DatasetMeta, status_code=status.HTTP_201_CREATED,
             summary="Загрузить датасет")
async def upload_dataset(
    file: UploadFile = File(..., description="CSV-файл"),
    settings: Settings = Depends(get_settings),
) -> DatasetMeta:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": "Файл пуст"})

    # Проверяем что парсится как CSV
    try:
        pd.read_csv(io.BytesIO(content), nrows=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": f"Не удалось распарсить CSV: {e}"})

    dataset_id = str(uuid.uuid4())
    dataset_dir = _dataset_dir(settings, dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_path = dataset_dir / "raw.csv"
    raw_path.write_bytes(content)
    logger.info("Dataset uploaded: filename=%s dataset_id=%s size_bytes=%d", file.filename, dataset_id, len(content))

    # Считаем размерность
    df_head = pd.read_csv(io.BytesIO(content), nrows=0)
    with io.BytesIO(content) as f:
        row_count = sum(1 for _ in f) - 1

    meta = DatasetMeta(
        dataset_id=dataset_id,
        filename=file.filename or f"{dataset_id}.csv",
        rows=row_count,
        columns=len(df_head.columns),
        file_size_bytes=len(content),
        uploaded_at=datetime.now(timezone.utc),
    )
    (dataset_dir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    return meta


# ── GET /datasets/{dataset_id} ────────────────────────────────────────────────

@router.get("/datasets/{dataset_id}", response_model=DatasetMeta,
            summary="Метаданные датасета")
def get_dataset(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
) -> DatasetMeta:
    return _load_dataset_meta(_dataset_dir(settings, dataset_id))


# ── POST /datasets/{dataset_id}/split ────────────────────────────────────────

@router.post("/datasets/{dataset_id}/split", response_model=SplitMeta,
             status_code=status.HTTP_201_CREATED,
             summary="Предобработка + holdout split")
def split_dataset(
    dataset_id: str,
    body: SplitRequest,
    settings: Settings = Depends(get_settings),
) -> SplitMeta:
    dataset_dir = _dataset_dir(settings, dataset_id)
    raw_path = dataset_dir / "raw.csv"
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Датасет не найден"})

    logger.info("Split started: dataset_id=%s holdout=%.0f%% sample_size=%d", dataset_id, body.holdout_size * 100, body.sample_size)

    # 1. Загрузка
    df = pd.read_csv(raw_path, na_values=body.na_values or [])
    df.dropna(inplace=True)

    if body.sample_size > 0 and body.sample_size < len(df):
        df = df.sample(body.sample_size, random_state=body.random_state).reset_index(drop=True)

    # 2. Предобработка
    processor = DataProcessor(df)
    processor.preprocess()

    # 3. Минимизация
    minimization_report: Dict[str, Any] = {}
    if body.direct_identifiers:
        _, minimization_report = processor.minimize(
            direct_identifiers=body.direct_identifiers,
            drop_high_cardinality=body.drop_high_cardinality,
            cardinality_threshold=body.cardinality_threshold,
        )

    df_clean = processor.get()

    # 4. Определение схемы колонок
    force_cat = list(body.force_categorical)
    if body.target_column and body.target_column not in force_cat:
        force_cat.append(body.target_column)

    schema = processor.detect_column_types(
        exclude_columns=body.exclude_columns or None,
        force_categorical=force_cat or None,
        force_continuous=body.force_continuous or None,
    )

    # 5. Дропаем исключённые колонки из df_clean, чтобы train/holdout не содержали их.
    # schema.categorical / schema.continuous уже не включают exclude_columns, но сам df_clean
    # содержит их до этого момента.
    if body.exclude_columns:
        drop_cols = [c for c in body.exclude_columns if c in df_clean.columns]
        if drop_cols:
            df_clean = df_clean.drop(columns=drop_cols)

    # 6. Holdout split (атомарен с предобработкой — не повторять позже)
    from sklearn.model_selection import train_test_split
    train_df, holdout_df = train_test_split(
        df_clean,
        test_size=body.holdout_size,
        random_state=body.random_state,
    )

    # 7. Сохранение
    split_id = str(uuid.uuid4())
    split_dir = settings.splits_dir / split_id
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = f"splits/{split_id}/train.csv"
    holdout_path = f"splits/{split_id}/holdout.csv"

    train_df.to_csv(settings.data_root / train_path, index=False)
    holdout_df.to_csv(settings.data_root / holdout_path, index=False)

    meta = SplitMeta(
        split_id=split_id,
        dataset_id=dataset_id,
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        train_path=train_path,
        holdout_path=holdout_path,
        categorical_columns=schema.categorical,
        continuous_columns=schema.continuous,
        target_column=body.target_column,
        minimization_report=minimization_report or None,
        created_at=datetime.now(timezone.utc),
    )
    (split_dir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    logger.info(
        "Split done: split_id=%s train=%d holdout=%d cat=%d cont=%d exclude=%s",
        split_id, len(train_df), len(holdout_df),
        len(schema.categorical), len(schema.continuous),
        body.exclude_columns,
    )
    return meta


# ── GET /datasets/{dataset_id}/splits/{split_id}/train ───────────────────────

@router.get("/datasets/{dataset_id}/splits/{split_id}/train",
            summary="Скачать train.csv")
def get_train(
    dataset_id: str,
    split_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    meta = _load_split_meta(settings.splits_dir, split_id)
    if meta.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Сплит не найден"})
    path = settings.data_root / meta.train_path
    return FileResponse(path=str(path), media_type="text/csv", filename="train.csv")


# ── GET /datasets/{dataset_id}/splits/{split_id}/holdout ─────────────────────

@router.get("/datasets/{dataset_id}/splits/{split_id}/holdout",
            summary="Скачать holdout.csv")
def get_holdout(
    dataset_id: str,
    split_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    meta = _load_split_meta(settings.splits_dir, split_id)
    if meta.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Сплит не найден"})
    path = settings.data_root / meta.holdout_path
    return FileResponse(path=str(path), media_type="text/csv", filename="holdout.csv")


# ── GET /datasets/{dataset_id}/splits/{split_id} ─────────────────────────────

@router.get("/datasets/{dataset_id}/splits/{split_id}", response_model=SplitMeta,
            summary="Метаданные сплита")
def get_split_meta(
    dataset_id: str,
    split_id: str,
    settings: Settings = Depends(get_settings),
) -> SplitMeta:
    meta = _load_split_meta(settings.splits_dir, split_id)
    if meta.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Сплит не найден"})
    return meta

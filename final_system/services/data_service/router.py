# services/data_service/router.py

from __future__ import annotations

import io
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → final_system/
from data_processor.processor import DataProcessor
from shared.log_context import set_run_id
from shared.schemas.datasets import DatasetMeta, SplitMeta, SplitRequest
from services.data_service.settings import Settings, get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_profile(df: pd.DataFrame, schema) -> dict:
    """Строит JSON-профиль предобработанного датасета."""
    columns = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct   = round(null_count / len(df), 4) if len(df) else 0.0
        if col in schema.continuous:
            columns[col] = {
                "type":       "continuous",
                "null_count": null_count,
                "null_pct":   null_pct,
                "min":        float(df[col].min()),
                "max":        float(df[col].max()),
                "mean":       round(float(df[col].mean()), 4),
                "median":     float(df[col].median()),
                "std":        round(float(df[col].std()), 4),
            }
        elif col in schema.categorical:
            top = df[col].value_counts(dropna=False).head(5)
            columns[col] = {
                "type":       "categorical",
                "null_count": null_count,
                "null_pct":   null_pct,
                "n_unique":   int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top.items()},
            }
    return {
        "total_rows":           len(df),
        "total_columns":        len(df.columns),
        "categorical_columns":  schema.categorical,
        "continuous_columns":   schema.continuous,
        "columns":              columns,
    }



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


# ── POST /datasets/from-db ────────────────────────────────────────────────────

class DatasetFromDbRequest(BaseModel):
    dsn: str            # PostgreSQL DSN (передаётся gateway из env, не хранится)
    query: str          # SQL-запрос для выборки данных
    name: Optional[str] = None

@router.post("/datasets/from-db", response_model=DatasetMeta, status_code=status.HTTP_201_CREATED,
             summary="Импортировать датасет из PostgreSQL")
def upload_dataset_from_db(
    body: DatasetFromDbRequest,
    settings: Settings = Depends(get_settings),
) -> DatasetMeta:
    try:
        import sqlalchemy as sa
        engine = sa.create_engine(body.dsn)
        with engine.connect() as conn:
            df = pd.read_sql(sa.text(body.query), conn)
        engine.dispose()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"code": "DB_ERROR", "message": str(e)})

    if df.empty:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": "Запрос вернул пустой датасет"})

    dataset_id  = str(uuid.uuid4())
    dataset_dir = _dataset_dir(settings, dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    (dataset_dir / "raw.csv").write_bytes(csv_bytes)

    name = (body.name or f"db_import_{dataset_id[:8]}") + ".csv"
    meta = DatasetMeta(
        dataset_id=dataset_id,
        filename=name,
        rows=len(df),
        columns=len(df.columns),
        file_size_bytes=len(csv_bytes),
        uploaded_at=datetime.now(timezone.utc),
    )
    (dataset_dir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    logger.info("Dataset imported from DB: query=%r dataset_id=%s rows=%d cols=%d",
                body.query[:80], dataset_id, len(df), len(df.columns))
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

    set_run_id(body.run_id)
    logger.info("Split started: dataset_id=%s holdout=%.0f%% sample_size=%d", dataset_id, body.holdout_size * 100, body.sample_size)

    # 1. Загрузка (пропуски заполняются внутри DataProcessor.preprocess: median/mode)
    df = pd.read_csv(raw_path, na_values=body.na_values or [])

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
    # Стратификация по target_column сохраняет баланс классов в обоих подмножествах.
    from sklearn.model_selection import train_test_split
    stratify_col = None
    if body.target_column and body.target_column in df_clean.columns:
        stratify_col = df_clean[body.target_column]
    train_df, holdout_df = train_test_split(
        df_clean,
        test_size=body.holdout_size,
        random_state=body.random_state,
        stratify=stratify_col,
    )

    # 7. Сохранение
    split_id = str(uuid.uuid4())
    split_dir = settings.splits_dir / split_id
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = f"splits/{split_id}/train.csv"
    holdout_path = f"splits/{split_id}/holdout.csv"

    train_df.to_csv(settings.data_root / train_path, index=False)
    holdout_df.to_csv(settings.data_root / holdout_path, index=False)

    # FR-02.7: профиль предобработанного датасета (до разбивки, на df_clean)
    import json as _json
    profile_rel = f"splits/{split_id}/profile.json"
    profile_data = _build_profile(df_clean, schema)
    (settings.data_root / profile_rel).write_text(
        _json.dumps(profile_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
        profile_path=profile_rel,
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


# ── GET /datasets/{dataset_id}/splits/{split_id}/profile ─────────────────────

@router.get("/datasets/{dataset_id}/splits/{split_id}/profile",
            summary="Профиль предобработанного датасета")
def get_split_profile(
    dataset_id: str,
    split_id: str,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    import json as _json
    meta = _load_split_meta(settings.splits_dir, split_id)
    if meta.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Сплит не найден"})
    if not meta.profile_path:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Профиль недоступен"})
    path = settings.data_root / meta.profile_path
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Файл профиля не найден"})
    return _json.loads(path.read_text(encoding="utf-8"))


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

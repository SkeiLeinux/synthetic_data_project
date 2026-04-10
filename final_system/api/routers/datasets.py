# api/routers/datasets.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.dependencies import require_auth
from api.schemas.datasets import DatasetPreview, DatasetSchema, DatasetSummary
from api.settings import Settings, get_settings

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _dataset_summary(path: Path) -> DatasetSummary:
    stat = path.stat()
    rows = cols = None
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=0)
        cols = len(df.columns)
        # Считаем строки без загрузки в память
        with open(path, "rb") as f:
            rows = sum(1 for _ in f) - 1  # минус заголовок
    except Exception:
        pass
    return DatasetSummary(
        name=path.stem,
        uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        rows=rows,
        columns=cols,
        file_size_bytes=stat.st_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /datasets
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=Dict[str, Any])
def list_datasets(
    page:     int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Dict[str, Any]:
    paths = sorted(settings.data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Исключаем файлы синтетики (суффикс _synth_)
    paths = [p for p in paths if "_synth_" not in p.stem]
    total = len(paths)
    offset = (page - 1) * per_page
    return {
        "items": [_dataset_summary(p).model_dump() for p in paths[offset: offset + per_page]],
        "meta": {"total": total, "page": page, "per_page": per_page, "pages": math.ceil(total / per_page) if total else 0},
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /datasets
# ──────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name:      str = Form(...),
    file:      UploadFile = File(...),
    na_values: str = Form(""),
    settings:  Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> DatasetSummary:
    dest = settings.data_dir / f"{name}.csv"
    if dest.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Датасет '{name}' уже существует"},
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": "Файл пуст"})

    # Базовая проверка: парсится ли как CSV
    try:
        import io, pandas as pd
        na_list = [v.strip() for v in na_values.split(",") if v.strip()] or None
        pd.read_csv(io.BytesIO(content), nrows=5, na_values=na_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"code": "VALIDATION_ERROR", "message": f"Не удалось распарсить CSV: {e}"})

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return _dataset_summary(dest)


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /datasets/{name}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    name:     str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> None:
    path = settings.data_dir / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Датасет '{name}' не найден"})
    path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# GET /datasets/{name}/schema
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{name}/schema", response_model=DatasetSchema)
def get_dataset_schema(
    name:     str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> DatasetSchema:
    path = settings.data_dir / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Датасет '{name}' не найден"})

    import pandas as pd
    from data_service.processor import DataProcessor

    df = pd.read_csv(path, na_values=["?"])
    processor = DataProcessor(df)
    schema = processor.detect_column_types()

    return DatasetSchema(
        categorical=schema.categorical,
        continuous=schema.continuous,
        ignored=schema.ignored,
        detected_at=datetime.now(timezone.utc),
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /datasets/{name}/preview
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{name}/preview", response_model=DatasetPreview)
def preview_dataset(
    name:     str,
    body:     Dict[str, Any] = {},
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> DatasetPreview:
    path = settings.data_dir / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Датасет '{name}' не найден"})

    import pandas as pd

    n_rows = min(int(body.get("n_rows", 10)), 1000)
    df_head = pd.read_csv(path, na_values=["?"], nrows=n_rows)

    # Статистика
    stats: Dict[str, Any] = {}
    for col in df_head.columns:
        s = df_head[col]
        info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "unique_count": int(s.nunique()),
        }
        if s.dtype.kind in ("i", "f"):
            info["mean"] = round(float(s.mean()), 4) if not s.isnull().all() else None
            info["min"]  = str(s.min()) if not s.isnull().all() else None
            info["max"]  = str(s.max()) if not s.isnull().all() else None
        stats[col] = info

    return DatasetPreview(
        columns=list(df_head.columns),
        rows=df_head.where(df_head.notna(), None).to_dict(orient="records"),
        stats=stats,
    )

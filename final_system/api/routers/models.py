# api/routers/models.py

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from api.dependencies import require_auth
from api.schemas.models import ModelDetail, ModelSummary, SampleRequest
from api.settings import Settings, get_settings

router = APIRouter(prefix="/models", tags=["models"])


def _load_sidecar(pkl_path: Path) -> Dict[str, Any]:
    """Читает {model_id}.meta.json рядом с .pkl.

    Сайдкар пишет synthesis_service при сохранении модели. Gateway не
    десериализует .pkl — это развязывает его от synthesizer-классов.
    Отсутствующий/битый сайдкар трактуется как "метаданные недоступны".
    """
    meta_path = pkl_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sidecar_field(sidecar: Dict[str, Any], *path: str) -> Any:
    """Достаёт вложенное поле из sidecar (privacy_report.dp_config.epsilon_initial)."""
    cur: Any = sidecar
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _model_summary(path: Path) -> ModelSummary:
    stat = path.stat()
    sidecar = _load_sidecar(path)
    created_raw = sidecar.get("created_at")
    try:
        created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except ValueError:
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return ModelSummary(
        model_id=path.stem,
        name=path.stem,
        run_id=sidecar.get("run_id"),
        dataset_name=sidecar.get("dataset_name", "unknown"),
        created_at=created_at,
        file_size_bytes=sidecar.get("file_size_bytes", stat.st_size),
        epsilon=_sidecar_field(sidecar, "privacy_report", "dp_config", "epsilon_initial"),
        epochs_completed=_sidecar_field(sidecar, "privacy_report", "dp_spent", "epochs_completed"),
        spent_epsilon=_sidecar_field(sidecar, "privacy_report", "dp_spent", "spent_epsilon_final"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /models
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=Dict[str, Any])
def list_models(
    page:         int = Query(1, ge=1),
    per_page:     int = Query(20, ge=1, le=100),
    dataset_name: str | None = Query(None),
    settings:     Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Dict[str, Any]:
    paths = sorted(settings.models_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(paths)
    offset = (page - 1) * per_page
    items = [_model_summary(p).model_dump() for p in paths[offset: offset + per_page]]
    return {
        "items": items,
        "meta": {"total": total, "page": page, "per_page": per_page, "pages": math.ceil(total / per_page) if total else 0},
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /models/{model_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{model_id}", response_model=ModelDetail)
def get_model(
    model_id: str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> ModelDetail:
    path = settings.models_dir / f"{model_id}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Модель '{model_id}' не найдена"})

    stat = path.stat()
    sidecar = _load_sidecar(path)
    dp_config = _sidecar_field(sidecar, "privacy_report", "dp_config")
    dp_spent  = _sidecar_field(sidecar, "privacy_report", "dp_spent")
    sample_size = _sidecar_field(sidecar, "privacy_report", "data", "sample_size")

    created_raw = sidecar.get("created_at")
    try:
        created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except ValueError:
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return ModelDetail(
        model_id=path.stem,
        name=path.stem,
        run_id=sidecar.get("run_id"),
        dataset_name=sidecar.get("dataset_name", "unknown"),
        created_at=created_at,
        file_size_bytes=sidecar.get("file_size_bytes", stat.st_size),
        epsilon=(dp_config or {}).get("epsilon_initial"),
        epochs_completed=(dp_spent or {}).get("epochs_completed"),
        spent_epsilon=(dp_spent or {}).get("spent_epsilon_final"),
        dp_config=dp_config,
        dp_spent=dp_spent,
        sample_size=sample_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /models/{model_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> None:
    path = settings.models_dir / f"{model_id}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Модель '{model_id}' не найдена"})
    path.unlink()
    sidecar = path.with_suffix(".meta.json")
    if sidecar.exists():
        sidecar.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# POST /models/{model_id}/samples
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{model_id}/samples")
def sample_from_model(
    model_id: str,
    body:     SampleRequest,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Any:
    path = settings.models_dir / f"{model_id}.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Модель '{model_id}' не найдена"})

    from api.clients import ServiceClient
    synth_cli = ServiceClient(settings.synthesis_service_url, timeout=300)
    result = synth_cli.post(f"/api/v1/models/{model_id}/sample", json={"n_rows": body.n_rows})
    synth_path = Path("/data") / result["synth_path"]
    if not synth_path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Файл синтетики не найден"})

    import pandas as pd
    synth_df = pd.read_csv(synth_path)

    if body.output_format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(content=synth_df.to_dict(orient="records"))

    buf = io.StringIO()
    synth_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{model_id}_{body.n_rows}rows.csv"'},
    )

# api/routers/models.py

from __future__ import annotations

import io
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from api.dependencies import require_auth
from api.schemas.models import ModelDetail, ModelSummary, SampleRequest
from api.settings import Settings, get_settings

router = APIRouter(prefix="/models", tags=["models"])


def _load_metadata(path: Path) -> Dict[str, Any]:
    """Извлекает метаданные из pkl без полной десериализации весов."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload
    except Exception:
        return {}


def _model_summary(path: Path, run_id: str | None = None, dataset_name: str = "unknown") -> ModelSummary:
    stat = path.stat()
    meta = _load_metadata(path)
    dp_spent = meta.get("dp_spent") or {}
    cfg = meta.get("config")
    return ModelSummary(
        model_id=path.stem,
        name=path.stem,
        run_id=run_id,
        dataset_name=dataset_name,
        created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        file_size_bytes=stat.st_size,
        epsilon=cfg.epsilon if cfg else None,
        epochs_completed=meta.get("epochs_completed"),
        spent_epsilon=meta.get("spent_epsilon"),
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
    meta = _load_metadata(path)
    cfg = meta.get("config")
    dp_config = None
    if cfg:
        dp_config = {
            "epsilon_initial": cfg.epsilon,
            "delta": meta.get("delta_used"),
            "sigma": cfg.sigma,
            "is_dp_enabled": not cfg.disabled_dp,
        }

    return ModelDetail(
        model_id=path.stem,
        name=path.stem,
        run_id=None,
        dataset_name="unknown",
        created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        file_size_bytes=stat.st_size,
        epsilon=cfg.epsilon if cfg else None,
        epochs_completed=meta.get("epochs_completed"),
        spent_epsilon=meta.get("spent_epsilon"),
        dp_config=dp_config,
        dp_spent={
            "spent_epsilon_final": meta.get("spent_epsilon"),
            "epochs_completed": meta.get("epochs_completed"),
        },
        sample_size=meta.get("sample_size"),
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

    from synthesizer.dp_ctgan import DPCTGANGenerator
    generator = DPCTGANGenerator.load(str(path))
    synth_df = generator.sample(body.n_rows)

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

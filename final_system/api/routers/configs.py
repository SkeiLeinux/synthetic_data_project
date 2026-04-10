# api/routers/configs.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import PlainTextResponse

from api.dependencies import require_auth
from api.schemas.configs import ConfigSummary, ConfigValidationResult
from api.settings import Settings, get_settings

router = APIRouter(prefix="/configs", tags=["configs"])


def _config_summary(path: Path) -> ConfigSummary:
    stat = path.stat()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        epsilon = (raw.get("generator") or {}).get("epsilon")
        epochs  = (raw.get("generator") or {}).get("epochs")
    except Exception:
        epsilon = epochs = None
    return ConfigSummary(
        name=path.stem,
        created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        epsilon=epsilon,
        epochs=epochs,
    )


def _validate_yaml(content: bytes) -> ConfigValidationResult:
    """Парсит и валидирует YAML через Pydantic AppConfig."""
    try:
        raw = yaml.safe_load(content.decode("utf-8"))
    except yaml.YAMLError as e:
        return ConfigValidationResult(
            valid=False,
            errors=[{"field": "yaml", "message": str(e)}],
        )
    if not raw:
        return ConfigValidationResult(
            valid=False,
            errors=[{"field": "yaml", "message": "Файл пуст"}],
        )
    try:
        from config_loader import AppConfig
        AppConfig.model_validate(raw)
    except Exception as e:
        return ConfigValidationResult(
            valid=False,
            errors=[{"field": "config", "message": str(e)}],
        )

    warnings: List[str] = []
    gen = raw.get("generator") or {}
    eps = gen.get("epsilon", 3.0)
    prep = gen.get("preprocessor_eps", 0.5)
    if prep / eps > 0.3:
        warnings.append("preprocessor_eps занимает >30% от epsilon — останется мало бюджета на обучение")

    return ConfigValidationResult(valid=True, warnings=warnings)


# ──────────────────────────────────────────────────────────────────────────────
# GET /configs
# ──────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=Dict[str, Any])
def list_configs(
    page:     int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Dict[str, Any]:
    paths = sorted(settings.configs_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(paths)
    offset = (page - 1) * per_page
    page_paths = paths[offset: offset + per_page]
    return {
        "items": [_config_summary(p).model_dump() for p in page_paths],
        "meta": {"total": total, "page": page, "per_page": per_page, "pages": math.ceil(total / per_page) if total else 0},
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /configs/validate
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ConfigValidationResult)
async def validate_config(
    file: UploadFile = File(...),
    _: None = Depends(require_auth),
) -> ConfigValidationResult:
    content = await file.read()
    return _validate_yaml(content)


# ──────────────────────────────────────────────────────────────────────────────
# POST /configs
# ──────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=ConfigSummary, status_code=status.HTTP_201_CREATED)
async def create_config(
    name: str = Form(...),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> ConfigSummary:
    dest = settings.configs_dir / f"{name}.yaml"
    if dest.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Конфиг '{name}' уже существует"},
        )
    content = await file.read()
    result = _validate_yaml(content)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.model_dump())

    dest.write_bytes(content)
    return _config_summary(dest)


# ──────────────────────────────────────────────────────────────────────────────
# GET /configs/{name}
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/{name}")
def get_config(
    name:     str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> Any:
    path = settings.configs_dir / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Конфиг '{name}' не найден"})
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-yaml")


# ──────────────────────────────────────────────────────────────────────────────
# PUT /configs/{name}
# ──────────────────────────────────────────────────────────────────────────────

@router.put("/{name}", response_model=ConfigSummary)
async def replace_config(
    name:     str,
    file:     UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> ConfigSummary:
    path = settings.configs_dir / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Конфиг '{name}' не найден"})

    content = await file.read()
    result = _validate_yaml(content)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.model_dump())

    path.write_bytes(content)
    return _config_summary(path)


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /configs/{name}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(
    name:     str,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_auth),
) -> None:
    path = settings.configs_dir / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Конфиг '{name}' не найден"})
    path.unlink()

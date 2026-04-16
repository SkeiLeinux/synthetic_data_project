# api/main.py
#
# Точка входа FastAPI-приложения.
# Запуск: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# (выполнять из директории final_system/)

from __future__ import annotations

import logging
import sys
import os

# Гарантируем что final_system/ в sys.path при запуске через uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import configs, datasets, models, runs, system
from api.settings import get_settings

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Инициализация и завершение при старте/остановке сервиса."""
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # Добавляем файловый хендлер поверх stdout (logs/app.log монтируется в Docker)
    try:
        import pathlib
        log_path = pathlib.Path(settings.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)
    except Exception as e:
        _log.warning("Не удалось открыть лог-файл %s: %s", settings.log_path, e)

    _log.info("Gateway started (log_path=%s)", settings.log_path)

    yield  # сервис работает

    # cleanup (если потребуется)


app = FastAPI(
    title="Synthetic Data Generation Service",
    version="1.0.0",
    description=(
        "REST API для генерации конфиденциальных синтетических данных "
        "с применением дифференциальной приватности (DP-CTGAN)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS — разрешаем всё в dev; в prod ограничить origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Глобальный обработчик ошибок ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": str(exc)},
    )


# ── Роутеры ───────────────────────────────────────────────────────────────────

PREFIX = "/api/v1"

app.include_router(system.router,      prefix=PREFIX)
app.include_router(runs.router,        prefix=PREFIX)
app.include_router(models.router,      prefix=PREFIX)
app.include_router(configs.router,     prefix=PREFIX)
app.include_router(datasets.router,    prefix=PREFIX)


# ── Корневой редирект ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

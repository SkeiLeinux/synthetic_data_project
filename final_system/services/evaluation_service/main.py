# services/evaluation_service/main.py
#
# Evaluation Service — порт 8003
# Запуск: uvicorn services.evaluation_service.main:app --host 0.0.0.0 --port 8003

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from shared.log_context import RunIdFormatter, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    force=True,
)
for _h in logging.root.handlers:
    _h.setFormatter(RunIdFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.evaluation_service.router import router
from services.evaluation_service.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    _add_shared_log_handler(settings.data_root / "logs" / "evaluation_service.log")
    yield


def _add_shared_log_handler(log_path) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(RunIdFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logging.getLogger().addHandler(fh)
    except Exception as e:
        logging.getLogger(__name__).warning("Cannot open shared log %s: %s", log_path, e)


app = FastAPI(
    title="Evaluation Service",
    version="1.0.0",
    description="Оценка приватности и полезности синтетических данных.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "message": str(exc)})


app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok"}

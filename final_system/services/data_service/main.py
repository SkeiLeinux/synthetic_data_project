# services/data_service/main.py
#
# Data Service — порт 8001
# Запуск: uvicorn main:app --host 0.0.0.0 --port 8001
# (из директории final_system/ или через Docker)

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

# До старта uvicorn — иначе basicConfig будет no-op
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request

from services.data_service.router import router
from services.data_service.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Data Service",
    version="1.0.0",
    description="Загрузка датасетов, предобработка и holdout split.",
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

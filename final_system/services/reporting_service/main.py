# services/reporting_service/main.py
#
# Reporting Service — порт 8004
# Запуск: uvicorn services.reporting_service.main:app --host 0.0.0.0 --port 8004

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.reporting_service.router import router
from services.reporting_service.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Reporting Service",
    version="1.0.0",
    description="Сборка финального отчёта и вынесение вердикта PASS/FAIL/PARTIAL.",
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

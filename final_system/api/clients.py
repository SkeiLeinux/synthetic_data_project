# api/clients.py
#
# Синхронные HTTP-клиенты для обращения к микросервисам из фоновых потоков.
# Используют httpx (синхронный режим), т.к. _execute_pipeline запускается
# в threading.Thread / FastAPI BackgroundTasks, а не в asyncio.

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class ServiceClient:
    """Тонкая обёртка вокруг httpx для вызова одного микросервиса."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def get(self, path: str, **kwargs) -> Any:
        resp = httpx.get(f"{self._base}{path}", timeout=self._timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, **kwargs) -> Any:
        resp = httpx.post(f"{self._base}{path}", timeout=self._timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post_file(self, path: str, file_path: str, field: str = "file") -> Any:
        p = Path(file_path)
        with open(p, "rb") as f:
            resp = httpx.post(
                f"{self._base}{path}",
                files={field: (p.name, f, "text/csv")},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        return resp.json()


def poll_synthesis_job(
    client: ServiceClient,
    job_id: str,
    poll_interval: int = 10,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """
    Ждёт завершения джоба синтеза (done / failed).
    Возвращает финальный SynthesisJobSummary dict.
    Бросает RuntimeError если джоб упал или исчерпан таймаут.
    """
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        job = client.get(f"/api/v1/jobs/{job_id}")
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            raise RuntimeError(f"Synthesis job failed: {job.get('error_message')}")
    raise TimeoutError(f"Synthesis job {job_id} timed out after {timeout}s")

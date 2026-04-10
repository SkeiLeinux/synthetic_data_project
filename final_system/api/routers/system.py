# api/routers/system.py

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.store import run_store

router = APIRouter(tags=["system"])

_start_time = time.time()


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """Liveness check. Не требует авторизации."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_sec": int(time.time() - _start_time),
    }


@router.get("/health/db")
def health_db() -> Dict[str, Any]:
    """Readiness check — PostgreSQL."""
    try:
        from registry.process_registry import ProcessRegistry
        from api.settings import get_settings
        settings = get_settings()
        from config_loader import load_config
        cfg = load_config(str(settings.base_dir / settings.default_config))
        reg = ProcessRegistry(app_config=cfg)
        ok = reg.test_connection()
        reg.close()
        if ok:
            return {"component": "database", "status": "ok", "latency_ms": None, "message": None}
        return {"component": "database", "status": "unavailable", "latency_ms": None, "message": "test_connection failed"}
    except Exception as e:
        return {"component": "database", "status": "unavailable", "latency_ms": None, "message": str(e)}


@router.get("/health/gpu")
def health_gpu() -> Dict[str, Any]:
    """Readiness check — CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory // (1024 * 1024)
            free  = (props.total_memory - torch.cuda.memory_allocated(idx)) // (1024 * 1024)
            return {
                "component": "gpu",
                "status": "ok",
                "device_name": props.name,
                "vram_total_mb": total,
                "vram_free_mb": free,
            }
        return {"component": "gpu", "status": "unavailable", "device_name": None, "vram_total_mb": None, "vram_free_mb": None}
    except Exception as e:
        return {"component": "gpu", "status": "unavailable", "message": str(e)}


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    """Prometheus-совместимые метрики."""
    from api.store import RunStatus

    all_runs, _ = run_store.list(per_page=10_000)

    counts: Dict[str, int] = {}
    durations = []

    for r in all_runs:
        key = f'status="{r.status.value}",verdict="{r.verdict or ""}"'
        counts[key] = counts.get(key, 0) + 1
        if r.duration_sec is not None:
            durations.append(r.duration_sec)

    queued  = sum(1 for r in all_runs if r.status == RunStatus.queued)
    running = sum(1 for r in all_runs if r.status == RunStatus.running)

    lines = ["# HELP synth_runs_total Total pipeline runs by status and verdict"]
    lines.append("# TYPE synth_runs_total counter")
    for label, count in counts.items():
        lines.append(f"synth_runs_total{{{label}}} {count}")

    lines.append("# HELP synth_queue_size Current number of queued/running tasks")
    lines.append("# TYPE synth_queue_size gauge")
    lines.append(f"synth_queue_size{{state=\"queued\"}} {queued}")
    lines.append(f"synth_queue_size{{state=\"running\"}} {running}")

    if durations:
        avg = sum(durations) / len(durations)
        lines.append("# HELP synth_run_duration_seconds_avg Average pipeline duration")
        lines.append("# TYPE synth_run_duration_seconds_avg gauge")
        lines.append(f"synth_run_duration_seconds_avg {avg:.2f}")

    return "\n".join(lines) + "\n"

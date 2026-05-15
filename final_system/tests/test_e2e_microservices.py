# final_system/tests/test_e2e_microservices.py
#
# E2E smoke test: полный пайплайн через docker-compose стек.
#
# Запуск:
#   cd <repo-root>
#   docker compose -f final_system/docker-compose.yml up -d --wait   # если стек не запущен
#   python -m pytest final_system/tests/test_e2e_microservices.py -v -m e2e
#
# Тест НЕ поднимает и НЕ гасит стек самостоятельно — он требует уже
# работающий стек (все healthcheck'и зелёные). Это позволяет запускать
# его многократно без долгого docker-compose up/down в каждом прогоне.
# Для полностью автономного CI добавьте --up/--down флаги или вызывайте
# `docker compose up -d --wait` в before_script.

import subprocess
import time
from pathlib import Path

import pytest
import requests

GATEWAY = "http://localhost:8000"
COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.yml"
POLL_INTERVAL = 5       # секунд между опросами статуса
TIMEOUT = 600           # максимальное время ожидания завершения пайплайна (сек)


def _all_services_healthy() -> bool:
    """Проверяет, что все сервисы отвечают на /health."""
    # gateway использует /api/v1/health, остальные — /health
    endpoints = [
        (8000, "/api/v1/health"),
        (8001, "/health"),
        (8002, "/health"),
        (8003, "/health"),
        (8004, "/health"),
    ]
    for port, path in endpoints:
        try:
            r = requests.get(f"http://localhost:{port}{path}", timeout=5)
            if r.status_code != 200:
                return False
        except requests.exceptions.RequestException:
            return False
    return True


@pytest.fixture(scope="session")
def docker_stack():
    """Проверяет здоровье стека. Пытается поднять его если не запущен."""
    if not _all_services_healthy():
        print("\n[e2e] Docker stack not healthy — attempting docker compose up -d --wait ...")
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--wait"],
            check=True,
        )
        # Дополнительно ждём healthcheck'ов
        for _ in range(30):
            if _all_services_healthy():
                break
            time.sleep(5)
        else:
            pytest.fail("Docker stack did not become healthy within 150s")
    yield


@pytest.mark.e2e
def test_full_pipeline_ctgan(docker_stack):
    """
    Smoke-тест полного пайплайна:
      data upload → split → synthesis (CTGAN, 5 эпох) → evaluation → reporting.

    Проверяет:
    - run создаётся и переходит в статус completed (не failed)
    - verdict присутствует (PASS / FAIL / PARTIAL — любой, лишь бы не None)
    - в RunDetail есть поля synth_rows и config_snapshot
    """
    # ── 1. Запускаем пайплайн ────────────────────────────────────────────────
    resp = requests.post(
        f"{GATEWAY}/api/v1/runs",
        json={"config_name": "e2e_ctgan", "quick_test": False},
        timeout=30,
    )
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    run_id = resp.json()["run_id"]
    print(f"\n[e2e] run_id={run_id}")

    # ── 2. Поллим до завершения ──────────────────────────────────────────────
    deadline = time.time() + TIMEOUT
    final_status = None
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        detail = requests.get(f"{GATEWAY}/api/v1/runs/{run_id}", timeout=10).json()
        status = detail["status"]
        print(f"[e2e]   status={status}")
        if status in ("completed", "failed"):
            final_status = status
            break

    assert final_status is not None, f"Pipeline did not finish within {TIMEOUT}s"
    assert final_status == "completed", (
        f"Pipeline failed. error_message={detail.get('error_message')}"
    )

    # ── 3. Проверяем структуру ответа ────────────────────────────────────────
    assert detail["verdict"] in ("PASS", "FAIL", "PARTIAL"), (
        f"Unexpected verdict: {detail['verdict']}"
    )
    assert detail["synth_rows"] is not None and detail["synth_rows"] > 0, (
        "synth_rows should be positive"
    )
    assert isinstance(detail.get("config_snapshot"), dict), (
        "config_snapshot should be a dict"
    )

    print(f"[e2e] DONE  verdict={detail['verdict']}  synth_rows={detail['synth_rows']}")

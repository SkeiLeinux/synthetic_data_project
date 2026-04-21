# api/settings.py
#
# Настройки API-сервиса через переменные окружения.
# Загружается один раз при старте через get_settings().

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Конфигурация API-сервиса.

    Переменные окружения (все опциональны, есть дефолты для разработки):
        API_KEY          — Bearer-токен для авторизации; если не задан — auth отключена
        DATA_DIR         — директория для загруженных датасетов
        MODELS_DIR       — директория для сохранённых моделей
        CONFIGS_DIR      — директория конфигов
        REPORTS_DIR      — директория JSON-отчётов
        DEFAULT_CONFIG   — конфиг по умолчанию (используется если не указан config_name)
        LOG_PATH         — путь к лог-файлу
        DB_DISABLED      — true = не использовать ProcessRegistry
        REDIS_URL        — строка подключения к Redis (default: redis://localhost:6379/0)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    api_key: str | None = None  # None = авторизация отключена (режим разработки)

    # Пути (относительно final_system/)
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    models_dir: Path = Path("/data") / "models"
    configs_dir: Path = Path(__file__).parent.parent / "configs"
    reports_dir: Path = Path("/data") / "reports"   # shared volume — пишет reporting_service, читает Gateway
    log_path: str = "logs/app.log"
    data_root: Path = Path("/data")   # shared volume; переопределяется через DATA_ROOT

    default_config: str = "configs/adult.yaml"
    db_disabled: bool = False
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "synthetic_data_db"
    db_user: str = "postgres"
    db_password: str = ""        # обязательно задать в .env: DB_PASSWORD=...
    db_schema: str = "synthetic_data_schema"
    redis_url: str = "redis://localhost:6379/0"

    # URL микросервисов (задаются в docker-compose через env).
    data_service_url: str = ""
    synthesis_service_url: str = ""
    evaluation_service_url: str = ""
    reporting_service_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()

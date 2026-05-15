# services/reporting_service/settings.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    data_root: Path = Path("/data")

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()

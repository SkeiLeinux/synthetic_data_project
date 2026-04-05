# final_system/registry/process_registry.py
#
# Внутренняя БД сервиса (System Storage на архитектурной диаграмме).
# Ведёт реестр процессов генерации, хранит метаданные и логи.
#
# Принимает AppConfig из config_loader — никакого configparser.
# Не знает ничего про данные компании — это зона ответственности DataIO.

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

try:
    from logger_config import setup_logger
except ImportError:
    from ..logger_config import setup_logger

logger = setup_logger(__name__)

# UUID типов метаданных — должны совпадать с db/init_db.sql
_META_SOURCE_INFO = "11111111-1111-1111-1111-111111111111"
_META_GEN_CONFIG  = "22222222-2222-2222-2222-222222222222"
_META_PRIVACY     = "33333333-3333-3333-3333-333333333333"
_META_UTILITY     = "44444444-4444-4444-4444-444444444444"


class ProcessRegistry:
    """
    Реестр процессов синтеза данных.

    Отвечает за:
      - создание и обновление записей о процессах (таблица processes)
      - хранение структурированных метаданных (таблица process_metadata, JSONB)
      - сохранение логов выполнения (таблица process_logs)

    Принимает AppConfig из config_loader — секция database.

    Пример использования:
        cfg = load_config("configs/adult.yaml")
        registry = ProcessRegistry(app_config=cfg)
        if registry.test_connection():
            registry.start_process(pid, source_info="adult.csv")
            ...
            registry.finish_process(pid, status="SUCCESS")
    """

    def __init__(self, app_config) -> None:
        """
        Параметры:
            app_config — AppConfig из config_loader.
                         Использует секцию app_config.database.
        """
        if app_config.database is None:
            raise ValueError(
                "Секция database: не заполнена в конфиге. "
                "ProcessRegistry требует подключения к System Storage."
            )
        db = app_config.database
        self._schema = db.schema_name
        self._engine = create_engine(
            f"postgresql+psycopg2://{db.user}:{db.password}"
            f"@{db.host}:{db.port}/{db.dbname}"
        )

    # ─────────────────────────────────────────────
    # Управление процессами
    # ─────────────────────────────────────────────

    def start_process(
        self,
        process_id: str,
        source_info: str,
        config_path: str = "configs/adult.yaml",
    ) -> None:
        """
        Регистрирует новый процесс генерации со статусом RUNNING.
        Вызывать в самом начале run_pipeline() — до любых вычислений.
        """
        sql = text(f"""
            INSERT INTO {self._schema}.processes
                (process_id, start_time, status, source_data_info, config_rout)
            VALUES (:pid, :start_ts, 'RUNNING', :src_info, :cfg)
        """)
        with self._engine.begin() as conn:
            conn.execute(sql, {
                "pid": process_id,
                "start_ts": datetime.now(),
                "src_info": source_info,
                "cfg": config_path,
            })
        logger.info(f"[ProcessRegistry] Процесс зарегистрирован: {process_id}")

    def finish_process(
        self,
        process_id: str,
        status: str,
        synth_location: Optional[str] = None,
        report_location: Optional[str] = None,
    ) -> None:
        """
        Обновляет статус и время завершения процесса.

        Параметры:
            status          — итоговый статус: SUCCESS, FAIL, PARTIAL, ERROR
            synth_location  — путь/таблица где лежит синтетика
            report_location — путь к JSON-отчёту
        """
        sql = text(f"""
            UPDATE {self._schema}.processes
            SET end_time = :end_ts,
                status = :status,
                synthetic_data_location = :synth_loc,
                report_location = :rep_loc
            WHERE process_id = :pid
        """)
        with self._engine.begin() as conn:
            conn.execute(sql, {
                "end_ts": datetime.now(),
                "status": status,
                "synth_loc": synth_location,
                "rep_loc": report_location,
                "pid": process_id,
            })
        logger.info(f"[ProcessRegistry] Процесс завершён: {process_id} → {status}")

    # ─────────────────────────────────────────────
    # Метаданные
    # ─────────────────────────────────────────────

    def save_source_info(self, process_id: str, info: Dict[str, Any]) -> None:
        """Сохраняет описание исходного датасета."""
        self._insert_metadata(process_id, _META_SOURCE_INFO, info)

    def save_generator_config(self, process_id: str, config: Dict[str, Any]) -> None:
        """Сохраняет конфигурацию генератора и DP-параметры."""
        self._insert_metadata(process_id, _META_GEN_CONFIG, config)

    def save_privacy_report(self, process_id: str, report: Dict[str, Any]) -> None:
        """Сохраняет результаты оценки приватности."""
        self._insert_metadata(process_id, _META_PRIVACY, report)

    def save_utility_report(self, process_id: str, report: Dict[str, Any]) -> None:
        """Сохраняет результаты оценки полезности."""
        self._insert_metadata(process_id, _META_UTILITY, report)

    def save_metadata(
        self,
        process_id: str,
        metadata_type_id: str,
        value: Dict[str, Any],
    ) -> None:
        """Универсальный метод для произвольных метаданных."""
        self._insert_metadata(process_id, metadata_type_id, value)

    # ─────────────────────────────────────────────
    # Логи
    # ─────────────────────────────────────────────

    def save_log(self, process_id: str, log_content: str) -> None:
        """
        Сохраняет содержимое лог-файла процесса в БД.
        Автоматически связывает лог с процессом через log_id.
        """
        log_id = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {self._schema}.process_logs
                    (log_id, log_content, created_at)
                VALUES (:lid, :content, NOW())
            """), {"lid": log_id, "content": log_content})

            conn.execute(text(f"""
                UPDATE {self._schema}.processes
                SET log_id = :lid
                WHERE process_id = :pid
            """), {"lid": log_id, "pid": process_id})

        logger.info(f"[ProcessRegistry] Лог сохранён: {log_id} → процесс {process_id}")

    def save_log_from_file(self, process_id: str, log_path: str) -> None:
        """Читает лог-файл и сохраняет его содержимое в БД."""
        from pathlib import Path
        path = Path(log_path)
        if not path.exists():
            logger.warning(f"[ProcessRegistry] Лог-файл не найден: {log_path}")
            return
        self.save_log(process_id, path.read_text(encoding="utf-8"))

    # ─────────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Проверяет подключение к внутренней БД сервиса."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("[ProcessRegistry] Подключение к System Storage успешно")
            return True
        except Exception as e:
            logger.error(f"[ProcessRegistry] Ошибка подключения: {e}")
            return False

    def close(self) -> None:
        """Закрывает соединение с БД."""
        self._engine.dispose()
        logger.info("[ProcessRegistry] Соединение с System Storage закрыто")

    def _insert_metadata(
        self,
        process_id: str,
        metadata_type_id: str,
        value: Dict[str, Any],
    ) -> None:
        sql = text(f"""
            INSERT INTO {self._schema}.process_metadata
                (process_id, metadata_type_id, metadata_value, created_at)
            VALUES (:pid, :mtid, :mval, NOW())
        """)
        with self._engine.begin() as conn:
            conn.execute(sql, {
                "pid": process_id,
                "mtid": metadata_type_id,
                "mval": json.dumps(value),
            })
# final_system/data_service/data_io.py
#
# Работа с внешними данными компании (Data Storage на архитектурной диаграмме).
# Читает исходные данные из CSV или внешней БД.
# Записывает синтетические данные обратно в CSV или SQL-таблицу.
#
# Не знает ничего про внутренние процессы, логи и метаданные сервиса —
# это зона ответственности ProcessRegistry.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from ..logger_config import setup_logger
except ImportError:
    from logger_config import setup_logger

logger = setup_logger(__name__)


def _engine_from_db_config(db) -> "sqlalchemy.engine.Engine":
    """
    Создаёт SQLAlchemy engine из DBConfig (Pydantic-объект из config_loader).
    db должен иметь поля: user, password, host, port, dbname.
    """
    return create_engine(
        f"postgresql+psycopg2://{db.user}:{db.password}"
        f"@{db.host}:{db.port}/{db.dbname}"
    )


class DataIO:
    """
    Ввод/вывод данных компании.

    Отвечает за:
      - загрузку исходных (реальных) данных из CSV или внешней БД компании
      - сохранение сгенерированных синтетических данных в CSV или SQL-таблицу

    Конфигурируется через AppConfig из config_loader.
    Если внешняя БД не нужна — достаточно вызвать load_from_csv() / save_to_csv()
    без передачи app_config.

    Пример минимального использования (только CSV, без БД):
        io = DataIO()
        df = io.load_from_csv("data/adult.csv")
        io.save_to_csv(synth_df, "data/adult_synth.csv")

    Пример с конфигом (с поддержкой БД):
        cfg = load_config("configs/adult.yaml")
        io = DataIO(app_config=cfg)
        df = io.load_from_db("SELECT * FROM raw_data")
    """

    def __init__(self, app_config=None) -> None:
        """
        Параметры:
            app_config — AppConfig из config_loader; передавать если нужна работа с БД.
                         Для работы только с CSV можно не передавать.
        """
        self._app_config = app_config

        # Движки создаются лениво — только при первом обращении к методам БД,
        # чтобы не падать при старте если внешней БД нет.
        self._source_engine = None
        self._target_engine = None

    # ─────────────────────────────────────────────
    # Загрузка данных
    # ─────────────────────────────────────────────

    def load_from_csv(
        self,
        file_path: str,
        na_values: Optional[list] = None,
        skipinitialspace: bool = True,
        **read_csv_kwargs,
    ) -> pd.DataFrame:
        """
        Загружает данные из CSV-файла.

        Параметры:
            na_values        — значения, которые считать NaN (например, ["?"])
            skipinitialspace — убирать пробелы перед значениями (типично для Adult)
        """
        try:
            logger.info(f"[DataIO] Загрузка из CSV: {file_path}")
            df = pd.read_csv(
                file_path,
                na_values=na_values or [],
                skipinitialspace=skipinitialspace,
                **read_csv_kwargs,
            )
            logger.info(f"[DataIO] Загружено {len(df)} строк, {len(df.columns)} колонок")
            return df
        except Exception:
            logger.exception(f"[DataIO] Ошибка при чтении CSV: {file_path}")
            raise

    def load_from_db(self, query: str) -> pd.DataFrame:
        """
        Загружает данные из внешней БД компании через SQL-запрос.
        Требует app_config с заполненной секцией data_source.
        """
        engine = self._get_source_engine()
        try:
            logger.info("[DataIO] Загрузка из БД (data_source)")
            df = pd.read_sql(query, engine)
            logger.info(f"[DataIO] Загружено {len(df)} строк")
            return df
        except Exception:
            logger.exception("[DataIO] Ошибка при загрузке из БД")
            raise

    # ─────────────────────────────────────────────
    # Сохранение данных
    # ─────────────────────────────────────────────

    def save_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Сохраняет датафрейм в CSV-файл."""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            logger.info(f"[DataIO] Сохранено {len(df)} строк → {file_path}")
        except Exception:
            logger.exception(f"[DataIO] Ошибка при записи CSV: {file_path}")
            raise

    def save_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "replace",
    ) -> None:
        """
        Сохраняет датафрейм в SQL-таблицу внешней БД.
        Требует app_config с заполненной секцией data_target.

        Параметры:
            table_name — имя целевой таблицы
            schema     — схема в БД (None = схема по умолчанию)
            if_exists  — поведение при существующей таблице: replace / append / fail
        """
        engine = self._get_target_engine()
        try:
            df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
            logger.info(
                f"[DataIO] Сохранено {len(df)} строк → "
                f"{schema + '.' if schema else ''}{table_name}"
            )
        except Exception:
            logger.exception("[DataIO] Ошибка при записи в БД")
            raise

    # ─────────────────────────────────────────────
    # Вспомогательные методы
    # ─────────────────────────────────────────────

    def test_source_connection(self) -> bool:
        """Проверяет подключение к источнику данных."""
        try:
            engine = self._get_source_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("[DataIO] Подключение к источнику успешно")
            return True
        except Exception as e:
            logger.error(f"[DataIO] Ошибка подключения к источнику: {e}")
            return False

    def close(self) -> None:
        """Закрывает все открытые соединения с БД."""
        if self._source_engine:
            self._source_engine.dispose()
            logger.info("[DataIO] Соединение с источником закрыто")
        if self._target_engine:
            self._target_engine.dispose()
            logger.info("[DataIO] Соединение с целевой БД закрыто")

    def _get_source_engine(self):
        if self._source_engine is None:
            if self._app_config is None or self._app_config.data_source is None:
                raise ValueError(
                    "data_source не задан в конфиге. "
                    "Заполните секцию data_source: в configs/adult.yaml."
                )
            self._source_engine = _engine_from_db_config(self._app_config.data_source)
        return self._source_engine

    def _get_target_engine(self):
        if self._target_engine is None:
            if self._app_config is None or self._app_config.data_target is None:
                raise ValueError(
                    "data_target не задан в конфиге. "
                    "Заполните секцию data_target: в configs/adult.yaml."
                )
            self._target_engine = _engine_from_db_config(self._app_config.data_target)
        return self._target_engine
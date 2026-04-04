# final_system/data_io.py
#
# Работа с внешними данными компании (Data Storage на архитектурной диаграмме).
# Читает исходные данные из CSV или внешней БД.
# Записывает синтетические данные обратно в CSV или SQL-таблицу.
#
# Не знает ничего про внутренние процессы, логи и метаданные сервиса —
# это зона ответственности ProcessRegistry.

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from .logger_config import setup_logger
except ImportError:
    from logger_config import setup_logger

logger = setup_logger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.ini"


def _engine_from_config(cfg: configparser.ConfigParser, section: str):
    """
    Создаёт SQLAlchemy engine из указанной секции конфига.
    Секция должна содержать: host, port, dbname, user, password.
    """
    db = cfg[section]
    return create_engine(
        f"postgresql+psycopg2://{db['user']}:{db['password']}"
        f"@{db['host']}:{db['port']}/{db['dbname']}"
    )


class DataIO:
    """
    Ввод/вывод данных компании.

    Отвечает за:
      - загрузку исходных (реальных) данных из CSV или внешней БД компании
      - сохранение сгенерированных синтетических данных в CSV или SQL-таблицу

    Конфигурируется через config.ini, секция [DATA_SOURCE] для чтения
    и [DATA_TARGET] для записи. Если внешняя БД не нужна — достаточно
    передать пути к файлам напрямую через методы load/save.

    Пример минимального использования (только CSV, без БД):
        io = DataIO()
        df = io.load_from_csv("data/adult.csv")
        io.save_to_csv(synth_df, "data/adult_synth.csv")
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg_file = Path(config_path) if config_path else _CONFIG_PATH
        self._cfg = configparser.ConfigParser()
        self._cfg.read(cfg_file)

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
            na_values       — значения, которые считать NaN (например, ["?"])
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

    def load_from_db(self, query: str, section: str = "DATA_SOURCE") -> pd.DataFrame:
        """
        Загружает данные из внешней БД компании через SQL-запрос.

        Параметры:
            query   — SQL-запрос для извлечения данных
            section — секция config.ini с параметрами подключения к источнику
        """
        engine = self._get_source_engine(section)
        try:
            logger.info(f"[DataIO] Загрузка из БД (секция [{section}])")
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
        section: str = "DATA_TARGET",
    ) -> None:
        """
        Сохраняет датафрейм в SQL-таблицу внешней БД.

        Параметры:
            table_name — имя целевой таблицы
            schema     — схема в БД (None = схема по умолчанию)
            if_exists  — поведение при существующей таблице: replace / append / fail
            section    — секция config.ini с параметрами подключения к цели
        """
        engine = self._get_target_engine(section)
        try:
            df.to_sql(
                table_name, engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
            )
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

    def test_source_connection(self, section: str = "DATA_SOURCE") -> bool:
        """Проверяет подключение к источнику данных. Возвращает True если успешно."""
        try:
            engine = self._get_source_engine(section)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"[DataIO] Подключение к источнику [{section}] успешно")
            return True
        except Exception as e:
            logger.error(f"[DataIO] Ошибка подключения к источнику [{section}]: {e}")
            return False

    def close(self) -> None:
        """Закрывает все открытые соединения с БД."""
        if self._source_engine:
            self._source_engine.dispose()
            logger.info("[DataIO] Соединение с источником закрыто")
        if self._target_engine:
            self._target_engine.dispose()
            logger.info("[DataIO] Соединение с целевой БД закрыто")

    def _get_source_engine(self, section: str):
        """Лениво инициализирует engine для источника данных."""
        if self._source_engine is None:
            if section not in self._cfg:
                raise ValueError(
                    f"Секция [{section}] не найдена в config.ini. "
                    f"Добавьте параметры подключения к источнику данных."
                )
            self._source_engine = _engine_from_config(self._cfg, section)
        return self._source_engine

    def _get_target_engine(self, section: str):
        """Лениво инициализирует engine для целевой БД."""
        if self._target_engine is None:
            if section not in self._cfg:
                raise ValueError(
                    f"Секция [{section}] не найдена в config.ini. "
                    f"Добавьте параметры подключения к целевой БД."
                )
            self._target_engine = _engine_from_config(self._cfg, section)
        return self._target_engine
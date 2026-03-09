import configparser
from sqlalchemy import create_engine, text
import pandas as pd
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from logger_config import setup_logger

logger = setup_logger(__name__)


class DataManager:
    def __init__(self):
        # чтение конфигурации
        configuration = configparser.ConfigParser()
        configuration.read('config.ini')

        db = configuration['DATABASE']
        self.engine = create_engine(
            f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
        )

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                # connection.execute(config['QUERIES']['test_query'])
                connection.execute(text("SELECT 1;"))
            logger.info("Подключение к базе данных успешно!")
        except Exception as e:
            logger.info(f"Ошибка подключения: {e}")

    def load_data_from_db(self, query_raw_data):
        try:
            logger.info("Загрузка данных из базы")
            df = pd.read_sql(query_raw_data, self.engine)
            logger.info(f"Загружено {len(df)} строк")
            return df
        except Exception as e:
            logger.exception("Ошибка при загрузке данных")
            raise

    def load_data_from_csv(self, file_path: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Загрузить датасет из CSV-файла.
        file_path — путь к файлу;
        read_csv_kwargs — дополнительные аргументы pd.read_csv.
        """
        try:
            logger.info(f"Загрузка данных из CSV: {file_path}")
            df = pd.read_csv(file_path, **read_csv_kwargs)
            logger.info(f"Загружено {len(df)} строк из CSV")
            return df
        except Exception as e:
            logger.exception("Ошибка при загрузке данных из CSV")
            raise

    def save_data(self, df, table_name, schema='synthetic_data_schema', process_id=None):
        try:
            df.to_sql(table_name, self.engine, schema=schema, if_exists='replace', index=False)
            logger.info(f"Сохранено {len(df)} строк в таблицу {table_name}")
            if process_id:
                # Обновляем запись процесса: сохраняем путь к таблице
                synth_loc = f"{schema}.{table_name}"
                self.update_process_end(process_id, datetime.now(), 'SAVED', synthetic_data_location=synth_loc)
        except Exception as e:
            logger.exception("Ошибка при сохранении данных")

    def close(self):
        self.engine.dispose()
        logger.info("Соединение с базой данных закрыто")

    # методы для работы с бд системы

    def create_process(self, process_id: str, start_time: datetime, status: str,
                       source_data_info: str, config_rout: str):
        sql = text("""
            INSERT INTO synthetic_data_schema.processes
            (process_id, start_time, status, source_data_info, config_rout)
            VALUES (:pid, :start_ts, :status, :src_info, :cfg);
        """)
        print("создаем процесс")
        with self.engine.begin() as conn:
            conn.execute(sql, {
                'pid': process_id,
                'start_ts': start_time,
                'status': status,
                'src_info': source_data_info,
                'cfg': config_rout
            })
        print("параметры заполнили")

    def update_process_end(self, process_id: str, end_time: datetime, status: str,
                           synthetic_data_location: str = None, report_location: str = None):
        sql = text("""
            UPDATE synthetic_data_schema.processes
            SET end_time = :end_ts,
                status = :status,
                synthetic_data_location = :synth_loc,
                report_location = :rep_loc
            WHERE process_id = :pid;
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {
                'end_ts': end_time,
                'status': status,
                'synth_loc': synthetic_data_location,
                'rep_loc': report_location,
                'pid': process_id
            })

    def insert_log(self, process_id: str, log_id: str, log_file_content: bytes):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO synthetic_data_schema.process_logs
                (log_id, log_content, created_at)
                VALUES (:lid, :log_content, NOW())
            """), {
                'lid': log_id,
                'log_content': log_file_content.decode('utf-8'),
            })

            conn.execute(text("""
                UPDATE synthetic_data_schema.processes
                SET log_id = :lid
                WHERE process_id = :pid
            """), {
                'lid': log_id,
                'pid': process_id
            })


    def insert_metadata(self, process_id: str, metadata_type_id: str, metadata_value: dict):
        sql = text("""
            INSERT INTO synthetic_data_schema.process_metadata
            (process_id, metadata_type_id, metadata_value, created_at)
            VALUES (:pid, :mtid, :mval, NOW());
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {
                'pid': process_id,
                'mtid': metadata_type_id,
                'mval': json.dumps(metadata_value)
            })


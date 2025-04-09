import configparser
from sqlalchemy import create_engine, text
import pandas as pd
from sqlalchemy.testing.plugin.plugin_base import config

from logger_config import setup_logger

logger = setup_logger(__name__)


class DataManager:
    def __init__(self):
        # Чтение конфигурации
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

    def load_data(self, query_raw_data):
        try:
            logger.info("Загрузка данных из базы")
            df = pd.read_sql(query_raw_data, self.engine)
            logger.info(f"Загружено {len(df)} строк")
            return df
        except Exception as e:
            logger.exception("Ошибка при загрузке данных")
            raise

    def save_data(self, df, table_name, schema='synthetic_data_schema'):
        try:
            df.to_sql(table_name, self.engine, schema=schema, if_exists='replace', index=False)
            logger.info(f"Сохранено {len(df)} строк в таблицу {table_name}")
        except Exception as e:
            logger.exception("Ошибка при сохранении данных")

    def close(self):
        self.engine.dispose()
        logger.info("Соединение с базой данных закрыто")

import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text

class DataManager:
    def __init__(self):
        self.db_config = {
            'dbname': 'synthetic_data_db',
            'user': 'postgres',             # Имя пользователя
            'password': '111',              # Пароль от БД PostgreSQL
            'host': 'localhost',
            'port': 5432
        }
        # Подключение через SQLAlchemy для удобной работы с pandas
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )

    def load_data(self, query):
        # Загрузка данных из базы в DataFrame
        return pd.read_sql(query, self.engine)

    def save_data(self, dataframe, table_name, schema='synthetic_data_schema'):
        # Запись данных из DataFrame в таблицу БД
        dataframe.to_sql(table_name, self.engine, schema=schema, if_exists='replace', index=False)

    def test_connection(self):
        # Тест подключения
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text('SELECT 1;'))
                print("Результат проверки подключения:", result.scalar())
            print("✅ Подключение к базе данных успешно!")
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")

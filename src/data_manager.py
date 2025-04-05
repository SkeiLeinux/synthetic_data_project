import psycopg2
import pandas as pd

class DataManager:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)

    def load_data(self, query):
        return pd.read_sql(query, self.conn)

    def save_data(self, dataframe, table_name):
        dataframe.to_sql(table_name, self.conn, if_exists='replace', index=False)

    def close(self):
        self.conn.close()


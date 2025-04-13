# preprocessing/data_extractor.py

import pandas as pd

def load_data(conn, query):
    # Извлечь данные из БД по заданному SQL-запросу и вернуть DataFrame
    df = pd.read_sql_query(query, conn)
    return df

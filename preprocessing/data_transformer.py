# preprocessing/data_transformer.py

def clean_and_normalize(data):
    # Удалить дубликаты
    data = data.drop_duplicates()
    # Заполнить пропуски для числовых столбцов средним значением
    for col in data.select_dtypes(include=['float', 'int']).columns:
        data[col] = data[col].fillna(data[col].mean())
    # Выполнить нормализацию числовых столбцов по методу min-max scaling
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
    return data

from data_manager import DataManager
from generator import generate_synthetic_data


def main():
    # Конфигурация БД
    db_config = {
        'dbname': 'synthetic_db',
        'user': 'postgres',
        'password': 'your_password',
        'host': 'localhost',
        'port': 5432
    }

    dm = DataManager(db_config)

    # Загрузка данных из БД
    query = "SELECT * FROM raw_data;"
    original_df = dm.load_data(query)

    print("Original data:")
    print(original_df.head())

    # Генерация синтетических данных
    synthetic_df = generate_synthetic_data(original_df)

    print("Synthetic data:")
    print(synthetic_df.head())

    dm.close()


if __name__ == "__main__":
    main()

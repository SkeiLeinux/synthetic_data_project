from data_manager import DataManager
from generator import generate_synthetic_data
from processor import DataProcessor


def main():
    db_config = {
        'dbname': 'synthetic_data_db',
        'user': 'postgres',
        'password': '111',  # Замените на свой пароль
        'host': 'localhost',
        'port': 5432
    }

    dm = DataManager(db_config)

    dm.test_connection()

    # Загрузка данных из БД
    query = "SELECT * FROM synthetic_data_schema.raw_data;"
    original_df = dm.load_data(query)

    print("🗃️ Исходные данные:")
    print(original_df.head())

    # Предобработка данных
    processor = DataProcessor(original_df)
    processed_df = processor.preprocess()

    print("\n🛠️ Данные после предобработки:")
    print(processed_df.head())

    # Основная статистика
    stats = processor.basic_statistics()
    print("\n📊 Статистика исходных данных:")
    print(stats)

    # Генерация синтетических данных (заглушка)
    synthetic_df = generate_synthetic_data(processed_df)

    print("\n🎲 Сгенерированные синтетические данные:")
    print(synthetic_df.head())

    dm.close()


if __name__ == "__main__":
    main()

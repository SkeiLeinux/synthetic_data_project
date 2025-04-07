from data_manager import DataManager
from generator import generate_synthetic_data
from processor import DataProcessor
from logger_config import setup_logger


logger = setup_logger(__name__)
logger.info("Старт программы")


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
    logger.info(f"Загружено {len(original_df)} строк из базы")

    print("🗃️ Исходные данные:")
    print(original_df.head())

    # Предобработка данных
    processor = DataProcessor(original_df)
    processed_df = processor.preprocess()
    logger.info(f"Выполнена предобработка данных")
    print("\n🛠️ Данные после предобработки:")
    print(processed_df.head())

    # Основная статистика
    stats = processor.basic_statistics()
    logger.info(f"Статистика исходных данных: \n {stats}")
    print("\n📊 Статистика исходных данных:")
    print(stats)

    # Укажи допустимые отклонения
    tolerances = {
        'column_int': {'percent': 10},  # целое число
        'column_float': {'percent': 5},  # число с плавающей точкой
        'column_date': {'days': 3},  # дата
        'column_timestamp': {'minutes': 60},  # дата-время (таймстамп)
    }
    logger.info(f"Допустимые отклонения: \n {tolerances}")
    synthetic_df = generate_synthetic_data(processed_df)

    stats_ok, violations = processor.compare_statistics(synthetic_df, tolerances)

    if stats_ok:
        print("✅ Статистики в допустимых пределах.")
        logger.info(f"✅ Статистики в допустимых пределах.")
    else:
        print("⚠️ Нарушения статистик:")
        logger.info(f":⚠️ Нарушения статистик:")
        for key, msg in violations.items():
            logger.info(f":\n {key}: {msg}")
            print(f"{key}: {msg}")

    print(synthetic_df)

    dm.close()

    logger.info(f":Работа завершена")


if __name__ == "__main__":
    main()

import configparser
from data_manager import DataManager
from generator import generate_synthetic_data
from processor import DataProcessor
from validator import DataValidator
import uuid
from datetime import datetime
from logger_config import setup_logger

logger = setup_logger(__name__)

def main():
    # Чтение конфигурации
    config = configparser.ConfigParser()
    config.read('config.ini')

    # db_config = {
    #     'dbname': config['DATABASE']['dbname'],
    #     'user': config['DATABASE']['user'],
    #     'password': config['DATABASE']['password'],
    #     'host': config['DATABASE']['host'],
    #     'port': config.getint('DATABASE', 'port')
    # }
    schema = config['DATABASE']['schema']
    logger.info("database data loaded")

    # Параметры анонимности
    k = config.getint('ANONYMITY', 'k')
    l = config.getint('ANONYMITY', 'l')
    t = config.getfloat('ANONYMITY', 't')

    # Параметры отклонений
    tolerances = {
        # config.get('')
        'column_2': {'absolute': config.getint('TOLERANCES', 'column_int_absolute')},
        'column_3': {'percent': config.getint('TOLERANCES', 'column_float_percent')},
        'created_at': {
            'days': config.getint('TOLERANCES', 'column_date_days'),
            'minutes': config.getint('TOLERANCES', 'column_timestamp_minutes')
        }
    }

    # SQL-запрос из конфига
    raw_data_query = config['QUERIES']['raw_data_query']

    # генерируем идент процесса
    process_id = str(uuid.uuid4())
    start_ts = datetime.now()

    logger.info("Подключение и загрузка данных из БД")
    dm = DataManager()
    dm.test_connection()

    source_info = raw_data_query  # источник данных
    config_rout = config.get('PATHS', 'config_rout', fallback='config.ini')  # путь конфига
    dm.create_process(process_id, start_ts, 'RUNNING', source_info, config_rout)

    # получаем оригинальный датасет
    original_df = dm.load_data(raw_data_query)
    logger.info(f"Загружено {len(original_df)} записей")

    processor = DataProcessor(original_df)
    processed_df = processor.preprocess()

    quasi_identifiers = [col.strip() for col in config['ANONYMITY']['quasi_identifiers'].split(',')]
    sensitive_attribute = [col.strip() for col in config['ANONYMITY']['sensitive_attribute'].split(',')]

    max_iterations = 10
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Итерация генерации №{iteration}")

        synthetic_df = generate_synthetic_data(processed_df)

        validator = DataValidator(synthetic_df)
        k_ok, _ = validator.check_k_anonymity(quasi_identifiers, k)
        l_ok, _ = validator.check_l_diversity(quasi_identifiers, sensitive_attribute, l)
        t_ok, t_val = validator.check_t_closeness(quasi_identifiers, sensitive_attribute, processed_df, t)
        stats_ok, violations = processor.compare_statistics(synthetic_df, tolerances)

        if k_ok and l_ok and t_ok and stats_ok:
            logger.info("Все проверки успешно пройдены, сохраняем датасет")

            meta = {
                "k_anonymity": bool(k_ok),
                "l_diversity": bool(l_ok),
                "t_closeness": {"ok": bool(t_ok), "value": t_val},
                "stats_ok": bool(stats_ok),
                "violations": violations
            }
            dm.insert_metadata(process_id, '33333333-3333-3333-3333-333333333333', meta)

            dm.save_data(synthetic_df, 'synthetic_data', schema=schema, process_id=process_id)

            final_status = 'SUCCESS'
            break
        else:
            logger.warning(f"Итерация {iteration}: обнаружены нарушения")
            logger.warning(f"k-анонимность: {k_ok}, l-разнообразие: {l_ok}, t-близость: {t_ok} (значение: {t_val:.3f}), статистики: {stats_ok}")
            if not stats_ok:
                for key, msg in violations.items():
                    logger.warning(f"Столбец {key}: {msg}")
    else:
        logger.error("Не удалось создать корректный датасет за допустимое число итераций")
        final_status = 'FAILED'

    end_ts = datetime.now()

    dm.update_process_end(process_id, end_ts, final_status)

    dm.close()
    logger.info("Завершение работы программы")

if __name__ == "__main__":
    main()

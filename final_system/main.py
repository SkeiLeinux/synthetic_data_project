import configparser
from importlib.util import source_hash

from data_manager import DataManager
from generator import generate_synthetic_data
from processor import DataProcessor
from validator import DataValidator
import uuid
from datetime import datetime
from logger_config import setup_logger

logger = setup_logger(__name__)

def main():
    # чтение конфигурации
    config = configparser.ConfigParser()
    config.read('config.ini')

    schema = config['DATABASE']['schema']
    logger.info("database data loaded")

    # Параметры анонимности
    k = config.getint('ANONYMITY', 'k')
    l = config.getint('ANONYMITY', 'l')
    t = config.getfloat('ANONYMITY', 't')

    # источник датасета из конфига
    raw_data_query = config['QUERIES']['raw_data_query']
    raw_data_file = config['PATHS'].get('raw_data_file', '').strip()

    # генерируем идент процесса
    process_id = str(uuid.uuid4())
    start_ts = datetime.now()

    logger.info("Подключение и загрузка данных из БД")
    dm = DataManager()

    # получаем оригинальный датасет
    if raw_data_file:
        original_df = dm.load_data_from_csv(raw_data_file)
        source_info = raw_data_file
    else:
        dm.test_connection()
        source_info = raw_data_query  # источник данных
        original_df = dm.load_data_from_db(raw_data_query)
    logger.info(f"Загружено {len(original_df)} записей")

    config_rout = config.get('PATHS', 'config_rout', fallback='config.ini')  # путь конфига
    dm.create_process(process_id, start_ts, 'RUNNING', source_info, config_rout)

    processor = DataProcessor(original_df)
    processed_df = processor.preprocess()

    quasi_identifiers = [col.strip() for col in config['ANONYMITY']['quasi_identifiers'].split(',')]
    sensitive_attribute = config['ANONYMITY']['sensitive_attribute']

    generator_cfg = config['GENERATOR']
    model_name = generator_cfg.get('model_name', 'copulagan')
    epochs = generator_cfg.getint('epochs', fallback=300)
    batch_size = generator_cfg.getint('batch_size', fallback=500)
    cuda = generator_cfg.getboolean('cuda', fallback=False)

    max_iterations = 10
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Итерация генерации №{iteration}")

        synthetic_df = generate_synthetic_data(
            processed_df,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda
        )

        validator = DataValidator(synthetic_df)
        k_ok, _ = validator.check_k_anonymity(quasi_identifiers, k)
        l_ok, _ = validator.check_l_diversity(quasi_identifiers, sensitive_attribute, l)
        t_ok, t_val = validator.check_t_closeness(quasi_identifiers, sensitive_attribute, processed_df, t)

        if k_ok and l_ok and t_ok:
            logger.info("Все проверки успешно пройдены, сохраняем датасет")

            meta = {
                "k_anonymity": bool(k_ok),
                "l_diversity": bool(l_ok),
                "t_closeness": {"ok": bool(t_ok), "value": t_val}
            }
            dm.insert_metadata(process_id, '33333333-3333-3333-3333-333333333333', meta)

            dm.save_data(synthetic_df, 'synthetic_data', schema=schema, process_id=process_id)

            final_status = 'SUCCESS'
            break
        else:
            logger.warning(f"Итерация {iteration}: обнаружены нарушения")
            logger.warning(f"k-анонимность: {k_ok}, l-разнообразие: {l_ok}, t-близость: {t_ok} (значение: {t_val:.3f})")

    else:
        logger.error("Не удалось создать корректный датасет за допустимое число итераций")
        final_status = 'FAILED'

    end_ts = datetime.now()

    dm.update_process_end(process_id, end_ts, final_status)

    dm.close()
    logger.info("Завершение работы программы")

if __name__ == "__main__":
    main()

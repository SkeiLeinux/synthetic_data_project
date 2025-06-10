import configparser
from email.contentmanager import raw_data_manager
from importlib.util import source_hash
import os
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
    # raw_data_query = config['QUERIES']['raw_data_query']
    raw_data_query = None
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

    meta_data_description = {
        "num_rows": len(original_df),
        "columns": list(original_df.columns),
        "source": source_info
    }
    dm.insert_metadata(process_id, '11111111-1111-1111-1111-111111111111', meta_data_description)

    # выполняем предпроцессную обработку
    processor = DataProcessor(original_df)
    processed_df = processor.preprocess()

    quasi_identifiers = [col.strip() for col in config['ANONYMITY']['quasi_identifiers'].split(',')]
    sensitive_attribute = config['ANONYMITY']['sensitive_attribute']

    generator_cfg = config['GENERATOR']
    model_name = generator_cfg.get('model_name', 'copulagan')
    epochs = generator_cfg.getint('epochs', fallback=300)
    batch_size = generator_cfg.getint('batch_size', fallback=500)
    cuda = generator_cfg.getboolean('cuda', fallback=False)

    meta_generation_config = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "cuda": cuda
    }
    dm.insert_metadata(process_id, '22222222-2222-2222-2222-222222222222', meta_generation_config)

    max_iterations = 10
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Итерация генерации №{iteration}")

        synthetic_df = generate_synthetic_data(
            processed_df,
            sensitive_attribute,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda
        )

        validator = DataValidator(synthetic_df)
        # k_val = validator.check_k_anonymity(quasi_identifiers, k)
        # l_val = validator.check_l_diversity(quasi_identifiers, sensitive_attribute, l)
        # t_val = validator.check_t_closeness(quasi_identifiers, sensitive_attribute, processed_df, t)
        k_l_t_ok = validator.check_k_l_t(quasi_identifiers, sensitive_attribute, original_df, k, l, t)


        if k_l_t_ok:
            logger.info("Все проверки успешно пройдены, сохраняем датасет")

            meta = {
                "k_anonymity": validator.check_k_anonymity(quasi_identifiers),
                "l_diversity": validator.check_l_diversity(quasi_identifiers, sensitive_attribute),
                "t_closeness": validator.check_t_closeness(quasi_identifiers,sensitive_attribute,original_df)
            }
            dm.insert_metadata(process_id, '33333333-3333-3333-3333-333333333333', meta)

            dm.save_data(synthetic_df, 'synthetic_data', schema=schema, process_id=process_id)
            synthetic_df.to_csv('synth.csv')

            validator.generate_quality_report(
                real_df=processed_df,
                synthetic_df=synthetic_df,
                process_id=process_id,
                data_manager=dm
            )

            final_status = 'SUCCESS'
            break
        else:
            logger.warning(f"Итерация {iteration}: обнаружены нарушения")
            logger.warning(f"k-анонимность: {validator.check_k_anonymity(quasi_identifiers)}, "
                           f"l-разнообразие: {validator.check_l_diversity(quasi_identifiers, sensitive_attribute)}, "
                           f"t-близость: {validator.check_t_closeness(quasi_identifiers,sensitive_attribute,original_df)}")

    else:
        logger.error("Не удалось создать корректный датасет за допустимое число итераций")
        final_status = 'FAILED'

    end_ts = datetime.now()

    dm.update_process_end(process_id, end_ts, final_status)

    log_path = config['PATHS']['logs_rout']
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        log_id = str(uuid.uuid4())
        dm.insert_log(
            process_id=process_id,
            log_id=log_id,
            log_file_content=log_content.encode('utf-8'),  # если понадобится для хранения
        )

    dm.close()

    logger.info("Завершение работы программы")

if __name__ == "__main__":
    main()

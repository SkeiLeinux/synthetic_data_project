CREATE SCHEMA IF NOT EXISTS synthetic_data_schema;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- Установка схемы по умолчанию
SET search_path TO synthetic_data_schema;

-- Таблица типов метаданных
CREATE TABLE IF NOT EXISTS metadata_types (
    metadata_type_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT
);

-- Таблица логов процессов
CREATE TABLE IF NOT EXISTS process_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Таблица процессов
CREATE TABLE IF NOT EXISTS processes (
    process_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_time TIMESTAMP NOT NULL DEFAULT NOW(),
    end_time TIMESTAMP,
    status VARCHAR(32) NOT NULL,
    source_data_info TEXT NOT NULL,
    synthetic_data_location TEXT,
    report_location TEXT,
    config_rout VARCHAR(512) NOT NULL,
    log_id UUID,
    FOREIGN KEY (log_id) REFERENCES process_logs(log_id) ON DELETE SET NULL
);

-- Таблица метаданных процессов
CREATE TABLE IF NOT EXISTS process_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    process_id UUID NOT NULL,
    metadata_type_id UUID NOT NULL,
    metadata_value JSONB NOT NULL,
    FOREIGN KEY (process_id) REFERENCES processes(process_id) ON DELETE CASCADE,
    FOREIGN KEY (metadata_type_id) REFERENCES metadata_types(metadata_type_id) ON DELETE CASCADE
);

-- Наполнение типов метадаты
INSERT INTO synthetic_data_schema.metadata_types (metadata_type_id, type_name, description)
VALUES
  ('11111111-1111-1111-1111-111111111111', 'Описание исходных данных', 'Характеристики и свойства исходного набора'),
  ('22222222-2222-2222-2222-222222222222', 'Параметры генерации', 'Параметры, по которым генерируется синтетика'),
  ('33333333-3333-3333-3333-333333333333', 'Результаты валидации приватности', 'Результаты k-anonymity, l-diversity и др.'),
  ('44444444-4444-4444-4444-444444444444', 'Результаты валидации полезности', 'Результаты проверки сохранения статистик и т.д.');

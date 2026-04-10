# Synthetic Data Generation Service

**Сервис генерации синтетических табличных данных с оценкой приватности и полезности**

Дипломная работа — НИУ ВШЭ, ФКН, Программная инженерия, 2024–2026.

---

## Описание

Система реализует полный пайплайн создания синтетических табличных данных с гарантиями дифференциальной приватности (DP-CTGAN на базе SmartNoise Synth). После генерации автоматически вычисляются метрики приватности и полезности, формируется отчёт с агрегированным вердиктом PASS / FAIL / PARTIAL.

**Ключевые возможности:**
- Генерация данных с DP-гарантиями (DP-SGD, отслеживание расхода ε/δ-бюджета)
- Оценка приватности: k/l/t-анонимность, дистанционные метрики DCR/NNDR, симуляция атаки MIA
- Оценка полезности: статистическое сходство (JSD, TVD), ML-эффективность (TSTR/TRTR)
- REST API (FastAPI) с документацией Swagger на `/docs`
- CLI для запуска из командной строки без сервера
- Сохранение и повторное использование обученных моделей

---

## Установка

```bash
pip install -r requirements.txt
```

Зависимости: `smartnoise-synth`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `torch`, `scikit-learn`, `pandas`, `sqlalchemy`, `psycopg2-binary`.

PostgreSQL необходим только для ProcessRegistry; без него можно запускать с флагом `--no-db`.

---

## Быстрый старт

### Запуск REST API

```bash
cd final_system
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

После запуска:
- **Swagger UI** — http://localhost:8000/docs
- **ReDoc** — http://localhost:8000/redoc
- **OpenAPI JSON** — http://localhost:8000/openapi.json

Полная спецификация всех эндпоинтов также доступна в файле `openapi.yaml` в корне репозитория.

#### Аутентификация

По умолчанию авторизация отключена (режим разработки). Чтобы включить:

```bash
export API_KEY=your-secret-token
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

После этого добавляйте заголовок `Authorization: Bearer your-secret-token` ко всем запросам.

#### Переменные окружения API

| Переменная | Назначение | Дефолт |
|---|---|---|
| `API_KEY` | Bearer-токен; если не задан — auth отключена | не задан |
| `DATA_DIR` | Директория датасетов | `final_system/data/` |
| `MODELS_DIR` | Директория моделей | `final_system/models/` |
| `CONFIGS_DIR` | Директория конфигов | `final_system/configs/` |
| `REPORTS_DIR` | Директория JSON-отчётов | `final_system/reporter/reports/` |
| `LOG_PATH` | Путь к лог-файлу | `logs/app.log` |
| `DB_DISABLED` | `true` — не использовать PostgreSQL | `false` |

Поддерживается файл `.env` в директории `final_system/`.

#### Пример: запуск пайплайна через API

```bash
# 1. Загрузить датасет
curl -X POST http://localhost:8000/api/v1/datasets \
  -F "name=adult" -F "file=@data/adult.csv"

# 2. Создать запуск
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset_name":"adult","config_name":"adult","save_model":true}'

# 3. Проверить статус (run_id из ответа шага 2)
curl http://localhost:8000/api/v1/runs/{run_id}

# 4. Скачать синтетику
curl http://localhost:8000/api/v1/runs/{run_id}/synthetic -o synth.csv
```

---

### Запуск через CLI

```bash
cd final_system

# Запуск с дефолтным конфигом (configs/adult.yaml)
python cli.py

# Указать другой конфиг
python cli.py --config configs/bank.yaml

# Быстрый тест (50 эпох, 5 000 строк, ~2–3 мин)
python cli.py --quick-test

# Валидация конфига без запуска
python cli.py --check

# Запуск без PostgreSQL
python cli.py --no-db

# Сохранить обученную модель
python cli.py --save-model models/adult_model.pkl

# Сгенерировать синтетику из сохранённой модели (без повторного обучения)
python cli.py --from-model models/adult_model.pkl --rows 10000
```

### Интеграционный тест

```bash
cd final_system
python run_adult.py   # QUICK_TEST = True внутри файла (~2–3 мин)
```

---

## Архитектура пайплайна

```
Реальные данные
      │
      ▼
[DataProcessor]           ← очистка, автодетекция типов колонок
      │
      ▼
 train / holdout split    ← ДО обучения генератора (методологически важно)
      │
   ┌──┴───────────────────────┐
   ▼                          ▼
[DPCTGANGenerator]        holdout (генератор не видит)
   │                          │
   ▼                          ▼
[Синтетические данные]   контрольная группа
   │
   ├──▶ [PrivacyEvaluator]   ← DCR, NNDR, MIA, k/l/t-анонимность
   └──▶ [UtilityEvaluator]   ← JSD, TVD, TSTR/TRTR
              │
              ▼
         [Reporter]          ← вердикт PASS / FAIL / PARTIAL + JSON-отчёт
              │
              ▼
    [ProcessRegistry]        ← PostgreSQL (опционально)
```

`real_holdout` не передаётся в генератор — только в оценщики. Это обеспечивает корректное измерение меморизации (DCR, MIA).

---

## Структура репозитория

```
.
├── openapi.yaml                 # OpenAPI 3.1.0 спецификация всех эндпоинтов
├── requirements.txt
├── db/
│   └── init_db.sql              # Инициализация схемы PostgreSQL
└── final_system/
    ├── api/                     # FastAPI REST API
    │   ├── main.py              # Точка входа (uvicorn api.main:app)
    │   ├── settings.py          # Конфигурация через env / .env
    │   ├── store.py             # In-memory хранилище запусков и оценок
    │   ├── dependencies.py      # Bearer-авторизация
    │   ├── routers/             # Эндпоинты по ресурсам
    │   │   ├── runs.py          # /runs — запуск и мониторинг пайплайна
    │   │   ├── models.py        # /models — управление моделями
    │   │   ├── datasets.py      # /datasets — загрузка датасетов
    │   │   ├── configs.py       # /configs — управление YAML-конфигами
    │   │   ├── evaluations.py   # /evaluations — изолированная оценка
    │   │   └── system.py        # /health, /metrics
    │   └── schemas/             # Pydantic-схемы запросов и ответов
    ├── synthesizer/
    │   └── dp_ctgan.py          # DP-CTGAN генератор (SmartNoise)
    ├── evaluator/
    │   ├── privacy/             # k/l/t, DCR, NNDR, MIA
    │   └── utility/             # JSD, TVD, TSTR, TRTR
    ├── reporter/
    │   └── reporter.py          # Вердикт PASS/FAIL/PARTIAL + JSON-отчёт
    ├── data_service/
    │   ├── data_io.py           # CSV / PostgreSQL I/O
    │   └── processor.py         # Предобработка, детекция типов
    ├── registry/
    │   └── process_registry.py  # PostgreSQL-трекинг запусков (опционально)
    ├── configs/
    │   └── adult.yaml           # Основной конфиг (пример)
    ├── pipeline.py              # run_pipeline() — главный оркестратор
    ├── config_loader.py         # YAML → Pydantic-объекты
    ├── cli.py                   # CLI-интерфейс
    └── run_adult.py             # Интеграционный тест (Adult Census)
```

---

## Конфигурация (YAML)

Основной конфиг — `final_system/configs/adult.yaml`. Валидируется через Pydantic в `config_loader.py`.

| Секция | Описание |
|---|---|
| `pipeline` | sample_size, holdout_size, random_state, n_synth_rows |
| `generator` | epsilon, delta, sigma, epochs, batch_size, cuda |
| `data_schema` | auto-детекция или явное задание categorical/continuous колонок |
| `privacy` | quasi_identifiers, sensitive_attribute |
| `utility` | target_column, task_type (classification/regression) |
| `thresholds` | PASS/FAIL пороги: max_utility_loss, max_mean_jsd, max_mia_auc и др. |

`config.ini` — легаси-формат, не используется.

---

## Эндпоинты API

| Метод | Путь | Описание |
|---|---|---|
| POST | `/api/v1/runs` | Запустить пайплайн |
| GET | `/api/v1/runs/{run_id}` | Статус запуска |
| GET | `/api/v1/runs/{run_id}/report` | JSON-отчёт |
| GET | `/api/v1/runs/{run_id}/synthetic` | Скачать синтетику (CSV/JSON) |
| GET | `/api/v1/datasets` | Список датасетов |
| POST | `/api/v1/datasets` | Загрузить датасет |
| GET | `/api/v1/models` | Список моделей |
| POST | `/api/v1/models/{model_id}/samples` | Сгенерировать из модели |
| POST | `/api/v1/evaluations/privacy` | Изолированная оценка приватности |
| POST | `/api/v1/evaluations/utility` | Изолированная оценка полезности |
| GET | `/api/v1/health` | Healthcheck |

Полный список (33 эндпоинта) — в `openapi.yaml` или на `/docs`.

---

## База данных (опционально)

```bash
psql -U postgres -f db/init_db.sql
```

Схема `synthetic_data_schema`, таблицы: `processes`, `process_logs`, `process_metadata`, `metadata_types`. Отключить через `--no-db` (CLI) или `DB_DISABLED=true` (API).

---

## Воспроизводимость

```python
# CLI
python cli.py --save-model models/my_model.pkl
python cli.py --from-model models/my_model.pkl --rows 50000

# Программно
from synthesizer.dp_ctgan import DPCTGANGenerator
generator = DPCTGANGenerator.load("models/my_model.pkl")
df = generator.sample(10000)
```

Фиксируйте `random_state` в конфиге и сохраняйте модель для воспроизводимости эксперимента.


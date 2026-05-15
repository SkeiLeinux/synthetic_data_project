# Synthetic Data Generation Service

**Сервис генерации синтетических табличных данных с оценкой приватности и полезности**

Магистерская выпускная работа.

---

## Описание

Система реализует полный пайплайн создания синтетических табличных данных с гарантиями дифференциальной приватности (DP). После генерации автоматически вычисляются метрики приватности и полезности, формируется отчёт с агрегированным вердиктом PASS / FAIL / PARTIAL.

**Ключевые возможности:**
- Пять генераторов: **DP-CTGAN** (SmartNoise, основной), **DP-TVAE** (Opacus), **CTGAN / TVAE / CopulaGAN** (SDV, baseline без DP)
- DP-SGD-обучение с отслеживанием расхода ε/δ-бюджета через Privacy Accountant (RDP)
- Оценка приватности: k/l/t-анонимность, дистанционные метрики DCR/NNDR, distance-based proxy MIA
- Оценка полезности: статистическое сходство (JSD, TVD), корреляции (Pearson, Cramér's V), ML-эффективность (TSTR/TRTR F1)
- REST API (FastAPI) с документацией Swagger на `/docs`
- Микросервисная архитектура: изолированные образы под тяжёлые ML-зависимости
- Сохранение и повторное использование обученных моделей без расходования ε-бюджета
- Импорт/экспорт данных: CSV и PostgreSQL

---

## Архитектура

Система построена как набор микросервисов, общающихся по HTTP. Файловые артефакты (датасеты, модели, отчёты) передаются через **shared Docker volume** — сервисы обмениваются путями, а не содержимым.

| Сервис | Порт | Образ | Ответственность |
|---|---|---|---|
| **Gateway** (`app`) | 8000 | lightweight | HTTP-точка входа, оркестрация пайплайна, Redis run state |
| **Data Service** | 8001 | lightweight | Загрузка CSV / PostgreSQL, предобработка, стратифицированный holdout-сплит |
| **Synthesis Service** | 8002 | ~13 GB (torch + opacus + smartnoise + sdv) | Обучение генераторов, семплирование, хранение моделей |
| **Evaluation Service** | 8003 | ~960 MB (sklearn/scipy) | Метрики приватности и полезности |
| **Reporting Service** | 8004 | minimal (fastapi + pydantic) | Итоговый вердикт PASS/FAIL + JSON-отчёт |
| **Redis** | 6379 | redis:7-alpine | State Store (run state, статусы) |
| **PostgreSQL** (system) | 5433 | postgres:16-alpine | ProcessRegistry (история запусков) |
| **PostgreSQL** (user_db) | 5434 | postgres:16-alpine | Тестовая БД пользователя для импорта/экспорта данных |

### Поток данных (POST /api/v1/runs)

```
Gateway (runs.py::_execute_pipeline_microservices)
  │
  ├─ Step 1  POST data_service/datasets              (upload CSV)
  ├─ Step 2  POST data_service/datasets/{id}/split   (preprocess + holdout split)
  ├─ Step 3  POST synthesis_service/jobs             (async training job)
  ├─ Step 4  GET  synthesis_service/jobs/{id}        (poll every 10s)
  ├─ Step 5  POST evaluation_service/evaluate/privacy
  ├─ Step 6  POST evaluation_service/evaluate/utility
  └─ Step 7  POST reporting_service/reports          (verdict + save JSON)
```

### Shared Volume (`/data` внутри контейнеров)

```
/data/
├── datasets/{dataset_id}/raw.csv
├── splits/{split_id}/train.csv
├── splits/{split_id}/holdout.csv
├── splits/{split_id}/meta.json
├── synth/{job_id}/synthetic.csv
├── models/{model_id}.pkl
├── models/{model_id}.meta.json      # sidecar: run_id, dataset_name, dp_config, dp_spent
└── reports/{dataset}__{generator}__{ts}.json
```

`holdout.csv` фиксируется однократно на этапе сплита и **не передаётся в генератор** — только в оценщики. Это обеспечивает корректное измерение меморизации (DCR, MIA).

---

## Установка и запуск

### Требования

- Docker Desktop
- (опционально) NVIDIA Container Toolkit — для обучения на GPU
- 15+ GB свободного места для сборки `synthesis_service`

### Запуск

```bash
cd final_system
cp .env.example .env    # отредактируйте пароли под себя
docker compose up -d --build
```

При первом запуске сборка `synthesis_service` занимает 10–20 минут (pytorch + SmartNoise + SDV). Последующие пересборки быстрые — меняется только COPY-слой.

После запуска:
- Swagger UI — http://localhost:8000/docs
- Healthcheck Gateway — http://localhost:8000/api/v1/health

### Проверка статуса контейнеров

```bash
docker compose ps               # все должны быть healthy
docker compose logs -f app      # логи оркестратора
docker compose logs -f synthesis_service   # прогресс обучения
```

### Пересборка отдельного сервиса

```bash
# После правок в api/, shared/, config_loader.py
docker compose build app && docker compose up -d app

# После правок в synthesizer/
docker compose build synthesis_service && docker compose up -d synthesis_service

# Аналогично для data_service / evaluation_service / reporting_service
```

### GPU

Оба сервиса `app` и `synthesis_service` в `docker-compose.yml` резервируют один NVIDIA-GPU. Проверить доступность:

```bash
docker compose exec synthesis_service python -c "import torch; print(torch.cuda.get_device_name(0))"
```

При отсутствии GPU обучение идёт на CPU — значительно медленнее для DP-TVAE.

---

## Переменные окружения (`.env`)

Шаблон — `final_system/.env.example`. Ключевые переменные:

| Переменная | Назначение | Дефолт |
|---|---|---|
| `DB_PASSWORD` | Пароль системной PostgreSQL | — (обязательно) |
| `DB_DISABLED` | `true` — не использовать ProcessRegistry | `false` |
| `REDIS_URL` | URL Redis | `redis://redis:6379/0` |
| `API_KEY` | Bearer-токен; если не задан — авторизация отключена | не задан |
| `DB_IMPORT_DSN` | DSN БД-источника для `data_import.type: postgres` | не задан |
| `DB_EXPORT_DSN` | DSN БД-приёмника для `data_export.type: postgres` | не задан |
| `*_SERVICE_URL` | Адреса микросервисов (подставляются автоматически в Docker) | см. compose |

---

## Пример: запуск пайплайна через API

```bash
# 1. Загрузить датасет
curl -X POST http://localhost:8000/api/v1/datasets \
  -F "name=adult" -F "file=@data/adult.csv"

# 2. Создать запуск с инлайн-конфигом
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d @configs/adult.yaml

# 3. Проверить статус (run_id из ответа шага 2)
curl http://localhost:8000/api/v1/runs/{run_id}

# 4. Получить JSON-отчёт
curl http://localhost:8000/api/v1/runs/{run_id}/report

# 5. Скачать синтетику
curl http://localhost:8000/api/v1/runs/{run_id}/synthetic -o synth.csv

# 6. Список сохранённых моделей
curl http://localhost:8000/api/v1/models

# 7. Семплирование из сохранённой модели (без расходования ε)
curl -X POST http://localhost:8000/api/v1/models/{model_id}/samples \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "output_format": "csv"}' \
  -o sampled.csv
```

Полный список эндпоинтов — на `/docs` (Swagger UI).

### Авторизация

По умолчанию отключена. Для включения:

```bash
# в .env
API_KEY=$(openssl rand -hex 32)
```

После рестарта `app` все запросы требуют заголовок `Authorization: Bearer <API_KEY>`.

---

## Конфигурация (YAML)

Основные конфиги в `final_system/configs/`:

| Файл | Генератор | DP | Назначение |
|---|---|---|---|
| `adult.yaml` | DP-CTGAN | ε=3.0 | Основной пример (Adult Census) |
| `adult_dptvae.yaml` | DP-TVAE | σ-based | Второй DP-генератор |
| `adult_ctgan.yaml` | CTGAN | — | Baseline без DP |
| `adult_tvae.yaml` | TVAE | — | Baseline без DP |
| `adult_copulagan.yaml` | CopulaGAN | — | Baseline без DP |
| `adult_dpctgan_db.yaml` | DP-CTGAN | ε=3.0 | С импортом/экспортом через PostgreSQL |

Структура конфига валидируется через Pydantic v2 в `config_loader.py`:

| Секция | Содержимое |
|---|---|
| `pipeline` | sample_size, holdout_size, random_state, n_synth_rows, max_iterations |
| `generator` | generator_type, epsilon, delta, sigma, epochs, batch_size, cuda, random_seed |
| `data_schema` | auto-детекция или явное задание `categorical`/`continuous`/`exclude` |
| `data_import` | type: `csv` / `postgres`, путь или SQL-запрос |
| `data_export` | type: `csv` / `postgres`, путь или имя таблицы |
| `privacy` | quasi_identifiers, sensitive_attribute |
| `utility` | target_column, task_type (classification/regression) |
| `thresholds` | max_utility_loss, max_mean_jsd, max_mia_auc, require_dp_enabled и др. |

### Пороги вердикта (по умолчанию)

| Метрика | PASS-критерий | Источник |
|---|---|---|
| `ε_spent` | ≤ `ε_target` | DP-SGD Privacy Accountant |
| MIA AUC | ≤ 0.55 | Distance-based proxy |
| DCR | `median(DCR_syn) > median(DCR_holdout)` | Дистанционная метрика |
| k-анонимность | ≥ 5 (настраивается) | ПНСТ |
| JSD (средняя) | < 0.40 | ПНСТ |
| TSTR F1 loss | < 25% от TRTR F1 | ML-бенчмарк |

Для генераторов без DP (CTGAN/TVAE/CopulaGAN) в конфиге должно быть `require_dp_enabled: false`.

---

## Эндпоинты API

| Метод | Путь | Описание |
|---|---|---|
| POST | `/api/v1/datasets` | Загрузить датасет (multipart CSV) |
| POST | `/api/v1/datasets/from-db` | Загрузить датасет из PostgreSQL |
| GET | `/api/v1/datasets` | Список датасетов |
| POST | `/api/v1/runs` | Запустить пайплайн |
| GET | `/api/v1/runs/{run_id}` | Статус и метаданные запуска |
| GET | `/api/v1/runs/{run_id}/report` | JSON-отчёт валидации |
| GET | `/api/v1/runs/{run_id}/synthetic` | Скачать синтетику (CSV/JSON) |
| DELETE | `/api/v1/runs/{run_id}` | Отменить активный / удалить завершённый |
| GET | `/api/v1/models` | Список сохранённых моделей (фильтр по `dataset_name`) |
| GET | `/api/v1/models/{model_id}` | Метаданные модели |
| DELETE | `/api/v1/models/{model_id}` | Удалить модель |
| POST | `/api/v1/models/{model_id}/samples` | Семплирование из сохранённой модели |
| GET | `/api/v1/configs` | CRUD конфигов |
| GET | `/api/v1/health` | Healthcheck |

Полная спецификация с примерами — на `/docs`.

---

## Структура репозитория

```
.
├── CLAUDE.md                # Инструкции для Claude Code (контекст кодовой базы)
├── ARCHITECTURE.md          # Детальное описание архитектуры
├── REQUIREMENTS.md          # Функциональные и нефункциональные требования
├── requirements.txt         # Зависимости для локальной разработки
├── db/
│   ├── init_db.sql          # Схема системной PostgreSQL
│   └── init_user_db.sh      # Инициализация user_db (raw_data + synth_data)
└── final_system/
    ├── docker-compose.yml   # Оркестрация всех сервисов
    ├── Dockerfile           # Образ Gateway
    ├── .env.example         # Шаблон переменных окружения
    ├── config_loader.py     # YAML → Pydantic-объекты (без ML-зависимостей)
    ├── api/                 # Gateway (порт 8000)
    │   ├── main.py          # FastAPI entry point
    │   ├── settings.py      # Pydantic Settings
    │   ├── store.py         # Redis-backed RunStore
    │   ├── clients.py       # httpx ServiceClient + polling
    │   ├── dependencies.py  # Bearer-авторизация
    │   ├── routers/         # runs, datasets, models, configs, system
    │   └── schemas/         # Pydantic схемы эндпоинтов Gateway
    ├── services/
    │   ├── data_service/       # порт 8001
    │   ├── synthesis_service/  # порт 8002 (тяжёлый образ с GPU)
    │   ├── evaluation_service/ # порт 8003
    │   └── reporting_service/  # порт 8004
    ├── shared/
    │   ├── schemas/         # Pydantic-схемы, общие между сервисами
    │   └── log_context.py   # ContextVar с run_id для сквозного логирования
    ├── synthesizer/         # DP-CTGAN, DP-TVAE, SDV-генераторы, BaseGenerator
    ├── evaluator/
    │   ├── privacy/         # k/l/t, DCR, NNDR, MIA
    │   └── utility/         # JSD, TVD, TSTR/TRTR, correlations
    ├── data_processor/      # Preprocessing, type detection, minimization
    ├── configs/             # YAML-конфиги под разные генераторы
    └── tests/               # pytest
```

`archive/` в корне — старые прототипы, не используются.

---

## Тесты

```bash
# Из корня репозитория (не из final_system/):
python -m pytest final_system/tests/ -v
```

---

## База данных

### Системная PostgreSQL (`postgres`, порт 5433)

Хранит `ProcessRegistry` — метаданные запусков (не исходные данные). Инициализируется автоматически из `db/init_db.sql` при первом `docker compose up`. Схема `synthetic_data_schema`, таблицы: `processes`, `process_logs`, `process_metadata`.

Подключение:

```bash
docker compose exec postgres psql -U postgres -d synthetic_data_db
\dt synthetic_data_schema.*
```

Отключается через `DB_DISABLED=true` в `.env` — система будет работать только с Redis.

### БД пользователя (`user_db`, порт 5434)

Имитирует внешнюю БД заказчика с двумя таблицами:
- `raw_data` — источник данных для `data_import.type: postgres`
- `synth_data` — приёмник синтетики для `data_export.type: postgres`

Используется для демонстрации сценария "end-to-end через БД". Конфигурируется через `DB_IMPORT_DSN` / `DB_EXPORT_DSN`.

---

## Воспроизводимость

- Во всех генераторах зафиксирован `random_seed` (по умолчанию 42) — проброшен в torch, numpy и SDV
- Sidecar `.meta.json` рядом с `.pkl` сохраняет: `run_id`, `dataset_name`, `dp_config` (ε/δ/σ), `dp_spent` (потраченный ε + история по эпохам), `created_at`
- Повторное семплирование из сохранённой модели (`POST /models/{id}/samples`) **не расходует** ε-бюджет — DP расходуется только в `fit()`

```bash
# Получить метаданные модели
curl http://localhost:8000/api/v1/models/{model_id}

# Сгенерировать 50k строк без повторного обучения
curl -X POST http://localhost:8000/api/v1/models/{model_id}/samples \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 50000, "output_format": "csv"}' \
  -o resampled.csv
```

---

## Ограничения

- **Single-table only** — мульти-таблица (relational) не поддерживается
- **Синтез job state in-memory** — при рестарте `synthesis_service` активные обучения теряются; Gateway получит 404 при поллинге и run перейдёт в FAIL
- **`--workers 1`** на всех сервисах — shared volume не поддерживает параллельную запись
- **DP-CTGAN при σ=5.0** — возможна потеря utility ~20% на редких категориях (mode collapse). Снижайте σ до 1.5–2.0 или поднимайте ε до 5–8 при строгих требованиях к полезности.

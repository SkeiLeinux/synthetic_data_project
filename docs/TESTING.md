# TESTING.md — Стратегия и реестр тестов

**Документ описывает:** уровни тестирования, фикстуры, что именно
проверяет каждый из 5 тестовых файлов в `final_system/tests/`,
и команды для их запуска.

**Связанные документы:**
* [REQUIREMENTS.md § 11](REQUIREMENTS.md#11-критерии-приёмки-и-тестирование) — критерии приёмки
* [SAD.md § 12](SAD.md#12-качественные-атрибуты) — testability как атрибут качества

---

## 1. Уровни тестирования

| Уровень | Что проверяет | Marker | Зависимости |
|---|---|---|---|
| **Unit** | чистая логика без сети и I/O — `config_loader`, `DataProcessor`, `Reporter._compute_verdict` | — (по умолчанию) | только pip-зависимости |
| **E2E (smoke)** | полный пайплайн через docker-compose | `@pytest.mark.e2e` | поднятый docker-compose stack |

Component-уровень (один сервис в изоляции через FastAPI `TestClient`) —
запланирован, но пока не реализован. См. PRD § 11.1.

---

## 2. Конфигурация pytest

Все маркеры регистрируются в `final_system/tests/conftest.py`:

```python
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end integration tests that require a running docker-compose stack",
    )
```

Дополнительный `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]`
не используется. Все unit-тесты добавляют корень `final_system/`
в `sys.path` через стандартный пролог:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

Это позволяет запускать тесты как из корня репозитория
(`pytest final_system/tests/`), так и из любой другой директории.

---

## 3. Команды запуска

### Из корня репозитория

```bash
# Все unit-тесты (быстрые, без Docker)
pytest final_system/tests/ -m "not e2e" -v

# Конкретный файл
pytest final_system/tests/test_config_loader.py -v

# Конкретный тест
pytest final_system/tests/test_reporter_verdict.py::test_verdict_pass_all_good -v

# E2E (требует запущенный docker-compose)
pytest final_system/tests/ -m e2e -v

# Всё подряд (unit + e2e)
pytest final_system/tests/ -v
```

### Подготовка стека для E2E

```bash
cd final_system
cp .env.example .env
docker compose up -d --wait        # ждёт healthcheck'ов
# ИЛИ запустить тест — он сам поднимет стек, если не запущен:
cd ..
pytest final_system/tests/ -m e2e -v
```

`test_e2e_microservices.py::docker_stack` — sessions-scoped fixture,
которая проверяет `/health` всех 5 сервисов и при необходимости
делает `docker compose up -d --wait`.

---

## 4. Реестр тестов

### 4.1. `tests/conftest.py`

Регистрирует маркер `@pytest.mark.e2e`. Не содержит фикстур —
unit-тесты используют локальные фикстуры в каждом файле.

### 4.2. `tests/test_config_loader.py` — Unit (≈25 кейсов)

Покрывает: `final_system/config_loader.py`.

**Группы тестов:**

| Группа | Что проверяется |
|---|---|
| Успешная загрузка | Минимальный валидный YAML создаёт `AppConfig`; `load_config('configs/adult.yaml')` работает; дефолты применяются (`epsilon=3.0`, `holdout_size=0.2`, `random_state=42`, `cardinality_threshold=0.9`) |
| Валидация `epsilon` | `epsilon=0` и `epsilon<0` → `ValidationError`; `preprocessor_eps >= epsilon` → `ValidationError`; `preprocessor_eps=0.5 < epsilon=3.0` валидно |
| Валидация `holdout_size` | `0.0` и `1.0` → `ValidationError`; `0.3` валидно |
| Валидация `data_source` | Неизвестный тип (`ftp`) → `ValidationError`; `csv` без `data_path` → `ValidationError`; `db` без `db_query` → `ValidationError` |
| `DataSchemaYamlConfig` | `is_auto=True` если `categorical` и `continuous` пусты; `is_auto=False` иначе; дефолты `direct_identifiers=[]`, `drop_high_cardinality=False`, `cardinality_threshold=0.9` |
| `apply_quick_test()` | Перезаписывает `sample_size→5000`, `epochs→50`, `cuda→False`, `distance_sample_size→500`, `mia_sample_size→250`, `max_utility_loss→0.40`, `max_mean_jsd→0.40`, `max_mia_auc→0.65`, `max_spent_epsilon→None` |
| Иммутабельность | `apply_quick_test(cfg)` возвращает новый объект, не мутирует исходный |
| Файловые ошибки | `load_config('nonexistent.yaml')` → `FileNotFoundError` |

**Минимальный валидный raw-конфиг** (используется в большинстве тестов):

```python
{
    "pipeline": {"dataset_name": "test", "data_source": "csv", "data_path": "data/test.csv"},
    "utility": {"target_column": "label"},
}
```

### 4.3. `tests/test_processor_core.py` — Unit (≈15 кейсов)

Покрывает: `final_system/data_processor/processor.py:DataProcessor.{preprocess, detect_column_types, get, drop_columns}`.

**Фикстуры:**
* `mixed_df` — DataFrame со смешанными типами, дубликатами, пропусками (для `preprocess`).
* `schema_df` — 20 строк со столбцами разных типов (для `detect_column_types`).

**Группы тестов:**

| Группа | Что проверяется |
|---|---|
| `preprocess()` | Полные дубликаты строк удаляются; числовые null заполняются median, категориальные null — mode; колонки не меняются; чистый DataFrame не меняется |
| `detect_column_types()` | `object` → `categorical`; `bool` → `categorical`; числовая колонка с >15 уникальных → `continuous`; числовая с ≤15 уникальных → `categorical`; `datetime` → `ignored`; `exclude_columns` идут в `ignored`; `force_categorical`/`force_continuous` перекрывают автодетекцию; каждая колонка попадает ровно в один список (categorical / continuous / ignored) |
| Граничные случаи | `force_*` overrides работают для уже классифицированных колонок |
| Утилиты | `get()` возвращает текущий `df`; `drop_columns()` удаляет существующие; не падает на несуществующих именах |

**Ключевая константа:** `_MAX_UNIQUE_FOR_CATEGORICAL = 15` — числовая
колонка с числом уникальных значений ≤15 классифицируется как
`categorical`. Тесты используют `n=20`, чтобы числовые колонки
гарантированно попадали в `continuous`.

### 4.4. `tests/test_processor_minimize.py` — Unit (≈7 кейсов)

Покрывает: `DataProcessor.minimize()`.

**Фикстура `sample_df`:** 5 строк, 6 колонок (`id`, `email`, `name`, `age`, `income`, `city`).

**Группы тестов:**

| Группа | Что проверяется |
|---|---|
| `direct_identifiers` | Перечисленные колонки удаляются; report содержит `removed_direct_identifiers`, `columns_before/after` |
| Несуществующие колонки | Не приводят к ошибке, не попадают в report |
| `drop_high_cardinality=True` | Удаляет object-колонки, у которых `unique/total > cardinality_threshold` (например, `email`, `name` с 100% уникальных при пороге 0.9); НЕ удаляет колонки с низкой уникальностью (`city` — 40%) |
| Защита числовых | Числовые колонки (`age`) не удаляются по cardinality, даже при пороге 0.1 |
| Инвариант строк | `len(df_result) == len(sample_df)` всегда; `report["rows_unchanged"] is True` |
| Структура отчёта | Обязательные ключи: `removed_direct_identifiers`, `removed_high_cardinality`, `columns_before`, `columns_after`, `rows_unchanged` |

### 4.5. `tests/test_reporter_verdict.py` — Unit (≈12 кейсов)

Покрывает: `services/reporting_service/reporter.py:Reporter.{build, _compute_verdict}`.

**Вспомогательные конструкторы** (не фикстуры — обычные функции):

```python
_good_dp_report(is_dp_enabled=True, spent_eps=2.5)     # → dict с dp_config + dp_spent
_good_utility_report(utility_loss=0.10, mean_jsd=0.15) # → dict с ml_efficacy + statistical + correlations
_good_privacy_report(attack_auc=0.51, dcr_privacy_preserved=True)  # → dict с empirical_risk + dp_guarantees + diagnostic
```

**Фикстура `reporter`:** `Reporter(VerdictThresholds(max_utility_loss=0.25, max_mean_jsd=0.40, max_mia_auc=0.60, require_dcr_privacy_preserved=True, require_dp_enabled=True, max_spent_epsilon=None))`.

**Группы тестов:**

| Группа | Что проверяется |
|---|---|
| **PASS** | Все три отчёта в норме → `overall=PASS`, `issues=[]`, `utility_ok=privacy_ok=dp_ok=True` |
| **FAIL — utility** | `utility_loss=0.50 > 0.25` → FAIL, в `issues` есть строка с "Utility Loss"; `mean_jsd=0.50 > 0.40` → FAIL, в `issues` есть "JSD" |
| **FAIL — privacy** | `attack_auc=0.75 > 0.60` → FAIL, в `issues` есть "MIA AUC"; `dcr_privacy_preserved=False` при `require_dcr_privacy_preserved=True` → FAIL, в `issues` есть "DCR" |
| **FAIL — DP** | `is_dp_enabled=False` при `require_dp_enabled=True` → FAIL, `dp_ok=False`; `spent_epsilon=3.5 > max_spent_epsilon=2.0` → FAIL, в `issues` есть "epsilon" |
| **PARTIAL** | `privacy_report=None` → `overall=PARTIAL`, `privacy_ok=None`; то же для `utility_report=None`; все три `None` → `overall=PARTIAL` |
| Структура отчёта | `data_processing.minimization` присутствует, если передан `minimization_report`; иначе `{}` |

**Важно:** конкретные строки в `issues` проверяются через
`any("Utility Loss" in issue ...)`, поэтому формулировки можно менять,
но ключевые слова (`Utility Loss`, `JSD`, `MIA AUC`, `DCR`, `DP`,
`epsilon`) должны оставаться.

### 4.6. `tests/test_e2e_microservices.py` — E2E (1 кейс)

Покрывает: полный пайплайн через docker-compose.

**Маркер:** `@pytest.mark.e2e`.

**Константы:**

```python
GATEWAY = "http://localhost:8000"
COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.yml"
POLL_INTERVAL = 5    # секунд между опросами статуса run
TIMEOUT = 600        # максимум ожидания завершения (10 минут)
```

**Helper `_all_services_healthy()`:**
GET на `/health` каждого сервиса:
* Gateway: `:8000/api/v1/health`
* Data: `:8001/health`
* Synthesis: `:8002/health`
* Evaluation: `:8003/health`
* Reporting: `:8004/health`

Возвращает `True` только если все вернули 200.

**Session-scoped fixture `docker_stack`:**
1. Если все сервисы healthy — сразу `yield`.
2. Иначе — `docker compose -f final_system/docker-compose.yml up -d --wait`.
3. Ждёт healthcheck'ов (до 30 итераций × 5 сек = 150 сек).
4. Если стек так и не стал healthy — `pytest.fail(...)`.

**Тест `test_full_pipeline_ctgan`:**

1. **Запуск:** `POST /api/v1/runs` с body `{"config_name": "e2e_ctgan", "quick_test": False}`.
   * Ожидание: HTTP 202, в ответе `run_id`.
2. **Поллинг:** `GET /api/v1/runs/{run_id}` каждые 5 сек, пока `status` не станет `completed` или `failed`.
   * Таймаут: 600 сек.
3. **Assertions:**
   * `final_status == "completed"` (любая ошибка падает с `error_message` из RunDetail).
   * `verdict in ("PASS", "FAIL", "PARTIAL")` — конкретное значение не проверяется (на CTGAN с 5 эпохами почти всегда FAIL).
   * `synth_rows is not None and synth_rows > 0`.
   * `config_snapshot` — `dict`.

**Используемый конфиг — `configs/e2e_ctgan.yaml`** (минимальный CTGAN, 5 эпох, без DP, без GPU). Датасет — `data/adult_tiny.csv` (урезанная версия Adult Census).

Цель теста — **не** оценить качество синтетики, а проверить, что
пайплайн проходит сквозь все 7 шагов без падений (smoke test).

---

## 5. Покрытие по компонентам

| Компонент | Тип покрытия | Где |
|---|---|---|
| `config_loader.py` (валидация YAML, quick_test) | Unit | `test_config_loader.py` |
| `DataProcessor.preprocess()` | Unit | `test_processor_core.py` |
| `DataProcessor.detect_column_types()` | Unit | `test_processor_core.py` |
| `DataProcessor.minimize()` | Unit | `test_processor_minimize.py` |
| `Reporter._compute_verdict()` | Unit | `test_reporter_verdict.py` |
| Gateway routers (`runs`, `datasets`, `models`, `configs`, `system`) | E2E only | `test_e2e_microservices.py` (через `/api/v1/runs`) |
| `ServiceClient` (httpx, polling) | E2E only | косвенно через E2E |
| `RunStore` (Redis WATCH/MULTI/EXEC) | E2E only | косвенно через E2E |
| Data Service router | E2E only | косвенно через `_execute_pipeline` |
| Synthesis Service router (`POST /jobs`, polling, sample) | E2E only | косвенно через `_execute_pipeline` |
| Evaluation Service router | E2E only | косвенно |
| Reporting Service router | E2E only | косвенно |
| Генераторы (DP-CTGAN, DP-TVAE, CTGAN, TVAE, CopulaGAN) | Не покрыты unit-тестами | проверяются библиотеками upstream и e2e (только CTGAN) |
| Privacy/utility evaluators (DCR, MIA, JSD, TSTR/TRTR) | Не покрыты unit-тестами | проверяются e2e на структуру отчётов |

**Пробелы (известный технический долг):**

1. Component-тесты сервисов через FastAPI `TestClient` — не реализованы.
2. Unit-тесты на `RunStore` (например, поведение при `WatchError`).
3. Unit-тесты на отдельные DP-генераторы (хотя бы smoke `fit/sample` на синтетических данных).
4. Unit-тесты на `PrivacyEvaluator` и `UtilityEvaluator` (на mock-данных с известным ответом).
5. E2E-вариант с DP-генератором (DP-CTGAN) — текущий e2e использует только CTGAN без DP, чтобы укладываться в timeout.

---

## 6. Acceptance criteria для релиза MVP

Из PRD § 11.3:

1. `docker compose up -d --build` поднимает все 7 контейнеров; healthcheck'и зелёные за < 3 минут.
2. `pytest final_system/tests/ -v` проходит полностью (unit + e2e).
3. `POST /api/v1/runs` с `e2e_ctgan` завершается со статусом `completed` (вердикт любой).
4. `POST /api/v1/runs` с `adult.yaml` (DP-CTGAN, 300 эпох) на полном Adult-датасете завершается с **PASS**: `mean_jsd < 0.20`, `utility_loss < 0.15`, `MIA AUC < 0.55`, `spent_ε ≤ 5.0`, `is_dp_enabled = true`.
5. `GET /metrics` возвращает Prometheus-совместимый формат с непустыми счётчиками.
6. Авторизация: запрос без `Authorization` при заданном `API_KEY` → 401.

Пункты 1, 2, 3 автоматизированы; пункты 4, 5, 6 — ручная приёмка.

---

## 7. Зависимости для запуска тестов

В корневом `requirements.txt`:

```
pytest>=7.0
requests>=2.28          # для test_e2e_microservices
pandas>=2.0             # для всех processor-тестов
pyyaml>=6.0             # для config_loader
pydantic>=2.0           # для config_loader
```

Для unit-тестов **не требуется** Docker, ML-зависимости (torch, opacus,
smartnoise, sdv) или сетевые соединения. Это ключевое следствие
ADR-012 (lazy ML imports в `config_loader.py`) — unit-тесты можно
запускать в CI на лёгком Python-образе без 13-гигабайтного
synthesis-образа.

E2E-тест требует Docker Desktop + образы из `docker-compose.yml`.
Если стек не запущен, fixture `docker_stack` сделает `up -d --wait`.

---

## 8. Что добавить при расширении

При добавлении нового кода в кодовую базу следуйте конвенциям:

| Добавляете | Куда писать тесты | Тип |
|---|---|---|
| Новый генератор (`synthesizer/my_gen.py`) | `tests/test_synthesizer_my_gen.py` (smoke fit/sample) + расширить e2e | Unit + E2E |
| Новую метрику в evaluator | `tests/test_evaluator_<metric>.py` с известным ответом | Unit |
| Новый эндпоинт Gateway | Component через `TestClient(app)` | Component |
| Новый раздел YAML-конфига | `tests/test_config_loader.py` (валидация + дефолты) | Unit |
| Новый порог в `VerdictThresholds` | `tests/test_reporter_verdict.py` (PASS + FAIL граница) | Unit |
| Изменение пайплайна (`_execute_pipeline`) | Расширить `test_e2e_microservices.py` | E2E |

**Конвенция именования:** `tests/test_<module>_<aspect>.py`
(например, `test_processor_minimize.py`).

**Конвенция структуры файла:**
1. Стандартный пролог с `sys.path.insert`.
2. Импорты тестируемого модуля.
3. Локальные фикстуры (`@pytest.fixture`) или helper-функции с `_`-префиксом.
4. Группы тестов, разделённые комментариями-разделителями
   (`# ──────────`).
5. Каждая группа — однородная (валидация / happy path / errors).

---

*Документ описывает тестовую инфраструктуру по состоянию репозитория
на 2026-04-25. При добавлении тестов обновите раздел 4 (реестр)
и при необходимости раздел 5 (покрытие).*

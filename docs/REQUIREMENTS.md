# REQUIREMENTS.md — Product Requirements Document

**Продукт:** Сервис генерации конфиденциальных синтетических табличных данных с гарантиями дифференциальной приватности (далее — **«Система»** или **DP-Synth Platform**)
**Версия документа:** 1.0
**Дата:** 2026-04-25
**Автор:** магистерская выпускная квалификационная работа, НИУ ВШЭ
**Статус:** прототип уровня MVP, готовый к академической защите; требует доработки до уровня корпоративного MLOps-продукта (см. раздел 12)

---

## Содержание

1. [Контекст и обоснование](#1-контекст-и-обоснование)
2. [Цели и не-цели продукта](#2-цели-и-не-цели-продукта)
3. [Пользователи, роли и сценарии](#3-пользователи-роли-и-сценарии)
4. [Объём системы (Scope) и допущения](#4-объём-системы-scope-и-допущения)
5. [Функциональные требования (FR)](#5-функциональные-требования-fr)
6. [Нефункциональные требования (NFR)](#6-нефункциональные-требования-nfr)
7. [Архитектура](#7-архитектура)
8. [Контракты API и модели данных](#8-контракты-api-и-модели-данных)
9. [Конфигурация пайплайна](#9-конфигурация-пайплайна)
10. [Метрики качества и приватности](#10-метрики-качества-и-приватности)
11. [Критерии приёмки и тестирование](#11-критерии-приёмки-и-тестирование)
12. [**Дальнейшие доработки до уровня корпоративного MLOps**](#12-дальнейшие-доработки-до-уровня-корпоративного-mlops) ⭐
13. [Глоссарий](#13-глоссарий)

---

## 1. Контекст и обоснование

### 1.1. Бизнес-проблема

Крупные организации (банки, страховые, телеком, ритейл, медицинские холдинги, государственные информационные системы) накапливают объёмные таблицы с персональными и чувствительными данными. Эти данные нужны:

* командам аналитики и BI — для построения отчётности и сегментации;
* командам Data Science — для обучения предсказательных моделей;
* партнёрам и подрядчикам — для совместной разработки и кросс-проверки гипотез;
* DevOps/SQA — для нагрузочного и регрессионного тестирования.

Прямой обмен реальными персональными данными ограничен **ФЗ-152 «О персональных данных»**, **GDPR**, отраслевыми стандартами (PCI DSS, HIPAA, СТО БР ИББС), а также внутренними политиками information security. Классические методы анонимизации (k-анонимность, l-разнообразие) подвержены атакам реидентификации (linkage / homogeneity / background-knowledge attacks) и не дают формальных гарантий.

**Дифференциальная приватность (DP)** — математический фреймворк, дающий формальную гарантию: «выходное распределение модели меняется не более чем в e^ε раз при добавлении/удалении одной записи». Это переводит вопрос приватности из эвристики в количественную плоскость.

### 1.2. Что делает Система

Система автоматизирует **полный цикл** генерации синтетических табличных данных с DP-гарантиями:

```
[CSV / PostgreSQL] → препроцессинг → DP-обучение генератора →
сэмплирование → оценка приватности и полезности → итоговый вердикт PASS/FAIL/PARTIAL
                                              ↓
                                    JSON-отчёт + синтетический CSV
```

Результат — синтетическая таблица, статистически близкая к исходной, но с проверяемыми гарантиями приватности (формальные DP-метрики + эмпирические тесты на меморизацию и атаки восстановления членства).

### 1.3. Соответствие нормативной базе

* **ПНСТ (предварительный национальный стандарт) ГОСТ Р по синтетическим данным** — Система реализует все четыре домена: «генерация», «оценка полезности», «оценка приватности», «формальные гарантии».
* **ФЗ-152, ст. 3 п.9** — синтетические данные не относятся к персональным, что снимает регуляторные требования по их хранению и передаче.
* **GDPR Art. 4 (5)** — генерация подпадает под понятие «pseudonymisation / anonymisation» при выполнении DP-условий.

### 1.4. Origin: магистерская работа → корпоративная платформа

Текущая реализация — **прототип уровня MVP**, выполненный как магистерская ВКР: показывает методологически корректный пайплайн, рабочую микросервисную архитектуру, набор DP- и не-DP-генераторов, и количественную оценку результата. Для **внедрения в MLOps-процессы крупной компании** требуется ряд доработок (см. раздел [12](#12-дальнейшие-доработки-до-уровня-корпоративного-mlops)).

---

## 2. Цели и не-цели продукта

### 2.1. Цели (in-scope)

| № | Цель | Метрика успеха                                           |
|---|---|----------------------------------------------------------|
| G1 | Генерация синтетических данных, неотличимых по статистике от исходных | mean JSD < 0.20, mean TVD < 0.20                         |
| G2 | Сохранение полезности данных для ML-задач | TSTR/TRTR utility loss < 15%                             |
| G3 | Формальные DP-гарантии для каждого артефакта | spent ε ≤ ε_target, δ ≤ 1/(n·√n)                         |
| G4 | Эмпирическая защита от меморизации | MIA AUC ≤ 0.55, DCR_synth ≥ DCR_holdout                  |
| G5 | Сквозной автоматический вердикт по приватности и качеству | PASS / FAIL / PARTIAL в JSON-отчёте                      |
| G6 | REST-API для интеграции в существующие пайплайны | OpenAPI 3.1, Swagger UI, Bearer-auth                     |
| G7 | Воспроизводимость экспериментов | random_seed во всех конфигах + сохранение config snapshot |
| G8 | Полное логгирование с трассировкой по run_id | сквозной run_id во всех 7 контейнерах                    |

### 2.2. Не-цели (out-of-scope для MVP)

* Генерация **временных рядов**, **изображений**, **текста**, **графовых данных** — текущая Система работает только с табличными данными.
* Генерация по **связанным таблицам** (multi-table relational synthesis) — поддерживается только одна таблица за один прогон.
* **Federated learning** между несколькими data owner'ами — Система предполагает централизованное обучение.
* **Автоматический выбор гиперпараметров** генератора (HPO/AutoML) — гиперпараметры задаются вручную в YAML-конфиге.
* **Real-time / streaming-генерация** — пайплайн пакетный (batch).
* **Управление identity и tenant-изолированный multi-tenant режим** — см. раздел [12](#12-дальнейшие-доработки-до-уровня-корпоративного-mlops).
* **Интеграция с витринами данных** (Snowflake, BigQuery, Databricks) — на текущем этапе только CSV и PostgreSQL.

---

## 3. Пользователи, роли и сценарии

### 3.1. Целевые пользователи

* **U1 — Data Scientist / ML-инженер.** Хочет получить синтетический датасет вместо реальных PII для разработки и обучения моделей.
* **U2 — Data Engineer / Platform Engineer.** Поддерживает пайплайн, конфигурирует подключения к источникам, мониторит SLA, разбирается с инцидентами.
* **U3 — Compliance officer / DPO (Data Protection Officer).** Сверяет DP-параметры, проверяет финальный вердикт, хранит JSON-отчёты как доказательную базу.
* **U4 — Аналитик BI.** Загружает синтетические данные в BI-инструменты для регулярных отчётов без касания реальных PII.
* **U5 — Системный администратор / SRE.** Обеспечивает работоспособность стека, поднимает/гасит контейнеры, следит за здоровьем.

### 3.2. Ключевые сценарии (User Stories)

**US-1.** Как Data Scientist, я хочу загрузить CSV, выбрать преднастроенный YAML-конфиг и получить синтетический CSV с автоматическим вердиктом, чтобы быстро проверить пригодность синтетики для конкретной задачи.

**US-2.** Как Compliance officer, я хочу видеть в JSON-отчёте `verdict.overall = PASS` + `dp_spent.spent_epsilon_final` ≤ заявленного, чтобы зафиксировать факт корректности применения DP.

**US-3.** Как Data Engineer, я хочу загрузить данные напрямую из PostgreSQL по DSN из переменной окружения и записать результат обратно в БД, чтобы избежать промежуточных CSV-файлов.

**US-4.** Как ML-инженер, я хочу один раз обучить генератор (`save_model=true`), а затем многократно сэмплировать произвольное число строк через `POST /models/{id}/samples` без повторного расхода DP-бюджета, чтобы получать дополнительные подвыборки.

**US-5.** Как разработчик пайплайна, я хочу опционально включить `quick_test=true`, чтобы прогнать сокращённый smoke-test за минуты вместо часов и получить быструю обратную связь.

**US-6.** Как SRE, я хочу проверить `/health`, `/health/db`, `/health/gpu` и `/metrics` (Prometheus-формат), чтобы интегрировать сервис в существующий мониторинг.

**US-7.** Как пользователь API, я хочу получить уведомление по `webhook_url` о завершении длительной задачи, чтобы не опрашивать `/runs/{id}` периодически.

---

## 4. Объём системы (Scope) и допущения

### 4.1. В объёме MVP

* Семь сервисов в Docker Compose (Gateway, Data, Synthesis, Evaluation, Reporting, Redis, две БД PostgreSQL).
* Пять генераторов: **DP-CTGAN**, **DP-TVAE**, **CTGAN**, **TVAE**, **CopulaGAN** (последние три — без DP, как baseline).
* Импорт из **CSV** или **PostgreSQL**, экспорт в **CSV** или **PostgreSQL**.
* Полный набор метрик: статистических (JSD, TVD, корреляции), ML-эффективности (TSTR/TRTR), приватности (k/l/t-anonymity, DCR, NNDR, MIA proxy), формальных DP-гарантий.
* JSON-отчёт + автоматический вердикт PASS/FAIL/PARTIAL по настраиваемым порогам.
* Bearer-token аутентификация на Gateway.
* Сохранение моделей с metadata-сайдкарами (`*.pkl` + `*.meta.json`).
* End-to-end smoke-тест (`pytest -m e2e`).

### 4.2. Допущения

* Один датасет умещается в RAM рабочего узла (порядок до 10 миллионов строк × 50 колонок). Распределённое обучение и стриминг данных — out-of-scope.
* Один прогон пайплайна выполняется на одной машине — никаких worker-кластеров.
* Доступная GPU-память — от 8 ГБ (T4 / A10) для DP-генераторов на разумных датасетах.
* Все микросервисы запускаются в одной сети Docker, доступ между ними осуществляется по внутренним DNS (имена сервисов).
* Пользователь готов вручную задавать YAML-конфиг (квазиидентификаторы, target column, schema, пороги) — автодетекция этих полей частична.

### 4.3. Ограничения текущего MVP (известные)

* Состояние джобов синтеза хранится **in-memory** (`synthesis_service/job_store.py`) и теряется при рестарте контейнера. Pod restart = потеря джоба.
* Используется **один воркер** на каждый сервис (`uvicorn --workers 1`) — разделяемый том `/data` не поддерживает конкурентную запись.
* `/data` — **локальный bind-mount Docker volume**, не сетевой (NFS/S3/Ceph).
* DP-бюджет **не аккумулируется централизованно** между запусками — каждая модель тратит бюджет независимо.
* Аутентификация — **single shared API_KEY**, без RBAC.
* Webhook-уведомления через `urllib.request` без подписи / retries / dead letter queue.

---

## 5. Функциональные требования (FR)

### 5.1. Управление датасетами (Data Service, port 8001 + Gateway-обёртка)

| ID | Требование | Приоритет |
|---|---|---|
| FR-01.1 | Загрузка CSV-файла через `POST /datasets` с базовой валидацией парсинга `pandas.read_csv` | MUST |
| FR-01.2 | Импорт из PostgreSQL по DSN-переменной окружения (`POST /datasets/from-db`) | MUST |
| FR-01.3 | Возврат метаданных датасета: `dataset_id`, `rows`, `columns`, `file_size_bytes`, `uploaded_at` | MUST |
| FR-01.4 | Список датасетов с фильтрацией по `dataset_name` и пагинацией (`GET /datasets`) | MUST |
| FR-01.5 | Удаление датасета (`DELETE /datasets/{name}`) | SHOULD |
| FR-01.6 | Получение детектируемой схемы колонок (`GET /datasets/{name}/schema`) | SHOULD |
| FR-01.7 | Превью первых N строк + summary-статистики (`POST /datasets/{name}/preview`) | SHOULD |

### 5.2. Препроцессинг и сплит

| ID | Требование | Приоритет |
|---|---|---|
| FR-02.1 | Удаление полных дубликатов; заполнение пропусков median (числовые) / mode (категориальные) | MUST |
| FR-02.2 | Автоматическое определение типов колонок: object→categorical, числовые с ≤15 уникальными → categorical, остальные числовые → continuous | MUST |
| FR-02.3 | Пользовательские overrides: `force_categorical`, `force_continuous`, `exclude` | MUST |
| FR-02.4 | Минимизация данных: удаление прямых идентификаторов; опционально — высоко-кардинальных колонок (по порогу) | MUST |
| FR-02.5 | Holdout split со стратификацией по `target_column` (если задан); фиксированный `random_state` | MUST |
| FR-02.6 | Атомарность: сплит выполняется один раз, train.csv и holdout.csv записываются атомарно | MUST |
| FR-02.7 | Профиль предобработанного датасета (`profile.json`): summary stats по каждой колонке | SHOULD |
| FR-02.8 | Holdout не виден генератору ни на каком этапе пайплайна | MUST |

### 5.3. Генерация (Synthesis Service, port 8002)

| ID | Требование | Приоритет |
|---|---|---|
| FR-03.1 | Поддержка пяти генераторов: `dpctgan`, `dptvae`, `ctgan`, `tvae`, `copulagan` | MUST |
| FR-03.2 | Асинхронный запуск джоба синтеза (`POST /jobs`) с возвратом 202 + `job_id` | MUST |
| FR-03.3 | Опрос статуса джоба (`GET /jobs/{job_id}`): `queued/running/done/failed/cancelled` | MUST |
| FR-03.4 | Отмена джоба (`DELETE /jobs/{job_id}`) с проверкой состояния перед `fit()` и перед `sample()` | MUST |
| FR-03.5 | Сохранение синтетики на shared volume по пути `synth/{job_id}/synthetic.csv` | MUST |
| FR-03.6 | Опциональное сохранение модели + JSON-сайдкара с DP-конфигом и расходом бюджета (`save_model=true`) | MUST |
| FR-03.7 | Повторное сэмплирование из сохранённой модели (`POST /models/{model_id}/sample`) без расхода DP-бюджета | MUST |
| FR-03.8 | DP-отчёт с полями: `is_dp_enabled`, `epsilon_initial`, `delta`, `spent_epsilon_final`, `epochs_completed`, `epochs_requested` | MUST |
| FR-03.9 | Прогресс-бар обучения через `tqdm` с отображением spent ε в реальном времени | SHOULD |
| FR-03.10 | Поддержка GPU (CUDA) с graceful fallback на CPU | MUST |
| FR-03.11 | Воспроизводимость: фиксация `random_seed` во всех слоях (numpy, torch, sklearn, ctgan, smartnoise) | MUST |
| FR-03.12 | Финализация артефактов: pending-файл (`synthetic_pending.csv`) переименовывается в final только после прохождения validation шага | SHOULD |

### 5.4. Оценка (Evaluation Service, port 8003)

#### 5.4.1. Полезность (Utility)

| ID | Требование | Приоритет |
|---|---|---|
| FR-04.1 | Покоменные статистические метрики: JSD (числовые), TVD (категориальные), summary mean_jsd / mean_tvd | MUST |
| FR-04.2 | Сравнение матриц корреляций: Pearson MAE для числовых, Cramér's V MAE для категориальных | MUST |
| FR-04.3 | TRTR / TSTR / Utility Loss на Random Forest (classification: F1+ROC-AUC; regression: MAE+R²) | MUST |
| FR-04.4 | `holdout` как единый test-set для честного сравнения TRTR vs TSTR | MUST |
| FR-04.5 | `drop_columns` для исключения тех. полей из ML-оценки | MUST |

#### 5.4.2. Приватность (Privacy)

| ID | Требование | Приоритет |
|---|---|---|
| FR-04.6 | Классические метрики: k-anonymity, l-diversity, t-closeness — раздел `diagnostic` | MUST |
| FR-04.7 | DCR (Distance to Closest Record) и NNDR (Nearest Neighbor Distance Ratio) с one-hot+MinMax кодированием | MUST |
| FR-04.8 | Distance-based proxy MIA: классификатор RandomForest различает train vs holdout по DCR-фичам | MUST |
| FR-04.9 | DP-гарантии переносятся из отчёта генератора в раздел `dp_guarantees` | MUST |
| FR-04.10 | Раздел `empirical_risk` (DCR/NNDR/MIA) отделён от `dp_guarantees` (формальные) и `diagnostic` (k/l/t) | MUST |
| FR-04.11 | Опциональное отключение групп метрик через флаги в конфиге | SHOULD |

### 5.5. Отчётность и вердикт (Reporting Service, port 8004)

| ID | Требование | Приоритет |
|---|---|---|
| FR-05.1 | Сборка единого JSON-отчёта из трёх sub-отчётов (utility / privacy / dp) + minimization | MUST |
| FR-05.2 | Автоматический вердикт PASS / FAIL / PARTIAL по настраиваемым порогам | MUST |
| FR-05.3 | Пороги: `max_utility_loss`, `max_mean_jsd`, `max_mia_auc`, `require_dcr_privacy_preserved`, `require_dp_enabled`, `max_spent_epsilon` | MUST |
| FR-05.4 | Список конкретных issues, по которым произошёл FAIL, в человекочитаемой форме | MUST |
| FR-05.5 | Сохранение отчёта в `reports/{dataset}__{generator}__{timestamp}.json` | MUST |
| FR-05.6 | Вердикт PARTIAL при отсутствующих sub-отчётах (например, утратили privacy_evaluator) | MUST |

### 5.6. Оркестрация (Gateway, port 8000)

| ID | Требование | Приоритет |
|---|---|---|
| FR-06.1 | Запуск пайплайна через `POST /api/v1/runs` с возвратом 202 + `run_id` | MUST |
| FR-06.2 | Состояние run хранится в Redis (TTL 1 час после завершения) | MUST |
| FR-06.3 | Полный список запусков из Redis + ProcessRegistry в PostgreSQL (`GET /runs`) | MUST |
| FR-06.4 | Активные запуски — только из Redis (`GET /runs/active`) | SHOULD |
| FR-06.5 | Детальная информация по запуску (`GET /runs/{id}`): включая `config_snapshot`, `error_message` | MUST |
| FR-06.6 | Скачивание артефактов: `/runs/{id}/synthetic` (CSV/JSON), `/runs/{id}/report` (JSON) | MUST |
| FR-06.7 | Лог-файл по run_id с фильтрацией по всем сервисам и сортировкой по timestamp (`/runs/{id}/logs`) | MUST |
| FR-06.8 | Отмена пайплайна (`DELETE /runs/{id}`) — корректно прерывает текущий джоб синтеза | MUST |
| FR-06.9 | Webhook POST по `webhook_url` при завершении (статус, вердикт, run_id) | SHOULD |
| FR-06.10 | Опциональный режим `quick_test=true` — сокращённый прогон для smoke-test | SHOULD |
| FR-06.11 | Повторная попытка генерации при FAIL (до `pipeline.max_iterations` раз) | SHOULD |
| FR-06.12 | Догенерация дополнительных строк из сохранённой модели запуска (`POST /runs/{id}/synthetic`) | SHOULD |

### 5.7. Управление конфигами

| ID | Требование | Приоритет |
|---|---|---|
| FR-07.1 | CRUD по конфигам через `/configs` (POST/GET/PUT/DELETE) | MUST |
| FR-07.2 | Валидация YAML при загрузке через Pydantic-схему `AppConfig` | MUST |
| FR-07.3 | Список конфигов с краткой информацией: `generator_type`, `epsilon`, `epochs` | MUST |
| FR-07.4 | Раздельный endpoint валидации (`POST /configs/validate`) для использования из UI | SHOULD |
| FR-07.5 | Семантические warnings (например, `preprocessor_eps > 30% от ε`) | SHOULD |

### 5.8. Управление моделями

| ID | Требование | Приоритет |
|---|---|---|
| FR-08.1 | Список моделей (`GET /models`) с метаданными из sidecar (`*.meta.json`) | MUST |
| FR-08.2 | Детали модели (`GET /models/{id}`): `dp_config`, `dp_spent`, `sample_size`, размер файла | MUST |
| FR-08.3 | Удаление модели (`DELETE /models/{id}`) с очисткой sidecar | MUST |
| FR-08.4 | Сэмплирование из модели (`POST /models/{id}/samples`) с CSV/JSON-выводом | MUST |

### 5.9. Системные ручки

| ID | Требование | Приоритет |
|---|---|---|
| FR-09.1 | Liveness probe `/api/v1/health` (без auth) — uptime и version | MUST |
| FR-09.2 | Readiness probe `/api/v1/health/db` — состояние PostgreSQL + latency | MUST |
| FR-09.3 | Readiness probe `/api/v1/health/gpu` — наличие CUDA, имя устройства, vram | SHOULD |
| FR-09.4 | Метрики в Prometheus exposition format `/api/v1/metrics`: количество запусков по статусу/вердикту, длина очереди, средняя длительность | MUST |

---

## 6. Нефункциональные требования (NFR)

### 6.1. Производительность

| ID | Требование |
|---|---|
| NFR-01 | Запуск пайплайна (`POST /runs`) возвращает 202 за < 500 мс на 95-м перцентиле |
| NFR-02 | Health-check (`/health`) — p95 < 50 мс |
| NFR-03 | Загрузка CSV размером 100 МБ — успешно до 30 секунд |
| NFR-04 | Полный пайплайн на датасете 50k строк × 15 колонок в режиме `quick_test`: < 5 минут |
| NFR-05 | Полный пайплайн на датасете 1М строк × 50 колонок в production-режиме (DP-CTGAN, 300 эпох, GPU): ≤ 4 часа |
| NFR-06 | Polling интервал статуса синтеза — 10 секунд (5 секунд в quick_test) |

### 6.2. Надёжность

| ID | Требование |
|---|---|
| NFR-07 | Все 7 контейнеров имеют healthcheck с retry-политикой; Docker Compose — `restart: on-failure` |
| NFR-08 | Падение одного из downstream-сервисов помечает run как `failed` с сохранённым `error_message`, не вешает Gateway |
| NFR-09 | Атомарность сплита и финализация синтетики через `pending` → `final` rename |
| NFR-10 | Optimistic locking при обновлении RunRecord в Redis (WATCH/MULTI/EXEC, до 3 retry) |
| NFR-11 | Graceful degradation БД: при `DB_DISABLED=true` Gateway работает без ProcessRegistry |

### 6.3. Безопасность (текущий уровень MVP)

| ID | Требование |
|---|---|
| NFR-12 | Bearer-token авторизация (`Authorization: Bearer <API_KEY>`); если `API_KEY` не задан — режим разработки |
| NFR-13 | DSN PostgreSQL передаётся через переменные окружения; никогда не сохраняется в YAML |
| NFR-14 | `.env` исключён из репозитория; коммитится только `.env.example` |
| NFR-15 | CORS открыт на dev (`allow_origins=["*"]`); в production требуется ограничение на whitelist |
| NFR-16 | Pickle-загрузка моделей — только из доверенных источников (внутренний `/data/models` volume) |

> **Замечание.** Текущая модель безопасности достаточна для академической демонстрации. См. раздел [12.4](#124-безопасность-и-комплаенс) — что нужно для production-уровня.

### 6.4. Воспроизводимость

| ID | Требование |
|---|---|
| NFR-17 | `random_seed` передаётся в numpy, torch, sklearn, smartnoise, ctgan |
| NFR-18 | `config_snapshot` сохраняется в RunRecord — позволяет повторно собрать тот же запуск |
| NFR-19 | Версии библиотек зафиксированы в `requirements.txt` каждого сервиса |
| NFR-20 | Образы помечены конкретными тегами (`postgres:16-alpine`, `redis:7-alpine`), не `latest` |

### 6.5. Наблюдаемость (текущий уровень MVP)

| ID | Требование |
|---|---|
| NFR-21 | Сквозной `run_id` логируется во всех 7 контейнерах через `ContextVar` |
| NFR-22 | Лог-файлы `logs/app.log` (Gateway) + `/data/logs/{service}.log` (микросервисы) |
| NFR-23 | Endpoint объединённой выборки логов по `run_id` (`/runs/{id}/logs`) |
| NFR-24 | Prometheus метрики через `/metrics` (Gateway): счётчики по `status`, `verdict`, длина очереди, средняя длительность |

### 6.6. Развёртывание

| ID | Требование |
|---|---|
| NFR-25 | Полный стек поднимается одной командой: `docker compose up -d --build` |
| NFR-26 | Первый билд (тяжёлый synthesis-образ ≈ 13 ГБ) — 10–20 минут на стандартной машине |
| NFR-27 | Каждый микросервис имеет независимый Dockerfile + requirements.txt |
| NFR-28 | Lazy-import ML-зависимостей в `config_loader.py` — Gateway-образ остаётся «лёгким» |
| NFR-29 | Shared volume `/data` смонтирован во все сервисы, кроме Redis и PostgreSQL |

### 6.7. Сопровождаемость

| ID | Требование |
|---|---|
| NFR-30 | Чёткое разделение ответственностей: `api/`, `services/*`, `synthesizer/`, `evaluator/`, `data_processor/`, `shared/` |
| NFR-31 | Pydantic v2-схемы как контракты между сервисами (`shared/schemas/`) |
| NFR-32 | `BaseGenerator`-интерфейс позволяет добавить новый генератор изменив только `synthesis_service/router.py` + новый файл в `synthesizer/` |
| NFR-33 | Unit-тесты для `config_loader`, `data_processor`, `reporter`; e2e-тест полного пайплайна (`@pytest.mark.e2e`) |

### 6.8. Совместимость

| ID | Требование |
|---|---|
| NFR-34 | Python 3.10+ во всех сервисах |
| NFR-35 | Поддерживаемая ОС хоста: Linux x86_64; macOS / Windows — через Docker Desktop |
| NFR-36 | GPU-поддержка через `nvidia-container-toolkit` для synthesis_service (CUDA 12.x) |
| NFR-37 | Совместимость с PostgreSQL 14+ |

---

## 7. Архитектура

### 7.1. Логическая диаграмма

```
                        ┌─────────────────────────────┐
                        │  Клиент (curl, Postman, UI) │
                        └──────────────┬──────────────┘
                                       │  HTTP + Bearer
                                       ▼
              ┌────────────────────────────────────────────┐
              │  Gateway (api/, port 8000) — оркестратор   │
              │  • RunStore (Redis)                        │
              │  • httpx-клиенты к четырём микросервисам   │
              │  • Webhook, /metrics, /logs                │
              └──────┬───────┬───────┬──────────┬──────────┘
                     │       │       │          │
            ┌────────▼─┐ ┌───▼────┐ ┌▼─────────┐ ┌▼─────────┐
            │ Data 8001│ │Synth   │ │Eval 8003 │ │Report8004│
            │ proces.  │ │ 8002   │ │privacy + │ │verdict + │
            │ +holdout │ │5 gens  │ │utility   │ │JSON      │
            │ +DSN imp.│ │+models │ │+ MIA     │ │          │
            └────────┬─┘ └───┬────┘ └────┬─────┘ └────┬─────┘
                     │       │           │            │
                     ▼       ▼           ▼            ▼
              ┌─────────────────────────────────────────┐
              │  Shared Docker volume /data             │
              │  datasets/  splits/  synth/  models/    │
              │  reports/   logs/                       │
              └─────────────────────────────────────────┘

         ┌───────────┐   ┌──────────────┐   ┌───────────────┐
         │ Redis     │   │ PostgreSQL   │   │ user_db       │
         │ (RunStore)│   │ (system,     │   │ (демо-БД      │
         │ TTL 1h    │   │  Process     │   │  пользова-    │
         │           │   │  Registry)   │   │  теля 5434)   │
         └───────────┘   └──────────────┘   └───────────────┘
              :6379          :5433
```

### 7.2. Перечень контейнеров

| Сервис | Порт | Роль |
|---|---|---|
| `app` (Gateway) | 8000 | HTTP-вход, оркестрация, RunStore, /metrics |
| `data_service` | 8001 | CSV/PostgreSQL импорт, preprocessing, holdout split |
| `synthesis_service` | 8002 | Обучение генераторов (DP-CTGAN, DP-TVAE, SDV-модели), сэмплирование |
| `evaluation_service` | 8003 | Privacy + Utility метрики |
| `reporting_service` | 8004 | Вердикт PASS/FAIL/PARTIAL + сохранение JSON-отчёта |
| `redis` | 6379 | RunStore (TTL 1 час) |
| `postgres` | 5433 | ProcessRegistry (история запусков) |
| `user_db` | 5434 | Демонстрационная БД пользователя для импорта/экспорта |

### 7.3. Поток выполнения пайплайна (`POST /runs`)

1. Gateway валидирует `config_name` → загружает YAML → парсит через Pydantic.
2. Gateway создаёт `RunRecord` в Redis (status: queued); запускает `_execute_pipeline` в `BackgroundTasks`.
3. **Step 1/7** — Data Service: `POST /datasets` (CSV) или `POST /datasets/from-db` (PostgreSQL) → `dataset_id`.
4. **Step 2/7** — Data Service: `POST /datasets/{id}/split` → preprocessing + minimization + stratified split → `split_id`, train.csv, holdout.csv, profile.json.
5. **Step 3/7** — Synthesis Service: `POST /jobs` с inline-конфигом генератора → `job_id`. RunRecord обновлён `current_job_id` для возможности отмены.
6. **Step 4/7** — Gateway polls `GET /jobs/{job_id}` каждые 10 секунд. По `done` → `dp_report`, `synth_path`, `model_id`. По `failed/cancelled` → run = failed.
7. **Step 5/7** — Evaluation Service: `POST /evaluate/privacy`.
8. **Step 6/7** — Evaluation Service: `POST /evaluate/utility`.
9. **Step 7/7** — Reporting Service: `POST /reports` → `verdict` + `report_path`.
10. Если `verdict=FAIL` и `iteration < max_iterations` → goto Step 3.
11. Финализация: переименование `synthetic_pending.csv` → `synthetic.csv`. Опциональный экспорт в PostgreSQL пользователя. Webhook (если задан).
12. RunRecord помечен `completed`, TTL 1 час.

### 7.4. Контракт между сервисами

* Все межсервисные вызовы — через httpx-клиент (`api/clients.py`).
* Контракты — Pydantic v2 модели в `shared/schemas/`: `DatasetMeta`, `SplitMeta`, `SplitRequest`, `SynthesisJobCreate/Summary`, `SampleRequest`, `PrivacyEvalRequest`, `UtilityEvalRequest`, `ReportRequest`, `ReportResponse`.
* Конфиг генератора передаётся **inline** в теле `POST /jobs` (synthesis_service не читает YAML с диска) — это развязывает добавление нового генератора от перебилда Gateway.

### 7.5. Структура данных в `/data`

```
/data/
├── datasets/{dataset_id}/
│   ├── raw.csv
│   └── meta.json
├── splits/{split_id}/
│   ├── train.csv
│   ├── holdout.csv
│   ├── profile.json
│   └── meta.json
├── synth/{job_id}/
│   └── synthetic.csv          # после rename из synthetic_pending.csv
├── models/
│   ├── {model_id}.pkl
│   └── {model_id}.meta.json
├── reports/
│   └── {dataset}__{generator}__{ts}.json
└── logs/
    ├── data_service.log
    ├── synthesis_service.log
    ├── evaluation_service.log
    └── reporting_service.log
```

### 7.6. Ключевые архитектурные решения

* **Lazy ML imports.** `config_loader.py` не импортирует `torch / opacus / smartnoise` напрямую — Gateway-образ остаётся слим (~150 МБ), тяжёлые зависимости (≈ 13 ГБ) изолированы в `synthesis_service`.
* **Inline-конфиг генератора.** Synthesis_service не имеет доступа к директории `/configs` — Gateway сериализует `GeneratorYamlConfig` и шлёт его телом запроса. Добавление нового генератора = новый файл в `synthesizer/` + ветка в `synthesis_service/router.py:_build_generator`.
* **In-memory job_store.** `synthesis_service/job_store.py` хранит джобы в памяти процесса. Перезапуск контейнера = потеря активных джобов. **Подлежит замене на Redis/Postgres в production** (см. раздел [12.1](#121-устойчивое-состояние-и-горизонтальное-масштабирование)).
* **Single-worker enforced.** Все микросервисы стартуют с `--workers 1`. Shared volume `/data` не поддерживает конкурентную запись.
* **ε-budget семантика.** DP-бюджет тратится **только в `fit()`**. Повторное сэмплирование через `POST /models/{id}/samples` бюджет не расходует — это согласуется с post-processing immunity DP.
* **Sidecar metadata.** Каждая сохранённая модель имеет `*.meta.json` рядом с `*.pkl`. Gateway читает метаданные без `pickle.load`, что развязывает его от наличия классов синтезаторов.

---

## 8. Контракты API и модели данных

### 8.1. Gateway — публичный API

Префикс: `/api/v1`. Все эндпоинты, кроме `/health`, требуют `Authorization: Bearer <API_KEY>`, если `API_KEY` задан.

#### Запуски (`/runs`)

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/runs` | Создание запуска (тело: `RunCreate`) → 202 + `RunSummary` |
| GET | `/runs` | Список Redis + PostgreSQL, фильтры: `status`, `verdict`, `dataset_name`; пагинация |
| GET | `/runs/active` | Только Redis (активные и в TTL) |
| GET | `/runs/{id}` | Детали (`RunDetail`) с `config_snapshot`, `error_message` |
| DELETE | `/runs/{id}` | Отмена / удаление |
| GET | `/runs/{id}/report` | JSON-отчёт |
| GET | `/runs/{id}/synthetic?format=csv\|json` | Скачать синтетику |
| POST | `/runs/{id}/synthetic` | Догенерация N строк из сохранённой модели запуска |
| GET | `/runs/{id}/logs?tail=N` | Объединённый лог-файл по run_id |

#### Датасеты (`/datasets`)

POST/GET/DELETE/PREVIEW + `/schema`.

#### Модели (`/models`)

GET (list/detail), DELETE, POST `/{id}/samples`.

#### Конфиги (`/configs`)

POST/GET/PUT/DELETE + `POST /validate` для UI.

#### Системные

`GET /health`, `GET /health/db`, `GET /health/gpu`, `GET /metrics` (Prometheus).

### 8.2. Internal API микросервисов

* **Data Service** (8001): `/api/v1/datasets`, `/api/v1/datasets/from-db`, `/api/v1/datasets/{id}/split`, `/api/v1/datasets/{id}/splits/{split_id}/{train|holdout|profile}`.
* **Synthesis Service** (8002): `/api/v1/jobs`, `/api/v1/jobs/{id}`, `/api/v1/jobs/{id}/dp_report`, `DELETE /api/v1/jobs/{id}`, `/api/v1/models/{id}/sample`.
* **Evaluation Service** (8003): `/api/v1/evaluate/privacy`, `/api/v1/evaluate/utility`.
* **Reporting Service** (8004): `/api/v1/reports`.

### 8.3. Формат ошибок

```json
{
  "code": "NOT_FOUND | VALIDATION_ERROR | UNAUTHORIZED | INTERNAL_ERROR | CONFLICT | DB_ERROR | SAMPLE_ERROR | RUN_NOT_FINISHED | MODEL_NOT_SAVED",
  "message": "человекочитаемое описание"
}
```

### 8.4. Жизненный цикл RunRecord

```
queued → running → completed
               ↘  failed
               ↘  cancelled
```

Поля: `run_id`, `dataset_name`, `config_name`, `status`, `verdict`, `save_model`, `webhook_url`, `n_synth_rows`, `current_job_id`, `model_id`, `synth_rows`, `synth_path`, `report_path`, `report`, `config_snapshot`, `error_message`, `created_at`, `started_at`, `finished_at`.

---

## 9. Конфигурация пайплайна

YAML в `final_system/configs/` валидируется `config_loader.py` (Pydantic v2). Главные секции:

### 9.1. `paths`

`logs`, `output_dir` — относительно `final_system/`.

### 9.2. `data_import`

```yaml
data_import:
  type: csv | postgres
  path: data/adult.csv          # для csv
  dsn_env: DB_IMPORT_DSN        # для postgres
  query: "SELECT * FROM table"  # для postgres
```

### 9.3. `data_export`

```yaml
data_export:
  type: none | csv | postgres
  dsn_env: DB_EXPORT_DSN
  table: synthetic_adult
  if_exists: replace | append | fail
```

### 9.4. `pipeline`

```yaml
pipeline:
  dataset_name: adult_census
  sample_size: 0          # 0 = все строки
  n_synth_rows: 0         # 0 = размер train
  holdout_size: 0.2       # ∈ (0, 1)
  random_state: 42
  max_iterations: 1       # повторных обучений при FAIL
```

### 9.5. `generator`

```yaml
generator:
  generator_type: dpctgan | dptvae | ctgan | tvae | copulagan
  epsilon: 3.0            # > 0
  preprocessor_eps: 0.5   # ∈ [0, epsilon)
  delta: null             # null = автонастройка 1/(n·√n)
  sigma: 5.0              # noise multiplier для DP-SGD
  max_per_sample_grad_norm: 1.0
  epochs: 300
  batch_size: 500
  embedding_dim: 128
  generator_dim: [256, 256]
  discriminator_dim: [256, 256]
  generator_lr: 0.0002
  discriminator_lr: 0.0002
  cuda: true
  random_seed: 42
  disabled_dp: false      # true = baseline без DP
```

#### Рекомендуемые диапазоны ε по отрасли:

| Отрасль | ε | Профиль риска |
|---|---|---|
| Финансы / здравоохранение | 1.0–2.0 | строгий регулятор, высокий риск |
| Общий enterprise | 2.0–5.0 | стандартный профиль |
| R&D / аналитика | 5.0–10.0 | качество важнее приватности |

### 9.6. `data_schema`

```yaml
data_schema:
  categorical: [...]
  continuous: [...]
  exclude: [fnlwgt]
  direct_identifiers: []           # удаляются до синтеза
  drop_high_cardinality: false
  cardinality_threshold: 0.9
```

Если `categorical` и `continuous` пусты — автодетекция через `DataProcessor.detect_column_types()`.

### 9.7. `utility`

```yaml
utility:
  target_column: income
  task_type: classification | regression
  drop_columns: [fnlwgt]
  n_estimators: 100
```

### 9.8. `privacy`

```yaml
privacy:
  quasi_identifiers: [age, education, occupation, sex, race]
  sensitive_attribute: income
  compute_classical: true
  compute_distance: true
  compute_mia: true
  distance_sample_size: 2000
  mia_sample_size: 1000
```

### 9.9. `thresholds`

```yaml
thresholds:
  max_utility_loss: 0.15
  max_mean_jsd: 0.20
  max_mia_auc: 0.55
  require_dcr_privacy_preserved: true
  require_dp_enabled: true
  max_spent_epsilon: 5.0
```

---

## 10. Метрики качества и приватности

### 10.1. Полезность (Utility)

| Метрика | Тип колонок | Диапазон | Желаемое |
|---|---|---|---|
| **JSD** (Jensen-Shannon Divergence) | числовые (binned) | [0, 1] | < 0.20 |
| **TVD** (Total Variation Distance) | категориальные | [0, 1] | < 0.20 |
| **Pearson MAE** | числовые корреляции | [0, 2] | < 0.10 |
| **Cramér's V MAE** | категориальные ассоциации | [0, 1] | < 0.15 |
| **TRTR F1 / R²** | ML на real_train→test_real | задача-зависимо | baseline (потолок) |
| **TSTR F1 / R²** | ML на synth→test_real | задача-зависимо | ≥ TRTR − 15% |
| **Utility Loss** | TRTR − TSTR | задача-зависимо | < 0.15 |

### 10.2. Приватность (Privacy)

#### Формальные DP-гарантии (`dp_guarantees`)

* `is_dp_enabled` — DP включён в обучении.
* `epsilon_initial` — заданный бюджет.
* `delta` — δ-параметр.
* `spent_epsilon_final` — фактически потраченный ε после `epochs_completed` эпох.
* `epochs_completed` / `epochs_requested` — реально / запрошено.

#### Эмпирический риск (`empirical_risk`)

* **DCR** (Distance to Closest Record) — `privacy_preserved=true`, если DCR_synth ≥ DCR_holdout.
* **NNDR** (Nearest Neighbor Distance Ratio) — `synth_mean ≥ 0.5`, `share_below_0.1 < 0.05`.
* **MIA proxy** (Membership Inference Attack) — RandomForest классификатор различает train vs holdout по DCR-фичам. `attack_auc ≤ 0.55` — атака не лучше случайного угадывания.

#### Диагностика (`diagnostic`, классические метрики анонимности)

* **k-anonymity** ≥ 5 (минимальный размер группы по QI).
* **l-diversity** ≥ 2 (минимум разных значений SA в группе).
* **t-closeness** ≤ 0.2 (близость распределения SA в группе к глобальному).

> Классические метрики k/l/t — **диагностические**, не замена DP-гарантиям.

### 10.3. Логика вердикта

```
PASS    — все активные проверки (utility_ok, privacy_ok, dp_ok) пройдены
FAIL    — хотя бы одна проверка провалена; в issues — список конкретных причин
PARTIAL — отсутствует хотя бы один из sub-отчётов (utility/privacy/dp)
```

---

## 11. Критерии приёмки и тестирование

### 11.1. Уровни тестирования

| Уровень | Что проверяет | Запуск | Marker |
|---|---|---|---|
| **Unit** | `config_loader`, `data_processor`, `reporter` | `pytest -m "not e2e"` | — |
| **Component** | один сервис в изоляции (TestClient FastAPI) | планируется | `@pytest.mark.component` (пока заглушка) |
| **E2E (smoke)** | полный пайплайн через docker-compose | `pytest -m e2e` | `@pytest.mark.e2e` |

### 11.2. Существующие тесты

* `test_config_loader.py` — валидация YAML, edge cases.
* `test_processor_core.py` — preprocessing.
* `test_processor_minimize.py` — минимизация (direct identifiers, high cardinality).
* `test_reporter_verdict.py` — логика PASS/FAIL/PARTIAL по порогам.
* `test_e2e_microservices.py` — полный CTGAN-прогон через docker-compose, проверка `verdict ∈ {PASS, FAIL, PARTIAL}`, `synth_rows > 0`, `config_snapshot is dict`.

### 11.3. Acceptance criteria для релиза MVP

1. `docker compose up -d --build` поднимает все 7 контейнеров, healthcheck'и зелёные за < 3 минут.
2. `pytest final_system/tests/ -v` проходит полностью (unit + e2e).
3. `POST /api/v1/runs` с `e2e_ctgan` завершается со статусом `completed` (вердикт любой).
4. `POST /api/v1/runs` с `adult.yaml` (DP-CTGAN, 300 эпох) на полном Adult-датасете завершается с **PASS**: `mean_jsd < 0.20`, `utility_loss < 0.15`, `MIA AUC < 0.55`, `spent_ε ≤ 5.0`, `is_dp_enabled = true`.
5. `GET /metrics` возвращает Prometheus-совместимый формат с непустыми счётчиками.
6. Авторизация: запрос без `Authorization` при заданном `API_KEY` → 401.

---

## 12. Дальнейшие доработки до уровня корпоративного MLOps

> **Контекст.** Текущая Система — методологически и архитектурно корректный прототип, но при внедрении в MLOps-процессы крупной компании (банк, телеком, ритейл с тысячами data scientist'ов и десятками тысяч моделей) обнаруживается ряд гэпов. Этот раздел исчерпывающе описывает **что именно** нужно доработать, **почему** это критично для enterprise, и **какой shape** должно принять решение.

Доработки сгруппированы по доменам и приоритезированы:
🔴 **P0 (блокирующие)** — без них продакшн-внедрение невозможно;
🟡 **P1 (высокий)** — без них продукт не масштабируется;
🟢 **P2 (средний)** — повышают качество эксплуатации, но не блокируют запуск.

---

### 12.1. Устойчивое состояние и горизонтальное масштабирование

#### 12.1.1. 🔴 P0 — Persistent Job Queue для синтеза

**Проблема.** `synthesis_service/job_store.py` — in-memory dict с threading.Lock. Перезапуск контейнера / OOM kill = потеря всех активных джобов. Деплой новой версии = потеря очереди. Невозможно пустить несколько реплик: каждая держит свой стейт.

**Решение.**
* Перенести джобы в **Redis Streams** или **PostgreSQL** (с advisory locks); либо использовать готовый менеджер очередей: **Celery + Redis/RabbitMQ**, **Dramatiq**, **Temporal.io**, **Apache Airflow** (для оркестрации больших пайплайнов), **Argo Workflows** (Kubernetes-native).
* Для длинных GPU-задач — выделенный пул воркеров с приоритетами, ack/nack семантикой, retry-policy с exponential backoff.
* Хранить артефакты пайплайна (intermediate datasets, models) в внешних object storage (см. 12.1.3), чтобы воркер был stateless.

#### 12.1.2. 🔴 P0 — Несколько worker'ов и шардирование

**Проблема.** Все микросервисы запускаются с `--workers 1`. Shared volume `/data` (Docker bind-mount) не поддерживает конкурентную запись — гонки на rename `pending → final`. Невозможен horizontal scale.

**Решение.**
* Заменить `/data` на **сетевое объектное хранилище** (см. 12.1.3) — устраняет конкурентность writes.
* Включить multi-worker через `uvicorn --workers N` или gunicorn с uvicorn-workers.
* Для Synthesis Service — **per-GPU worker pool** (1 процесс ⇒ 1 GPU); при наличии нескольких GPU использовать GPU-aware scheduler (NVIDIA MIG, MPS, или K8s + nvidia-device-plugin).

#### 12.1.3. 🔴 P0 — Object storage вместо Docker volume

**Проблема.** `/data` — локальный bind-mount. Не работает при multi-host deploy, нет versioning, нет lifecycle policies, нет шифрования at rest by default.

**Решение.**
* **S3-compatible storage**: AWS S3 / MinIO / Ceph RGW / Yandex Object Storage / VK Cloud / Selectel.
* Бакет-структура: `s3://dp-synth/{tenant}/{environment}/{datasets|splits|synth|models|reports}/...`.
* Server-side encryption (SSE-KMS).
* Object versioning + lifecycle: модели — Glacier через 90 дней, синтетика — удаление через 30.
* Pre-signed URL для скачивания клиентами.
* Использовать s3fs / boto3 / minio-py в коде; абстрагировать через интерфейс `Storage` (Strategy pattern).

#### 12.1.4. 🟡 P1 — Kubernetes-deploy вместо Docker Compose

**Проблема.** Docker Compose — **dev-only**. Ни автомасштабирования, ни rolling updates, ни node draining, ни service mesh.

**Решение.**
* Helm-чарты для всех 5 микросервисов.
* HPA (Horizontal Pod Autoscaler) на CPU/RAM/custom metrics (длина очереди в Redis Streams).
* Для Synthesis Service — отдельный node pool с GPU + tolerations/taints + nvidia-device-plugin.
* PodDisruptionBudget для стабильности при rollout.
* NetworkPolicies для изоляции сервисов.
* Init-containers для миграций БД.
* Namespace-per-tenant или namespace-per-environment.

#### 12.1.5. 🟡 P1 — Multi-tenancy

**Проблема.** Сейчас все запуски шарят один Redis, одну БД, один volume. При внедрении в крупную компанию десятки команд пересекутся: команда A видит модели команды B; квоты не разделены; биллинг невозможен.

**Решение.**
* `tenant_id` как первоклассный параметр в API, RunRecord, путях артефактов, метриках.
* Изоляция по namespace в K8s, по schema в PostgreSQL, по prefix в S3.
* Per-tenant квоты: количество одновременных runs, размер хранимых моделей, ε-budget pool.
* RBAC роли в разрезе tenant: viewer / data-scientist / admin / dpo.

---

### 12.2. Управление моделями и lineage (Model Registry / Data Catalog)

#### 12.2.1. 🔴 P0 — Полноценный Model Registry

**Проблема.** Текущее «model storage» — `*.pkl` + `*.meta.json` на диске. Нет версионирования, нет stage-управления (staging/prod/archived), нет linking с обучающими данными, нет дедупликации.

**Решение.**
* Интеграция с **MLflow Model Registry** или **Vertex AI Model Registry** или **SageMaker Model Registry**, или собственный реестр на базе PostgreSQL + S3.
* Каждая модель имеет: `model_id`, `version`, `stage` (Staging/Production/Archived), `created_by`, `tags`, `signature` (input/output schema), `dp_config`, `dp_spent`, **hash обучающего split'а**, `source_run_id`.
* Promotion workflow: Staging → Production через approval gate (review + comments).

#### 12.2.2. 🔴 P0 — Data Lineage и Reproducibility

**Проблема.** Сейчас связь «датасет → split → модель → синтетика → отчёт» хранится только в RunRecord (Redis, TTL 1 час) и `reports/*.json`. Нет графа lineage, нет tracking version контролируемых артефактов.

**Решение.**
* **OpenLineage / Marquez** для генерации событий lineage из всех 7 сервисов.
* Каждый артефакт имеет:
  * **content hash** (SHA-256 файла);
  * **logical version** (semver или monotonic counter);
  * ссылку на родительский артефакт.
* Возможность ответить на вопросы: «какие модели обучены на этом split'е?», «какие синтетические датасеты используют эту модель?», «какие отчёты содержат вердикт PASS для конкретного бизнес-кейса?».

#### 12.2.3. 🟡 P1 — Data Versioning

**Проблема.** `raw.csv` затирается при перезагрузке датасета. Нет версионирования.

**Решение.**
* **DVC**, **LakeFS**, **Pachyderm** или versioned S3.
* `dataset_id` фиксирует логическую сущность; `version_id` — конкретный snapshot контента.
* Diff между версиями: новые/удалённые колонки, изменения в распределениях (data drift report).

#### 12.2.4. 🟡 P1 — Experiment Tracking

**Проблема.** Каждый run — отдельный JSON-отчёт. Нет UI для сравнения 50 экспериментов между собой, нет hyperparameter sweep'ов.

**Решение.**
* **MLflow Tracking** / **Weights & Biases** / **Neptune.ai** / **Comet** / самописная UI на FastAPI + React.
* Логирование hyperparams, metrics, artifacts, plots в каждый run.
* Comparison view: side-by-side `dpctgan ε=1.0 vs ε=3.0 vs ε=10.0 vs CTGAN-baseline`.
* Hyperparameter sweep: автоматическое формирование cartesian / random / Bayesian search над YAML-конфигом.

---

### 12.3. Наблюдаемость production-уровня

#### 12.3.1. 🔴 P0 — Структурированное логгирование

**Проблема.** Логи — plaintext с `[timestamp] [run X] [LEVEL] message`. Не парсятся как JSON, не индексируются в Elasticsearch / Loki.

**Решение.**
* JSON-структурированные логи через `structlog` или `python-json-logger`.
* Поля: `timestamp` (RFC3339), `level`, `service`, `run_id`, `tenant_id`, `trace_id`, `span_id`, `message`, custom contextual fields.
* Корреляция через **OpenTelemetry**: единый `trace_id` пересекает все 7 контейнеров.

#### 12.3.2. 🔴 P0 — Distributed Tracing

**Проблема.** `run_id` есть, но нет visualization вызовов между сервисами с временами и dependency graph.

**Решение.**
* **OpenTelemetry SDK** во всех сервисах + auto-instrumentation FastAPI/httpx.
* Backend: **Jaeger** / **Tempo** / **Zipkin** / **Datadog APM**.
* Span на каждый этап: `data.upload`, `data.split`, `synthesis.fit`, `synthesis.sample`, `eval.privacy`, `eval.utility`, `report.build`.
* Атрибуты span: `dataset_id`, `split_id`, `job_id`, `model_id`, `epoch_count`, `n_rows`.

#### 12.3.3. 🔴 P0 — Метрики Production-уровня

**Проблема.** Текущие `/metrics` — единственный endpoint только на Gateway, перечисление статусов out-of-the-box. Нет:
* RED-метрик (Rate, Errors, Duration) на каждом эндпоинте;
* USE-метрик (Utilization, Saturation, Errors) на инфраструктуре;
* GPU метрик (utilization, memory, temperature);
* business-метрик (DP-ε spent per tenant, models in production, vertex sample rate).

**Решение.**
* `prometheus-fastapi-instrumentator` на каждый сервис.
* Custom Prometheus Counter/Histogram/Gauge:
  * `dp_synth_runs_total{tenant, generator, verdict, status}`
  * `dp_synth_run_duration_seconds_bucket{tenant, generator, stage}`
  * `dp_synth_epsilon_spent_total{tenant, generator}`
  * `dp_synth_jobs_inflight{tenant}`
  * `dp_synth_gpu_utilization` (через NVIDIA DCGM exporter)
  * `dp_synth_storage_bytes{tenant, kind}`
* Grafana dashboards: Service Overview, GPU Health, Privacy Budget, Error Rate.
* Alertmanager: алерты на error rate > 5%, длина очереди > 100, deviation от baseline duration.

#### 12.3.4. 🟡 P1 — Audit Log

**Проблема.** Нет неизменяемого audit-трейла: «кто запустил run», «кто скачал модель», «кто менял пороги вердикта».

**Решение.**
* Append-only audit-таблица в PostgreSQL или S3 Object Lock.
* События: `run.created`, `run.cancelled`, `model.downloaded`, `model.deleted`, `config.modified`, `tenant.quota.changed`, `auth.login.success/failure`.
* Поля: `event_id`, `timestamp`, `actor (user/service)`, `tenant_id`, `resource_type`, `resource_id`, `action`, `outcome`, `client_ip`, `user_agent`.
* WORM (Write-Once Read-Many) сторадж для compliance (152-ФЗ, GDPR Art. 30).

#### 12.3.5. 🟢 P2 — SLO / SLI / Error Budget

* Определить SLI: `availability`, `p95 latency POST /runs`, `pipeline success rate`.
* SLO: 99.5% availability; 99% pipeline-ов завершаются с verdict (не с failed); p95 < 500 мс на API.
* Error budget burn-rate alerts.

---

### 12.4. Безопасность и комплаенс

#### 12.4.1. 🔴 P0 — Корпоративная аутентификация

**Проблема.** Single shared `API_KEY` — никакой identity, никаких ролей, никаких audit'ов «кто что делал».

**Решение.**
* **OIDC / SAML 2.0** интеграция с корпоративным IDP (Keycloak, Okta, Azure AD, Active Directory, Yandex 360, Astra Linux Aldpro).
* JWT-токены с short TTL + refresh tokens.
* Service-to-service auth через mTLS или OAuth2 client credentials.
* MFA для административных действий.

#### 12.4.2. 🔴 P0 — RBAC / ABAC

**Проблема.** Нет ролей. Любой обладатель API_KEY может:
* запустить генерацию на любом датасете,
* скачать любую модель,
* менять пороги вердикта.

**Решение.**
* Роли:
  * **Viewer** — read-only metadata, без скачивания артефактов;
  * **Data Scientist** — создаёт runs в рамках своего tenant'а; скачивает синтетику и модели своих runs;
  * **DPO / Compliance Officer** — read-only на все tenants; верифицирует verdict; экспорт audit-логов;
  * **Tenant Admin** — управляет квотами, конфигами, моделями своего tenant'а;
  * **Platform Admin** — supertenant; управление платформой.
* ABAC-атрибуты: `dataset.classification ∈ {public, internal, confidential, secret}` × `user.clearance`.
* OPA (Open Policy Agent) или Casbin для проверки правил.

#### 12.4.3. 🔴 P0 — Secret Management

**Проблема.** DSN PostgreSQL — в env-переменных, видны в `docker inspect`, не ротируются. Pickle-загрузка моделей небезопасна (RCE-риск при доверии untrusted source).

**Решение.**
* **HashiCorp Vault** / **AWS Secrets Manager** / **Azure Key Vault** / **Yandex Lockbox**.
* Динамические credentials к БД с TTL.
* SOPS/Sealed Secrets для git-управляемых конфигов.
* Замена pickle на безопасные форматы: **safetensors** (для весов), **ONNX** (для inference), JSON для метаданных. Если pickle необходим — подпись HMAC + проверка перед `pickle.load`.

#### 12.4.4. 🔴 P0 — Шифрование данных

**Проблема.** Данные в `/data` — plaintext. PostgreSQL без TLS. Сетевая коммуникация Gateway↔Microservices — HTTP, не HTTPS.

**Решение.**
* **At rest**: SSE-KMS для S3, encrypted EBS, pgcrypto / TDE для PostgreSQL.
* **In transit**: TLS 1.3 повсеместно. Service mesh (**Istio** / **Linkerd**) для автоматического mTLS между микросервисами.
* **Application-level encryption** для PII-полей при необходимости.

#### 12.4.5. 🟡 P1 — Privacy Budget Accounting (Per-Tenant)

**Проблема.** Каждая модель тратит ε независимо. Нет «годового бюджета приватности» на tenant. Невозможно ограничить total exposure: 100 моделей × ε=3.0 = ε=300 фактически (хотя formally они на разных датасетах).

**Решение.**
* Сущность **PrivacyBudgetPool** в БД: `tenant_id`, `dataset_id`, `total_budget`, `spent_budget`, `period (annual / quarterly)`, `reset_at`.
* Перед каждым `POST /runs` — проверка бюджета.
* Бюджет уменьшается на `epsilon_initial` (даже если модель уйдёт в FAIL).
* Алерты при достижении 80% бюджета.
* Reporting для DPO: расход бюджета по командам / периодам.

#### 12.4.6. 🟡 P1 — Data Classification and Tagging

**Проблема.** Все датасеты обрабатываются одинаково. Нет различия между «общедоступные данные» и «банковской тайной».

**Решение.**
* Tagging датасетов при upload: `classification`, `data_owner`, `business_unit`, `retention_policy`, `pii_columns`.
* Автоматическая PII-detection (Presidio / Comprehend / самописные правила) — алерт при загрузке, если в датасете есть номера паспортов, СНИЛС, телефоны без маскирования.
* Применение разных дефолтных порогов в зависимости от classification:
  * `secret` → ε ≤ 1.0, MIA AUC ≤ 0.52;
  * `confidential` → ε ≤ 3.0, MIA AUC ≤ 0.55;
  * `internal` → ε ≤ 10.0, MIA AUC ≤ 0.60.

#### 12.4.7. 🟡 P1 — Соответствие 152-ФЗ / GDPR

* Право на удаление (Art. 17 GDPR / 152-ФЗ ст. 21): rfm-операции на all artifacts по `data_subject_id`.
* DPIA (Data Protection Impact Assessment) — генерация автоматически при создании tenant.
* Договоры с под-обработчиками (Art. 28).
* Cross-border transfer controls.
* Сертификация по ГОСТ Р ИСО/МЭК 27001 / SOC 2 Type II.

#### 12.4.8. 🟢 P2 — Static / Dynamic Application Security Testing

* SAST: bandit, semgrep, codeql в CI.
* DAST: OWASP ZAP, Burp.
* Dependency scanning: Trivy, Snyk, Dependabot.
* Container scanning: Trivy, Clair.
* SBOM-генерация на каждый build (CycloneDX / SPDX).

---

### 12.5. CI/CD и DevEx

#### 12.5.1. 🔴 P0 — Полноценный CI/CD pipeline

**Проблема.** Сейчас в репо только `pytest -m e2e` для smoke. Нет автоматизации: build → test → security-scan → publish → deploy.

**Решение.**
* GitHub Actions / GitLab CI / Argo CD:
  1. Pre-commit hooks: black, isort, ruff, mypy, gitleaks.
  2. Unit + component tests на каждый PR.
  3. Build образов с тегом `git-sha` + `branch` + `latest`.
  4. Security scans (Trivy, bandit).
  5. Push в registry (Docker Hub / ECR / GCR / Yandex Container Registry).
  6. Deploy в staging через Argo CD / Flux.
  7. E2E тесты в staging.
  8. Manual approval → production.
  9. Canary / blue-green deployment.
  10. Rollback on metrics regression.

#### 12.5.2. 🟡 P1 — Infrastructure as Code

* Terraform / Pulumi / OpenTofu для инфраструктуры (VPC, EKS/GKE/AKS, S3, RDS, KMS).
* Helm-чарты + Kustomize overlays для приложений.
* Environment-as-code: dev / staging / prod через различия в overlays.

#### 12.5.3. 🟡 P1 — Schema migrations

**Проблема.** Текущий `db/init_db.sql` запускается один раз при init контейнера. Нет миграций.

**Решение.**
* **Alembic** для PostgreSQL.
* Init container выполняет `alembic upgrade head` перед стартом сервиса.
* Backward-compatible миграции (additive only) для zero-downtime deploy.

#### 12.5.4. 🟢 P2 — Developer SDK / CLI

* Python SDK с type-hints для интеграции из ML-блокнотов.
* CLI на базе Click/Typer: `dpsynth runs create --config adult.yaml`, `dpsynth models list --tenant team-a`.
* Jupyter-magics: `%dpsynth run adult.yaml`.
* Контейнер dev-environment (devcontainer.json) для VSCode.

---

### 12.6. Производительность и масштабирование ML-нагрузок

#### 12.6.1. 🔴 P0 — Оптимизация GPU-утилизации

**Проблема.** Один джоб = монопольный захват одной GPU. T4 / A10 простаивают на маленьких датасетах.

**Решение.**
* Batch-job scheduling: несколько мелких джобов на одной GPU через MPS (Multi-Process Service).
* MIG (Multi-Instance GPU) на A100 для жёсткой изоляции.
* GPU sharing через Kubernetes nvidia-device-plugin + time-slicing.
* Prediction API на CPU для inference / sampling из обученной модели (sample не требует GPU).

#### 12.6.2. 🟡 P1 — Распределённое обучение

**Проблема.** Текущие генераторы — single-GPU, single-node. На датасетах > 100M строк — узкое горло.

**Решение.**
* Data Parallel через DDP (PyTorch). Совместимо с Opacus через `make_private(distributed=True)`.
* Для DP-CTGAN — patches к smartnoise-synth для DDP (значительная работа, требует upstream-контрибуции).
* Альтернативные DP-frameworks: **TensorFlow Privacy**, **Google's tf-privacy**, **JAX-Privacy** (DeepMind) — проще распараллеливаются.

#### 12.6.3. 🟡 P1 — Caching и инкрементальные пересчёты

* Кэшировать preprocessing-результаты по hash датасета — не пересчитывать holdout split при том же random_state.
* Кэшировать промежуточные ml_efficacy результаты.
* Кэш на Redis с инвалидацией по hash входов.

#### 12.6.4. 🟢 P2 — Мониторинг качества модели на inference (Drift Detection)

**Проблема.** Сохранённая модель сэмплируется через год. Распределение реальных данных изменилось — синтетика устарела.

**Решение.**
* **Evidently AI** / **Whylabs** / самописная стат-проверка на свежесть модели.
* Алерт «model drift detected» при KS-test > threshold между реальными данными и сэмплом из модели.
* Auto-retraining trigger при drift > critical level.

---

### 12.7. UX и интерфейсы

#### 12.7.1. 🟡 P1 — Web UI

**Проблема.** Сейчас взаимодействие — только через Swagger / curl. Не подходит для compliance officer / BI-аналитика.

**Решение.**
* SPA на React/Vue: dashboards, run history, side-by-side comparison отчётов, model registry, config editor с live-валидацией, Grafana embed для метрик.
* **Streamlit** / **Gradio** для быстрых PoC.

#### 12.7.2. 🟡 P1 — Notebook integration

* Jupyter-плагин: «загрузить датасет с моего notebook → запустить пайплайн → получить synth_df назад в DataFrame».
* Магики: `%%dpsynth fit ...`, `%%dpsynth sample 10000`.

#### 12.7.3. 🟢 P2 — Конфигурация через UI вместо YAML

* Мастер: выбор датасета → автодетекция схемы → выбор target → выбор generator/ε по профилю риска → preview порогов.
* Сохранение как YAML с возможностью экспорта в git.

---

### 12.8. Расширение функциональности

#### 12.8.1. 🟡 P1 — Multi-table relational synthesis

**Проблема.** Реальные корпоративные данные — нормализованные таблицы с FK-связями. Текущая Система генерирует одну таблицу за раз.

**Решение.**
* Интеграция с **SDV's HMA1 / SDV-multi-table** или **Synthcity**.
* DSL для описания схемы реляционной БД.
* Сохранение referential integrity между сгенерированными таблицами.

#### 12.8.2. 🟡 P1 — Time-series synthesis

* Поддержка генераторов для последовательных данных: **DoppelGANger**, **TimeGAN**, **PAR (SDV)**.
* DP-аналоги (DP-DoppelGANger).

#### 12.8.3. 🟢 P2 — AutoML для DP-генерации

* Auto-tuning ε по бюджету: «дай мне максимальное качество при ε ≤ 3.0».
* Bayesian optimization над `epochs / batch_size / sigma / preprocessor_eps`.
* Стратегия: probe-runs с малыми эпохами → extrapolation на полный запуск.

#### 12.8.4. 🟢 P2 — Расширенная privacy-evaluation

* **Полноценная shadow-model MIA** (Shokri et al.) вместо distance-proxy.
* **Attribute Inference Attack** simulation.
* **Reconstruction attacks** (генерация реальных записей из модели).
* Сертифицированные приватность-аудиторы — генерация report'ов в форме, пригодной для регулятора.

#### 12.8.5. 🟢 P2 — Federated synthesis

* Несколько data owner'ов обучают совместный генератор без обмена сырыми данными.
* Интеграция с **Flower**, **PySyft**, **OpenFL**.

---

### 12.9. Качество отчётов и юридическая значимость

#### 12.9.1. 🟡 P1 — Подписанные отчёты

* JSON-отчёт + GPG-подпись от платформы.
* Verifiable credentials / W3C Verifiable Data Registry.
* QR-код на отчёте для quick verification.

#### 12.9.2. 🟡 P1 — Юридически значимые форматы

* Экспорт отчёта в PDF/A с внедрённой подписью (для печатных архивов).
* Шаблоны отчёта под отраслевые регуляторы: ЦБ РФ, Роскомнадзор, Минздрав, Минцифры.
* Bilingual reports (RU/EN) для международных партнёров.

#### 12.9.3. 🟢 P2 — Comparative reports

* Diff-режим: «отчёт DP-CTGAN vs CTGAN на одном датасете» — наглядно показывает trade-off приватность ↔ полезность.
* Trend reports: «как менялось качество моделей за последние 6 месяцев».

---

### 12.10. Сводная таблица доработок (Roadmap)

| # | Доработка | Приоритет | Ориентир. трудоёмкость |
|---|---|---|---|
| 12.1.1 | Persistent Job Queue (Celery/Redis) | 🔴 P0 | 3 sprint'а |
| 12.1.2 | Multi-worker support | 🔴 P0 | 1 sprint |
| 12.1.3 | S3 object storage | 🔴 P0 | 2 sprint'а |
| 12.1.4 | Kubernetes deploy + Helm | 🟡 P1 | 4 sprint'а |
| 12.1.5 | Multi-tenancy | 🟡 P1 | 4 sprint'а |
| 12.2.1 | Model Registry (MLflow integration) | 🔴 P0 | 3 sprint'а |
| 12.2.2 | Data Lineage (OpenLineage) | 🔴 P0 | 3 sprint'а |
| 12.2.3 | Data Versioning (DVC/LakeFS) | 🟡 P1 | 2 sprint'а |
| 12.2.4 | Experiment Tracking (MLflow) | 🟡 P1 | 2 sprint'а |
| 12.3.1 | JSON-структурированные логи | 🔴 P0 | 1 sprint |
| 12.3.2 | OpenTelemetry tracing | 🔴 P0 | 2 sprint'а |
| 12.3.3 | Production-grade Prometheus metrics | 🔴 P0 | 2 sprint'а |
| 12.3.4 | Audit Log | 🟡 P1 | 2 sprint'а |
| 12.4.1 | OIDC/SAML auth | 🔴 P0 | 3 sprint'а |
| 12.4.2 | RBAC/ABAC | 🔴 P0 | 3 sprint'а |
| 12.4.3 | Secret Management (Vault) | 🔴 P0 | 2 sprint'а |
| 12.4.4 | Encryption (TLS, KMS, mTLS) | 🔴 P0 | 3 sprint'а |
| 12.4.5 | Per-Tenant Privacy Budget | 🟡 P1 | 3 sprint'а |
| 12.4.6 | Data Classification | 🟡 P1 | 2 sprint'а |
| 12.4.7 | 152-ФЗ / GDPR compliance | 🟡 P1 | 5 sprint'ов |
| 12.5.1 | CI/CD pipeline | 🔴 P0 | 2 sprint'а |
| 12.5.2 | IaC (Terraform) | 🟡 P1 | 3 sprint'а |
| 12.5.3 | DB migrations (Alembic) | 🟡 P1 | 1 sprint |
| 12.6.1 | GPU sharing (MIG/MPS) | 🔴 P0 | 2 sprint'а |
| 12.6.2 | Distributed training | 🟡 P1 | 5 sprint'ов |
| 12.7.1 | Web UI | 🟡 P1 | 6 sprint'ов |
| 12.7.2 | Notebook integration | 🟡 P1 | 2 sprint'а |
| 12.8.1 | Multi-table synthesis | 🟡 P1 | 5 sprint'ов |
| 12.8.2 | Time-series synthesis | 🟡 P1 | 5 sprint'ов |
| 12.9.1 | Подписанные отчёты | 🟡 P1 | 1 sprint |

> **Итого по P0** — около **30 sprint'ов** (≈ 15 человеко-месяцев при команде 4–6 инженеров).
> **Итого по P0+P1** — около **75 sprint'ов** (≈ 18–24 месяца разработки до полноценного enterprise-продукта).

---

### 12.11. Приоритеты внедрения по фазам

**Фаза 0 — Foundation (3–4 месяца):** P0-блок безопасности, состояния и наблюдаемости.
* Persistent Job Queue, S3, OIDC, RBAC, structured logging, distributed tracing, secret management, encryption, CI/CD.

**Фаза 1 — MLOps Core (4–6 месяцев):** Model Registry, Lineage, Multi-tenancy, GPU sharing, K8s.

**Фаза 2 — Compliance & UX (3–4 месяца):** Privacy Budget Accounting, Data Classification, Audit Log, Web UI, подписанные отчёты, 152-ФЗ/GDPR-соответствие.

**Фаза 3 — Расширение (6+ месяцев):** Multi-table, Time-series, Federated, AutoML, продвинутая privacy-evaluation.

---

## 13. Глоссарий

| Термин | Расшифровка |
|---|---|
| **DP** | Differential Privacy — формальная гарантия приватности через ε-δ неотличимость |
| **DP-SGD** | Differentially Private Stochastic Gradient Descent — DP-вариант SGD (Abadi et al., 2016) |
| **ε / epsilon** | Бюджет приватности; меньше = строже защита; типичные значения 1.0–10.0 |
| **δ / delta** | Вероятность нарушения DP-гарантии; обычно 1/(n·√n) |
| **DP-CTGAN** | Conditional Tabular GAN с встроенным DP-SGD (SmartNoise Synth) |
| **DP-TVAE** | Tabular VAE с DP-SGD через Opacus (PyTorch) |
| **CTGAN/TVAE/CopulaGAN** | SDV-генераторы без DP — baseline'ы |
| **JSD** | Jensen-Shannon Divergence — симметричная мера различия распределений |
| **TVD** | Total Variation Distance — мера различия категориальных распределений |
| **TRTR** | Train Real, Test Real — потолок ML-качества |
| **TSTR** | Train Synthetic, Test Real — качество синтетики на ML-задаче |
| **Utility Loss** | TRTR_score − TSTR_score |
| **DCR** | Distance to Closest Record |
| **NNDR** | Nearest Neighbor Distance Ratio |
| **MIA** | Membership Inference Attack — атака «была ли запись в обучающих данных» |
| **k-anonymity** | Каждая запись неотличима как минимум от k−1 других по квазиидентификаторам |
| **l-diversity** | В каждой группе по QI — не менее l разных значений чувствительного атрибута |
| **t-closeness** | Распределение SA в каждой группе близко к глобальному (расстояние ≤ t) |
| **Quasi-Identifier (QI)** | Атрибут, по которому в комбинации можно идентифицировать субъекта |
| **Sensitive Attribute (SA)** | Чувствительный атрибут (доход, диагноз, и т.п.) |
| **Holdout** | Отложенная выборка реальных данных, никогда не виденная генератором |
| **Run** | Один запуск пайплайна end-to-end |
| **Verdict** | Итоговый автоматический вердикт PASS / FAIL / PARTIAL |
| **MLOps** | Machine Learning Operations — практики promotion ML-моделей в production |
| **DPO** | Data Protection Officer — ответственный за защиту персональных данных |
| **PII** | Personally Identifiable Information |
| **OIDC** | OpenID Connect — стандарт identity-федерации поверх OAuth2 |
| **RBAC / ABAC** | Role-Based / Attribute-Based Access Control |
| **SLO / SLI** | Service Level Objective / Indicator |
| **DPIA** | Data Protection Impact Assessment (по GDPR Art. 35) |
| **ПНСТ** | Предварительный национальный стандарт (ГОСТ Р) |

---

*Документ описывает Систему по состоянию репозитория на дату составления (2026-04-25). Любые изменения архитектуры или функциональности должны сопровождаться обновлением соответствующих разделов и инкрементом версии PRD.*

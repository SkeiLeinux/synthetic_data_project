# ADR-008 — In-memory `job_store` в Synthesis Service

**Статус:** Принято (с известным техническим долгом)
**Дата:** 2026-04-25
**Связанные ADR:** [ADR-001](ADR-001-microservices.md), [ADR-007](ADR-007-redis-postgres.md), [ADR-010](ADR-010-background-tasks.md)

## Контекст

Synthesis Service выполняет тяжёлые задачи обучения генераторов
(минуты — часы). Каждое такое обучение оформлено как **джоб**
(`POST /api/v1/jobs` → `job_id` → polling `GET /api/v1/jobs/{id}`).

Нужно хранить состояние джобов:

* `job_id`, `status` (`queued | running | done | failed | cancelled`);
* `started_at`, `finished_at`;
* результат: `synth_path`, `model_id`, `dp_report`;
* при ошибке — `error_message`.

Кто и как читает/пишет это состояние:

* Пишет — фоновый поток `_run_job` (создан в обработчике
  `POST /jobs`).
* Читает — handler'ы `GET /jobs/{id}`, `DELETE /jobs/{id}`,
  `GET /jobs/{id}/dp_report`. Polling из Gateway каждые 10 секунд.
* Пишет также `DELETE /jobs/{id}` (отмена).

Контейнер запускается с `--workers 1`, поэтому конкурентный доступ —
только между несколькими потоками одного процесса.

## Рассмотренные варианты

### A. In-memory dict + threading.Lock

Простой словарь `Dict[job_id, JobRecord]` под `threading.Lock`.

* **За:** ноль зависимостей, нулевая латентность; для single-worker
  достаточно `threading.Lock`.
* **Против:** состояние теряется при рестарте контейнера; не работает
  при multi-replica.

### B. Redis (как у Gateway, см. ADR-007)

Хранить `JobRecord` в Redis по ключу `job:{job_id}`.

* **За:** durable между рестартами; работает при multi-replica;
  единый стек хранилищ с Gateway.
* **Против:** дополнительная зависимость, дополнительная сетевая ручка
  для каждого update; для MVP-нужд избыточно.

### C. PostgreSQL

Таблица `synthesis_jobs` с теми же полями.

* **За:** durable, аудит-трейл, SQL-запросы.
* **Против:** массивно для размера и срока жизни данных
  (несколько активных джобов одновременно, истории как таковой
  не требуется — она уже хранится у Gateway в RunRecord).

### D. Очередь задач (Celery / Dramatiq / RQ)

Передать жизненный цикл джоба полноценному менеджеру очередей.

* **За:** production-grade, retry, durability, multi-worker.
* **Против:** существенное усложнение архитектуры на MVP-уровне
  (broker, backend, beat, отдельные процессы воркеров); см.
  [ADR-010](ADR-010-background-tasks.md), который аналогично решает
  отказаться от полноценной очереди в пользу `BackgroundTasks`.

## Решение

Принят вариант **A — in-memory dict + threading.Lock**.

Реализация — `final_system/services/synthesis_service/job_store.py`:

```python
class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def add / get / update     # все методы под self._lock
```

Аргументация:

1. **Соответствие профилю джобов.** Активных джобов одновременно
   в типовом сценарии — единицы; история конкретных джобов
   нужна Gateway (и она хранится в RunRecord), а не самому
   Synthesis Service.
2. **Single-worker.** Контейнер запускается с `--workers 1`
   (см. PRD, NFR-25–NFR-29 и ограничения shared volume в
   [ADR-003](ADR-003-shared-volume.md)) — `threading.Lock`
   достаточен, race conditions между worker'ами невозможны.
3. **Минимум зависимостей.** Не вводит дополнительной сетевой
   зависимости для каждого update джоба.
4. **Соответствие духу MVP.** Аналогично подходу
   [ADR-010](ADR-010-background-tasks.md) — выбираем простейшее
   достаточное решение и фиксируем технический долг для
   production-перехода.

Этот выбор прямо отмечен как известное ограничение в
`CLAUDE.md` и в PRD (раздел 12.1.1) — он заменяется на persistent
job queue при первой же необходимости multi-replica деплоя.

## Последствия

### Положительные

* Минимальный код (`job_store.py` < 70 строк).
* Нулевая латентность чтения/записи.
* Нет дополнительных контейнеров и сетевых хопов.
* Отмена джоба (`DELETE /jobs/{id}`) — простой апдейт поля статуса,
  фоновый поток проверяет `rec.status == JobStatus.cancelled`
  в контрольных точках до `fit()` и до `sample()`.

### Отрицательные

* **Состояние теряется при рестарте контейнера.** Все активные
  джобы становятся «забытыми»: фоновые потоки убиты вместе
  с процессом, а Gateway продолжит поллить `GET /jobs/{id}` и
  получать 404. Запуск помечается `failed`.
* **Multi-replica невозможен.** Если поднять две реплики
  Synthesis Service, каждая будет хранить свой набор джобов;
  Gateway, попавший к «не той» реплике на поллинге, получит 404.
* **Нет durability**: все следы джобов исчезают через минуты после
  завершения (если процесс никогда не рестартует — данные растут
  до OOM; в практике это маловероятно, поскольку джобы редкие).
* Нет SQL-аудита по самим джобам Synthesis Service —
  только косвенно через `RunRecord`/ProcessRegistry в Gateway.

### Условия пересмотра

Перейти на persistent job queue (PRD 12.1.1) при наступлении
любого из:

* Деплой требует multi-replica Synthesis Service.
* Появление требования durability джобов (например, рестарт контейнера
  не должен убивать обучение, занявшее уже 2 часа).
* Появление требования retry-policy / dead-letter queue / rate limiting
  на уровне джобов.

Кандидаты на замену:

* **Celery + Redis/RabbitMQ** — стандарт Python-экосистемы;
* **Dramatiq** — более лёгкая альтернатива Celery;
* **Temporal.io** — workflow-ориентированный движок,
  пригодный для длинных пайплайнов с retry/checkpoint;
* **Argo Workflows** — для случая полного переезда в K8s.

## Связанные ADR

* [ADR-001](ADR-001-microservices.md) — Synthesis Service как
  отдельный процесс породил необходимость в собственном job_store.
* [ADR-007](ADR-007-redis-postgres.md) — Redis уже используется
  на Gateway, но осознанно не задействован для джобов синтеза.
* [ADR-010](ADR-010-background-tasks.md) — аналогичное по духу
  решение «MVP = простота, признаём долг» для оркестрации пайплайна.

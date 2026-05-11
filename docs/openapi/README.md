# OpenAPI спецификации сервисов

В этой папке лежат **OpenAPI 3.1.0** спецификации для всех 5 сервисов
платформы. Их можно загрузить в Swagger UI / Redoc / Postman / Insomnia
и сразу увидеть полный список ручек, схем запросов и ответов.

## Файлы

| Файл | Сервис | Порт | Префикс |
|---|---|---|---|
| `gateway.openapi.yaml` | Gateway (публичный API) | 8000 | `/api/v1` |
| `data_service.openapi.yaml` | Data Service | 8001 | `/api/v1` |
| `synthesis_service.openapi.yaml` | Synthesis Service | 8002 | `/api/v1` |
| `evaluation_service.openapi.yaml` | Evaluation Service | 8003 | `/api/v1` |
| `reporting_service.openapi.yaml` | Reporting Service | 8004 | `/api/v1` |

> Health-эндпоинты (`/health`, `/health/db`, `/health/gpu`, `/metrics` на Gateway) — **без** префикса `/api/v1`.

---

## Как открыть в Swagger UI

### Вариант 1 — онлайн-редактор (быстрее всего)

1. Откройте **<https://editor.swagger.io>**.
2. `File → Import file` → выберите нужный YAML.
3. Слева — редактор, справа — интерактивная документация.

### Вариант 2 — локальный Swagger UI в Docker

```bash
docker run --rm -p 8080:8080 \
  -e SWAGGER_JSON=/specs/gateway.openapi.yaml \
  -v "$(pwd)/openapi:/specs" \
  swaggerapi/swagger-ui
```

Откройте <http://localhost:8080>.

Чтобы переключаться между всеми спеками одновременно, используйте `URLS`:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/openapi:/specs" \
  -e URLS='[
    {"url":"/specs/gateway.openapi.yaml","name":"Gateway"},
    {"url":"/specs/data_service.openapi.yaml","name":"Data"},
    {"url":"/specs/synthesis_service.openapi.yaml","name":"Synthesis"},
    {"url":"/specs/evaluation_service.openapi.yaml","name":"Evaluation"},
    {"url":"/specs/reporting_service.openapi.yaml","name":"Reporting"}
  ]' \
  -e BASE_URL=/swagger \
  swaggerapi/swagger-ui
```

### Вариант 3 — Redoc (более читаемо для крупных API)

```bash
npx @redocly/cli preview-docs openapi/gateway.openapi.yaml
```

### Вариант 4 — встроенный Swagger каждого сервиса

Каждый сервис, поднятый через `docker compose up`, отдаёт **актуальную**
спецификацию автоматически:

| Сервис | Swagger UI | Сырой OpenAPI |
|---|---|---|
| Gateway | <http://localhost:8000/docs> | <http://localhost:8000/openapi.json> |
| Data | <http://localhost:8001/docs> | <http://localhost:8001/openapi.json> |
| Synthesis | <http://localhost:8002/docs> | <http://localhost:8002/openapi.json> |
| Evaluation | <http://localhost:8003/docs> | <http://localhost:8003/openapi.json> |
| Reporting | <http://localhost:8004/docs> | <http://localhost:8004/openapi.json> |

YAML-файлы в этой папке — **самодостаточные** (с примерами и описаниями)
и предназначены для случая, когда стек не запущен или нужно изучить
контракты офлайн.

---

## Как импортировать в Postman / Insomnia

### Postman

1. `File → Import` → выберите YAML.
2. Postman автоматически создаст коллекцию со всеми ручками.
3. Для Gateway добавьте переменную окружения `bearerAuth = <ваш API_KEY>`.

### Insomnia

1. `Application → Preferences → Data → Import Data → From File`.
2. Выберите YAML.

---

## Авторизация

* **Gateway** требует `Authorization: Bearer <API_KEY>` если переменная
  окружения `API_KEY` задана. В Swagger UI нажмите кнопку **Authorize**
  и вставьте токен.
* **Внутренние сервисы** (Data / Synthesis / Evaluation / Reporting)
  авторизации не имеют — они доступны только внутри docker-сети
  и не должны экспонироваться наружу.

---

## Связь между сервисами

Поток вызовов в пайплайне (упрощённо):

```
client ──POST /api/v1/runs──► Gateway
                                 │
                                 ├─► Data:       POST /api/v1/datasets (или from-db)
                                 ├─► Data:       POST /api/v1/datasets/{id}/split
                                 ├─► Synthesis:  POST /api/v1/jobs              (асинхронно)
                                 ├─► Synthesis:  GET  /api/v1/jobs/{id}         (поллинг каждые 10с)
                                 ├─► Evaluation: POST /api/v1/evaluate/privacy
                                 ├─► Evaluation: POST /api/v1/evaluate/utility
                                 └─► Reporting:  POST /api/v1/reports
```

Артефакты передаются через Shared Volume `/data` (а не через тело HTTP):

* `splits/{split_id}/{train,holdout}.csv` — Data Service пишет, Synthesis и Evaluation читают.
* `synth/{job_id}/synthetic.csv` — Synthesis пишет, Evaluation читает.
* `models/{model_id}.{pkl,meta.json}` — Synthesis пишет, Gateway читает метаданные.
* `reports/{dataset}__{generator}__{ts}.json` — Reporting пишет.

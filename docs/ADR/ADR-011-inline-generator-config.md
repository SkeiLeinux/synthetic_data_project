# ADR-011 — Inline-конфиг генератора в `POST /jobs`

**Статус:** Принято
**Дата:** 2026-04-25
**Связанные ADR:** [ADR-001](ADR-001-microservices.md), [ADR-005](ADR-005-pydantic.md), [ADR-012](ADR-012-lazy-ml-imports.md)

## Контекст

Gateway (`api/routers/runs.py:_execute_pipeline`) на одном из шагов
пайплайна вызывает Synthesis Service:

```
POST /api/v1/jobs
{
  "split_id": "...",
  "generator": { ... }     # ← вопрос ADR: что здесь?
  ...
}
```

Параметры генератора (тип, ε, эпохи, архитектура, batch size и т.д.)
описаны в YAML-конфиге пайплайна (`final_system/configs/*.yaml`).
Вопрос — как Synthesis Service должен получить эти параметры:

* **(a)** Gateway передаёт **имя конфига** (`config_name: "adult"`),
  а Synthesis Service читает YAML с диска;
* **(b)** Gateway сериализует валидированный конфиг и передаёт
  его **inline** в теле запроса как JSON.

Дополнительные обстоятельства:

* `quick_test=true` модифицирует параметры конфига **на стороне
  Gateway** (`apply_quick_test()` в `config_loader.py`) — sample_size,
  epochs, cuda, пороги вердикта;
* при `max_iterations > 1` (повторное обучение при FAIL) Gateway
  может захотеть подменить параметры между итерациями;
* директория `configs/` примонтирована в Gateway (`./configs:/app/configs`);
  можно либо примонтировать её и в Synthesis Service, либо нет.

## Рассмотренные варианты

### A. Передача `config_name`, чтение YAML из Synthesis Service

Synthesis Service монтирует `configs/`, читает `{config_name}.yaml`,
парсит, валидирует.

* **За:** короткое тело запроса.
* **Против:** Synthesis Service зависит от **формата YAML** и от
  директории `configs/`; при изменении формата конфига нужно
  обновлять оба сервиса; `quick_test`-overrides нужно либо дублировать
  на стороне synthesis, либо передавать поверх в теле запроса —
  ломает «единственный источник правды».

### B. Inline-конфиг генератора в теле запроса

Gateway парсит YAML, валидирует через `GeneratorYamlConfig`,
применяет `apply_quick_test` если нужно, сериализует через
`model_dump(mode="json")` и передаёт как `body.generator`.

* **За:** Synthesis Service не видит ни `configs/`, ни YAML, ни
  `quick_test` — только финальный валидный набор параметров.
* **Против:** размер тела запроса больше; при добавлении нового
  поля в `GeneratorYamlConfig` нужно учесть его на обоих концах
  (но контракт уже общий — `shared/schemas/synthesis.py`).

### C. Гибрид: `config_name` + override-секция

Передавать имя + дельту изменений.

* **За:** короткое тело + гибкость override.
* **Против:** хуже всего по отладке («что реально применилось?»);
  Synthesis Service всё равно зависит от YAML.

## Решение

Принят вариант **B — inline-конфиг генератора**.

Реализация:

* В `_execute_pipeline` (Gateway):
  ```python
  cfg = load_config(config_path)
  if quick_test:
      cfg = apply_quick_test(cfg)
  generator_body = cfg.generator.model_dump(mode="json")
  ...
  synth_cli.post("/api/v1/jobs", json={
      "split_id": split_id,
      "generator": generator_body,
      ...
  })
  ```
* В Synthesis Service (`services/synthesis_service/router.py:_run_job`):
  ```python
  gen_yaml = GeneratorYamlConfig.model_validate(body.generator)
  generator = _build_generator(gen_yaml)
  ```
* `_build_generator` — единственное место в коде, где `generator_type`
  превращается в конкретный класс (`DPCTGANGenerator`,
  `DPTVAEGenerator`, `CTGANGenerator`, `TVAEGenerator`,
  `CopulaGANGenerator`).
* Директория `configs/` **не примонтирована** в Synthesis Service
  (см. `final_system/docker-compose.yml`).

Аргументация:

1. **Развязка по формату конфига.** Synthesis Service не знает
   ничего про YAML, `apply_quick_test` или `configs/`. Если в будущем
   формат конфига сменится (TOML, БД, UI-billder) — Synthesis Service
   не затронут.
2. **Развязка по добавлению генераторов.** Чтобы добавить новый
   генератор, достаточно создать `synthesizer/my_generator.py`
   и добавить ветку в `_build_generator()`. Никаких изменений
   в Gateway (ни в коде, ни в перебилде образа). См.
   [ADR-012](ADR-012-lazy-ml-imports.md), который вместе с этим ADR
   обеспечивает изоляцию ML-зависимостей.
3. **Аудитируемость.** Что именно ушло на синтез — однозначно
   видно в `RunRecord.config_snapshot` (Gateway сохраняет туда
   `yaml.safe_load(open(config_path))`) и в логах `POST /jobs`
   у Synthesis Service. При гибридной схеме «name + override»
   восстанавливать «что реально применилось» сложнее.
4. **`quick_test` обрабатывается на одной стороне.** Gateway
   модифицирует конфиг **до** отправки; Synthesis Service видит
   уже финальный набор параметров без условной логики.

## Последствия

### Положительные

* Synthesis Service не имеет volume `configs/` — image и runtime
  чище.
* Любые модификации конфига (`apply_quick_test`, потенциальные
  будущие override'ы) централизованы в Gateway.
* Добавление нового генератора не требует **перебилда Gateway** —
  достаточно перебилда `synthesis_service` + правки одной ветки
  в `_build_generator()`.
* Контракт `SynthesisJobCreate.generator: Dict[str, Any]` валидируется
  строгой Pydantic-схемой `GeneratorYamlConfig` (см.
  [ADR-005](ADR-005-pydantic.md)) на входе Synthesis Service —
  ошибка типа выявляется до начала обучения.

### Отрицательные

* **Размер тела запроса.** `GeneratorYamlConfig` — десятки полей;
  JSON-сериализация — ~1–2 КБ. Не критично, но больше, чем
  одно имя.
* **Контракт надо поддерживать.** Любое новое поле в
  `GeneratorYamlConfig` нужно обработать в `_build_generator()`
  (то есть передать в `DPCTGANConfig` / `DPTVAEConfig` / и т.д.).
  Pydantic-валидация ловит опечатки в имени поля, но не «забыли
  пробросить дальше».
* `apply_quick_test`-overrides не видны у Synthesis Service как
  «это quick_test»; они видны только как финальные значения. Для
  отладки это компромисс — приходится сверяться с Gateway-логом.

### Условия пересмотра

Не пересматривать без существенной перестройки. Возможные триггеры:

* Появление массивных, многомегабайтных параметров (например,
  pre-trained embedding'и, передаваемые в инициализацию генератора) —
  тогда логичнее передавать ссылку на файл вместо inline.
* Появление UI-конфигуратора, генерирующего параметры
  программно без YAML, — фактически это укрепляет данное ADR
  (Gateway в любом случае останется единственным потребителем
  «где конфиг лежит»).

## Связанные ADR

* [ADR-001](ADR-001-microservices.md) — Synthesis Service как
  отдельный сервис, который мы хотим минимально связывать с Gateway.
* [ADR-005](ADR-005-pydantic.md) — `GeneratorYamlConfig` как
  Pydantic-модель — общий контракт между сервисами.
* [ADR-012](ADR-012-lazy-ml-imports.md) — двойственное
  следствие: Gateway не импортирует ML-классы, Synthesis Service
  не импортирует логику работы с YAML/конфигами.

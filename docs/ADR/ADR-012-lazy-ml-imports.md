# ADR-012 — Lazy ML imports в `config_loader.py`

**Статус:** Принято
**Дата:** 2026-04-25
**Связанные ADR:** [ADR-001](ADR-001-microservices.md), [ADR-009](ADR-009-pickle-sidecar.md), [ADR-011](ADR-011-inline-generator-config.md)

## Контекст

Файл `final_system/config_loader.py` выполняет три задачи:

1. Парсит YAML-конфиг пайплайна.
2. Валидирует его через Pydantic-схемы (`AppConfig` и nested-классы).
3. Предоставляет методы конвертации в конфиги конкретных
   генераторов: `to_dpctgan_config()`, `to_dptvae_config()`,
   `to_ctgan_config()` и т.д. — каждый возвращает
   `DPCTGANConfig` / `DPTVAEConfig` / `CTGANConfig` из соответствующих
   модулей `synthesizer/`.

Этот же файл импортируется:

* **Gateway-ом** — чтобы валидировать конфиги при загрузке через
  `POST /configs`, при создании запуска через `POST /runs`,
  и при сохранении `config_snapshot` в RunRecord.
* **CLI / monolith-режимом** (исторический сценарий) — там нужны
  именно методы `to_*_config()` для конструирования генераторов.

Если бы `config_loader.py` импортировал классы синтезаторов
**top-level** (`from synthesizer.dp_ctgan import DPCTGANConfig`),
то цепочка получилась бы такая:

```
Gateway → import config_loader
        → import synthesizer.dp_ctgan
        → import snsynth, torch
```

И Gateway-образ обязан был бы содержать `torch + opacus +
smartnoise-synth + sdv` (~ 13 ГБ) только для того, чтобы
прочитать YAML и вернуть `200 OK`.

## Рассмотренные варианты

### A. Top-level импорты

Стандартный Python-стиль: `from synthesizer.dp_ctgan import DPCTGANConfig`
в начале файла.

* **За:** идиоматично, явно, IDE подсказывает.
* **Против:** Gateway тащит все ML-зависимости.

### B. Lazy imports внутри методов

Импорты `synthesizer.*` сделать внутри `to_dpctgan_config()` и
аналогичных методов:

```python
def to_dpctgan_config(self) -> Any:
    from synthesizer.dp_ctgan import DPCTGANConfig
    return DPCTGANConfig(...)
```

* **За:** Gateway не выполняет эти импорты, пока не вызовет
  `to_*_config()` (а он не вызывает — только Synthesis Service в
  monolith-режиме, и сам Synthesis Service в микросервисном
  режиме строит конфиги без `to_*_config()`).
* **Против:** lazy-импорты «размазаны» по коду; статические анализаторы
  и type-checkers могут жаловаться без `TYPE_CHECKING`-блоков
  (приходится `Any` в return-аннотации).

### C. Разделить `config_loader` на «лёгкую» и «тяжёлую» части

Например, `config_loader_base.py` (только Pydantic-валидация)
и `config_loader_synthesizer.py` (методы `to_*_config()` с импортами).

* **За:** явное разделение, без lazy-имхортов.
* **Против:** существенное дублирование кода между двумя файлами;
  риск рассинхронизации; усложнение import-графа без существенной
  выгоды по сравнению с lazy-импортами.

### D. Перенос `to_*_config()` в `synthesizer/loader.py` или прямо в
`synthesis_service/router.py`

Полностью убрать из `config_loader` любые упоминания классов
синтезаторов.

* **За:** теоретически чистое разделение.
* **Против:** monolith/CLI-сценарий требует одного места, где
  Pydantic-конфиг превращается в конкретный объект; либо это
  место — `config_loader`, либо где-то нужна аналогичная функция,
  которая всё равно импортирует все синтезаторы.

## Решение

Принят вариант **B — lazy imports внутри методов**.

Реализация:

* `config_loader.py` **не имеет** top-level импортов из `synthesizer/`,
  `evaluator/`, `services/reporting_service/`.
* Каждый `to_*_config()` начинается с локального import:
  ```python
  def to_dpctgan_config(self) -> Any:
      from synthesizer.dp_ctgan import DPCTGANConfig
      return DPCTGANConfig(...)
  ```
* Аналогично — `UtilityYamlConfig.to_utility_config()`,
  `PrivacyYamlConfig.to_privacy_config()`,
  `ThresholdsYamlConfig.to_verdict_thresholds()`.
* Return-тип аннотирован как `Any`, чтобы не пришлось делать
  top-level `from ... import` ради type-hint'а (альтернатива —
  `TYPE_CHECKING`-блоки, в текущей реализации не использованы).

Это решение в коде явно прокомментировано:

> `config_loader — чистый YAML-парсер без зависимостей на ML-библиотеки.`
> `Импорты synthesizer / evaluator / reporter намеренно вынесены`
> `ВНУТРЬ методов (lazy imports), чтобы:`
> `1. Gateway мог импортировать config_loader не имея torch/opacus/smartnoise.`
> `2. Добавление нового генератора не требовало перебилда Gateway —`
> `   достаточно изменить только synthesis_service.`

В микросервисном режиме (см. [ADR-001](ADR-001-microservices.md))
методы `to_*_config()` фактически **не используются** (Synthesis
Service строит конфиги напрямую в `_build_generator()`,
см. [ADR-011](ADR-011-inline-generator-config.md)). Lazy-импорты
сохранены для совместимости со старым monolith/CLI-режимом.

## Последствия

### Положительные

* **Gateway-образ остаётся слим** (десятки–сотни МБ) — содержит
  только `fastapi`, `pydantic`, `redis`, `pandas`, `httpx`,
  `sqlalchemy`, `psycopg2-binary`. Никакого `torch`/`opacus`/
  `smartnoise`/`sdv`.
* **Reporting Service** аналогично может импортировать
  `config_loader` (если потребуется) без получения тяжёлых
  зависимостей.
* **Время старта Gateway** — секунды, а не десятки секунд
  (top-level `import torch` сам по себе занимает заметное время).
* Добавление нового генератора (новый файл в `synthesizer/`,
  новые поля в `GeneratorYamlConfig`, новая ветка в
  `_build_generator()`) **не требует ребилда Gateway** — нужно
  пересобрать только `synthesis_service`.

### Отрицательные

* Lazy-импорты не сразу очевидны при чтении кода — нужен
  комментарий, объясняющий зачем они так сделаны (комментарий
  в `config_loader.py` присутствует).
* Type checkers (mypy, pyright) не могут вывести типы возвращаемых
  объектов из `to_*_config()` без дополнительных усилий
  (`TYPE_CHECKING` или `Any`).
* Ошибки импорта (например, отсутствует `snsynth`) проявляются
  не на старте, а в момент первого вызова `to_dpctgan_config()`.
  В микросервисном режиме это не проблема (Gateway не вызывает
  эти методы вообще), но в monolith/CLI-режиме это можно поймать
  только во время первого пайплайна.
* **Хрупкое ограничение.** Любой коммит, добавляющий top-level
  `import torch` (или любую ML-зависимость) в `config_loader`,
  автоматически нарушает правило и раздувает Gateway. Нужна
  явная договорённость в команде / lint-проверка.

### Условия пересмотра

Не пересматривать без существенных причин. Триггеры теоретического
пересмотра:

* Если monolith-режим будет полностью удалён, можно перенести
  методы `to_*_config()` в `synthesizer/loader.py` (или удалить
  их вообще, оставив `_build_generator` как единственного
  потребителя `GeneratorYamlConfig`) — тогда `config_loader`
  станет действительно «чистым» Pydantic-парсером без любых
  зависимостей.
* Если появится lint-инструмент, формализующий правило
  «Gateway не импортирует ML», его нужно добавить в CI.

## Связанные ADR

* [ADR-001](ADR-001-microservices.md) — изоляция ML-зависимостей —
  следствие микросервисной архитектуры.
* [ADR-011](ADR-011-inline-generator-config.md) — Synthesis Service
  работает напрямую с `GeneratorYamlConfig` и не использует
  `to_*_config()` методы; lazy-импорты остались только для
  monolith-режима.
* [ADR-009](ADR-009-pickle-sidecar.md) — другое следствие изоляции:
  Gateway не выполняет `pickle.load` модели, поэтому метаданные
  отделены в JSON-сайдкар.

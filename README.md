# Synthetic Data Generation Service

**Сервис генерации полностью синтетических табличных данных с оценкой приватности и полезности**

Дипломная работа — НИУ ВШЭ, ФКН, Программная инженерия, 2024–2026.

---

## Описание

Система реализует полный пайплайн создания синтетических табличных данных на основе реального датасета. Поддерживаются модели с дифференциальной приватностью (DP-CTGAN, DP-TVAE) и без неё (CTGAN, CopulaGAN). После генерации автоматически вычисляются метрики приватности и полезности, формируется отчёт с агрегированным вердиктом PASS/FAIL/PARTIAL.

**Ключевые возможности:**
- Генерация синтетических данных на базе SmartNoise Synth с DP-гарантиями (DP-SGD)
- Полный аудит расхода DP-бюджета (spent epsilon, delta, история по эпохам)
- Оценка приватности: классические метрики (k/l/t), дистанционные метрики (DCR, NNDR), симуляция атаки MIA
- Оценка полезности: статистическое сходство (JSD, TVD), сравнение корреляций, ML-эффективность (TSTR/TRTR)
- Хранение результатов в PostgreSQL + JSON-отчёты на диск

---

## Структура репозитория

```
.
├── db/
│   └── init_db.sql                  # Инициализация схемы БД
├── final_system/
│   ├── synthesizer/
│   │   └── dp_ctgan.py              # DP-CTGAN генератор (SmartNoise)
│   ├── evaluator/
│   │   ├── privacy/
│   │   │   ├── classical.py         # k-анонимность, l-разнообразие, t-близость
│   │   │   ├── distance_metrics.py  # DCR и NNDR метрики
│   │   │   ├── attack_simulation.py # Симуляция Membership Inference Attack
│   │   │   └── privacy_evaluator.py # Оркестратор оценки приватности
│   │   └── utility/
│   │       ├── statistical.py       # JSD, TVD, дельта корреляций
│   │       ├── ml_efficacy.py       # TSTR / TRTR
│   │       └── utility_evaluator.py # Оркестратор оценки полезности
│   ├── reporter/
│   │   ├── reporter.py              # Сборка отчёта и вердикта
│   │   └── reports/                 # Сохранённые JSON-отчёты
│   ├── archive/                     # Предыдущие версии компонентов (не используются)
│   ├── data/                        # Датасеты (не включены в репозиторий, см. .gitignore)
│   ├── logs/                        # Логи запусков
│   ├── main.py                      # Сквозной пайплайн run_pipeline()
│   ├── run_adult.py                 # Точка входа — запуск на датасете Adult Census
│   ├── processor.py                 # Предобработка данных
│   ├── data_manager.py              # Работа с БД (PostgreSQL через SQLAlchemy)
│   ├── logger_config.py             # Настройка логгера
│   ├── config.ini                   # Конфигурация системы
│   └── metadata.json                # Метаданные SDV для датасета Adult
├── requirements.txt
└── README.md
```

---

## Требования

- Python 3.10+
- PostgreSQL 14+ (для хранения результатов; запуск без БД возможен в упрощённом режиме)
- CUDA-совместимый GPU (опционально, но значительно ускоряет обучение)

Зависимости Python:

```
numpy>=1.26,<3.0
pandas>=2.1,<3.0
SQLAlchemy>=2.0,<3.0
psycopg2-binary>=2.9
smartnoise-synth>=1.0.6
sdv
scikit-learn>=1.4,<2.0
scipy>=1.12
tqdm
```

Установка:

```bash
pip install -r requirements.txt
```

---

## Быстрый старт

### 1. Инициализация базы данных

```bash
psql -U postgres -d synthetic_data_db -f db/init_db.sql
```

### 2. Настройка конфигурации
**НЕАКТУАЛЬНО**

Отредактируйте `final_system/config.ini`. Основные параметры:

```ini
[DATABASE]
host = localhost
port = 5432
dbname = synthetic_data_db
user = postgres
password = your_password

[GENERATOR]
# Параметры в JSON-формате, см. DPCTGANConfig в synthesizer/dp_ctgan.py

[PRIVACY]
# quasi_identifiers, sensitive_attribute и флаги метрик

[UTILITY]
# target_column и task_type (classification / regression)
```

### 3. Подготовка данных

Поместите исходный CSV-файл в `final_system/data/`. Пример для датасета Adult Census:

```bash
# Скачайте adult.csv с UCI ML Repository и положите в final_system/data/
```

### 4. Запуск пайплайна

```bash
cd final_system
python run_adult.py
```

Для быстрого теста (5k строк, 50 эпох, ~2–3 минуты):

```python
# В run_adult.py установите:
QUICK_TEST = True
```

---

## Архитектура пайплайна

```
Реальные данные
      │
      ▼
[DataProcessor]           ← очистка, заполнение пропусков
      │
      ▼
 train / holdout split    ← ДО обучения генератора (методологически важно)
      │
   ┌──┴───────────────┐
   ▼                  ▼
[DPCTGANGenerator]  holdout (генератор не видит)
   │
   ▼
[Синтетические данные]
   │
   ├──▶ [PrivacyEvaluator]   ← DCR, NNDR, MIA, k/l/t
   └──▶ [UtilityEvaluator]   ← JSD, TVD, TSTR/TRTR
              │
              ▼
         [Reporter]          ← вердикт PASS / FAIL / PARTIAL + JSON-отчёт
```

Разделение реальных данных на `real_train` и `real_holdout` выполняется **до** обучения генератора. Это обеспечивает корректную оценку меморизации: `real_holdout` остаётся невидимым для генератора и используется как контрольная группа при вычислении DCR и MIA.

---

## Компоненты

### Генератор (`synthesizer/dp_ctgan.py`)

Обёртка над SmartNoise Synth DPCTGAN с полным аудитом DP-бюджета.

`privacy_report()` возвращает структуру с разделением:
- `dp_config` — параметры DP (epsilon, delta, sigma, max_grad_norm)
- `dp_spent` — фактический расход бюджета (spent_epsilon_final, epochs_completed, история)
- `reproducibility` — random_seed и версия библиотеки

### Оценка приватности (`evaluator/privacy/`)

Структура отчёта:
- `dp_guarantees` — формальные DP-гарантии от генератора
- `empirical_risk.distance_metrics` — DCR (synth→real vs holdout→real), NNDR
- `empirical_risk.membership_inference` — AUC атакующего классификатора
- `diagnostic.classical` — k-анонимность, l-разнообразие, t-близость

### Оценка полезности (`evaluator/utility/`)

Структура отчёта:
- `statistical` — JSD по числовым колонкам, TVD по категориальным, дельты mean/std/median
- `correlations` — MAE матриц Pearson и Cramér's V
- `ml_efficacy` — TRTR score, TSTR score, Utility Loss

### Репортер (`reporter/reporter.py`)

Вердикт:
- `PASS` — все проверки пройдены
- `FAIL` — одна или несколько проверок провалены (перечислены в `issues`)
- `PARTIAL` — часть модулей не запускалась

---

## Пример вывода

```
============================================================
  Вердикт: PASS
============================================================
  DP:       spent_ε = 2.48, epochs = 87/300
  Utility:  TRTR F1 = 0.834, TSTR F1 = 0.811, loss = 0.023
  Privacy:  MIA AUC = 0.516, DCR ok = True
  k/l/t:    k=43, l=12, t=0.377
  Синтетика: data/adult_synth.csv (26000 строк)
  Отчёт:    reporter/reports/
============================================================
```

---

## База данных

Схема `synthetic_data_schema` содержит четыре таблицы:

| Таблица | Назначение |
|---------|-----------|
| `processes` | Запись о каждом запуске пайплайна (статус, время, пути к данным) |
| `process_logs` | Логи запуска в текстовом виде |
| `process_metadata` | Метаданные в JSONB (параметры генерации, метрики приватности и полезности) |
| `metadata_types` | Справочник типов метаданных |

Инициализация: `db/init_db.sql`

---

## Конфигурация

Все параметры системы задаются в `final_system/config.ini`:

| Секция | Описание |
|--------|----------|
| `[DATABASE]` | Параметры подключения к PostgreSQL |
| `[PATHS]` | Пути к входным данным, логам, конфигу |
| `[GENERATOR]` | Параметры DPCTGANConfig в JSON-формате |
| `[PRIVACY]` | Квазиидентификаторы, чувствительный атрибут, флаги метрик |
| `[UTILITY]` | Целевой признак, тип задачи, исключаемые колонки |
| `[DATA_SCHEMA]` | Разбивка колонок по типам (категориальные / непрерывные) |
| `[PIPELINE]` | Параметры пайплайна (размер выборки, holdout, предобработка) |

---

## Воспроизводимость

Для воспроизводимости экспериментов:
- Фиксируйте `random_seed` в `DPCTGANConfig`
- Сохраняйте обученную модель: `generator.save("model.pkl")`
- Загрузка: `DPCTGANGenerator.load("model.pkl")`
- Версия библиотеки фиксируется в `privacy_report()["snsynth_version"]`

---

## Лицензия

Проект создан в учебных целях в рамках выпускной квалификационной работы НИУ ВШЭ.
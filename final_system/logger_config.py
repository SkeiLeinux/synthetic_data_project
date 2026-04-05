# final_system/logger_config.py
#
# Централизованная настройка логирования.
#
# Вызов setup_logger(__name__) в любом модуле:
#   1. Настраивает корневой логгер (один раз) — весь лог идёт в файл + консоль
#   2. Возвращает именованный логгер для модуля
#
# Благодаря настройке корневого логгера сюда попадает всё:
#   - логи из data_service, synthesizer, evaluator, reporter
#   - logging.basicConfig-логи из main.py и cli.py (их можно убрать)
#   - логи сторонних библиотек уровня WARNING и выше

import logging
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _configure_root_logger() -> None:
    """
    Настраивает корневой логгер: консоль (INFO) + файл (DEBUG).
    Вызывается один раз при первом обращении к setup_logger().
    Повторные вызовы — no-op благодаря проверке root.handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # уже настроен — выходим

    root.setLevel(logging.DEBUG)

    # ── Консоль: INFO и выше ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)

    # ── Файл: DEBUG и выше (полный лог) ──────────────────────────────────────
    _LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)

    root.addHandler(console_handler)
    root.addHandler(file_handler)


def setup_logger(name: str) -> logging.Logger:
    """
    Возвращает именованный логгер для модуля.
    При первом вызове настраивает корневой логгер (файл + консоль).

    Использование:
        logger = setup_logger(__name__)
        logger.info("Привет из модуля")
    """
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.propagate = True  # передавать записи корневому логгеру
    return logger
# shared/log_context.py
#
# Контекстная переменная run_id для сквозной трассировки запросов по всем сервисам.
# Работает через Python contextvars: каждый поток/asyncio-задача имеет свой контекст.
#
# Использование:
#   from shared.log_context import set_run_id, RunIdFilter, LOG_FORMAT, LOG_DATE_FORMAT
#
#   # В начале обработчика/фонового треда:
#   set_run_id(body.run_id)
#
#   # В main.py после basicConfig:
#   logging.getLogger().addFilter(RunIdFilter())

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

_run_id_var: ContextVar[str] = ContextVar("run_id", default="")


def set_run_id(run_id: Optional[str]) -> None:
    _run_id_var.set(run_id or "")


def clear_run_id() -> None:
    _run_id_var.set("")


class RunIdFormatter(logging.Formatter):
    """Форматтер, добавляющий run_id из ContextVar в каждую запись лога.

    Форматтер вызывается на хендлере и корректно работает при propagation
    из дочерних логгеров — в отличие от фильтра на root logger.
    """

    def format(self, record: logging.LogRecord) -> str:
        rid = _run_id_var.get()
        record.run_id_tag = f" [run {rid}]" if rid else ""  # type: ignore[attr-defined]
        return super().format(record)


LOG_FORMAT      = "[%(asctime)s]%(run_id_tag)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

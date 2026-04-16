# shared/schemas/evaluation.py
#
# Контракт Evaluation Service: запросы на оценку приватности и полезности.
# Используется: Evaluation Service (потребляет), Gateway (отправляет).

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PrivacyEvalRequest(BaseModel):
    """
    Тело запроса POST /evaluate/privacy.

    split_id используется для поиска train.csv и holdout.csv на Shared Volume.
    synth_path — путь к synthetic.csv на Shared Volume (относительно data_root).
    dp_report  — DP-отчёт от генератора (опционально; нужен для dp_guarantees секции).
    quasi_identifiers и sensitive_attribute нужны для k/l/t-анонимности;
    если не указаны — классические метрики пропускаются.
    """
    split_id: str
    synth_path: str
    dp_report: Optional[Dict[str, Any]] = None
    quasi_identifiers: List[str] = []
    sensitive_attribute: Optional[str] = None


class UtilityEvalRequest(BaseModel):
    """
    Тело запроса POST /evaluate/utility.

    Схема колонок передаётся явно, т.к. Evaluation Service не знает о конфиге —
    он получает только данные и задание.
    """
    split_id: str
    synth_path: str
    target_column: str
    categorical_columns: List[str]
    continuous_columns: List[str]

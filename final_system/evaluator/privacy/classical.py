"""
classical.py

Классические метрики анонимности: k-анонимность, l-разнообразие, t-близость.
Перенесены из validator.py курсовой с рефакторингом под новую архитектуру.

ВАЖНО: это диагностические метрики, не замена DP-гарантиям.
k/l/t описывают структуру таблицы, но не дают математических гарантий
против атак восстановления. В отчете эти метрики идут в раздел
'diagnostic', отдельно от 'dp_guarantees' и 'empirical_risk'.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
    """
    k-анонимность: минимальный размер группы с одинаковой комбинацией QI.
    Чем больше k, тем сложнее идентифицировать конкретного субъекта.
    k=1 означает, что в датасете есть уникальные записи — серьезный риск.
    """
    missing = [q for q in quasi_identifiers if q not in df.columns]
    if missing:
        raise ValueError(f"Квазиидентификаторы не найдены в датасете: {missing}")

    group_sizes = df.groupby(quasi_identifiers, observed=True).size()
    k = int(group_sizes.min())
    logger.info(f"[classical] k-anonymity = {k} (группы: min={k}, max={group_sizes.max()})")
    return k


def compute_l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attribute: str,
) -> int:
    """
    l-разнообразие: минимальное число различных значений чувствительного атрибута
    внутри каждой QI-группы. Защищает от атрибутивных атак.
    l=1 означает, что в какой-то группе все записи имеют одно значение SA.
    """
    if sensitive_attribute not in df.columns:
        raise ValueError(f"Чувствительный атрибут '{sensitive_attribute}' не найден")

    l_values = df.groupby(quasi_identifiers, observed=True)[sensitive_attribute].nunique()
    l = int(l_values.min())
    logger.info(f"[classical] l-diversity = {l}")
    return l


def compute_t_closeness(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attribute: str,
    reference_df: pd.DataFrame,
) -> float:
    """
    t-близость: максимальное расстояние Earth Mover's Distance (EMD) между
    распределением SA в группе и глобальным распределением SA в reference_df.
    Чем меньше t, тем ближе локальное распределение к глобальному.

    reference_df — реальный датасет, с которым сравниваем синтетику.
    Это важно: для синтетики мы сравниваем с оригиналом, а не сами с собой.
    """
    if sensitive_attribute not in df.columns:
        raise ValueError(f"Чувствительный атрибут '{sensitive_attribute}' не найден")

    # Глобальное распределение SA из эталонного (реального) датасета
    global_dist = reference_df[sensitive_attribute].value_counts(normalize=True)

    max_t = 0.0
    for _, group in df.groupby(quasi_identifiers, observed=True):
        group_dist = group[sensitive_attribute].value_counts(normalize=True)

        # EMD для категориального SA: сумма абсолютных разностей частот
        all_vals = set(global_dist.index) | set(group_dist.index)
        emd = sum(
            abs(global_dist.get(v, 0.0) - group_dist.get(v, 0.0))
            for v in all_vals
        ) / 2.0  # делим на 2, т.к. сумма abs разностей двух распределений = 2 * TVD

        max_t = max(max_t, emd)

    t = round(max_t, 6)
    logger.info(f"[classical] t-closeness = {t}")
    return t


def compute_classical_metrics(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attribute: str,
) -> Dict:
    """
    Считает все три классические метрики в одном вызове.
    Возвращает словарь, совместимый с форматом privacy_report.
    """
    return {
        "k_anonymity": compute_k_anonymity(synth_df, quasi_identifiers),
        "l_diversity": compute_l_diversity(synth_df, quasi_identifiers, sensitive_attribute),
        "t_closeness": compute_t_closeness(synth_df, quasi_identifiers, sensitive_attribute, real_df),
        "quasi_identifiers": quasi_identifiers,
        "sensitive_attribute": sensitive_attribute,
    }

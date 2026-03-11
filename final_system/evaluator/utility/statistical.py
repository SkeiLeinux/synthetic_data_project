"""
statistical.py

Статистические метрики сходства между реальными и синтетическими данными.
Работает покоменно: числовые колонки → JSD + stats delta, категориальные → TVD.
Отдельно считается разница матриц корреляций.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Разделяет колонки на числовые и категориальные по dtype."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


def _compute_jsd(real_col: pd.Series, synth_col: pd.Series, bins: int = 50) -> float:
    """
    Jensen-Shannon Divergence для числовых колонок.
    JSD ∈ [0, 1]: 0 = идентичные распределения, 1 = полностью различные.
    Биннинг выполняется по объединенному диапазону, чтобы гистограммы были сравнимы.
    """
    combined_min = min(real_col.min(), synth_col.min())
    combined_max = max(real_col.max(), synth_col.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    # Нормируем в вероятностное распределение (density=True)
    real_hist, _ = np.histogram(real_col.dropna(), bins=bin_edges, density=True)
    synth_hist, _ = np.histogram(synth_col.dropna(), bins=bin_edges, density=True)

    # Добавляем малый сглаживающий шум, чтобы избежать деления на 0 в JSD
    eps = 1e-10
    real_hist = real_hist + eps
    synth_hist = synth_hist + eps

    # jensenshannon возвращает корень из JSD, поэтому возводим в квадрат
    return float(jensenshannon(real_hist, synth_hist) ** 2)


def _compute_tvd(real_col: pd.Series, synth_col: pd.Series) -> float:
    """
    Total Variation Distance для категориальных колонок.
    TVD ∈ [0, 1]: 0 = одинаковые распределения, 1 = нет общих категорий.
    """
    # Считаем нормализованные частоты по всем категориям из обоих датасетов
    all_categories = set(real_col.dropna().unique()) | set(synth_col.dropna().unique())

    real_freq = real_col.value_counts(normalize=True)
    synth_freq = synth_col.value_counts(normalize=True)

    tvd = 0.5 * sum(
        abs(real_freq.get(cat, 0.0) - synth_freq.get(cat, 0.0))
        for cat in all_categories
    )
    return float(tvd)


def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """
    Cramér's V — симметричная мера ассоциации для двух категориальных колонок.
    Значение ∈ [0, 1]: 0 = нет ассоциации, 1 = полная ассоциация.
    Используется вместо Pearson там, где Pearson неприменим.
    """
    contingency = pd.crosstab(col_a, col_b)
    n = contingency.sum().sum()
    if n == 0:
        return 0.0

    chi2 = 0.0
    expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / n
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            e = expected[i, j]
            if e > 0:
                chi2 += (contingency.iloc[i, j] - e) ** 2 / e

    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


# ─────────────────────────────────────────────
# Публичные функции
# ─────────────────────────────────────────────

def compute_marginal_stats(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> Dict:
    """
    Покоменное сравнение распределений.
    Числовые → JSD + разница mean/std/median.
    Категориальные → TVD.
    """
    num_cols, cat_cols = _detect_column_types(real_df)
    results = {"numerical": {}, "categorical": {}}

    for col in num_cols:
        if col not in synth_df.columns:
            continue
        r, s = real_df[col].dropna(), synth_df[col].dropna()
        results["numerical"][col] = {
            "jsd": round(_compute_jsd(r, s), 6),
            "mean_delta": round(float(r.mean() - s.mean()), 4),
            "std_delta": round(float(r.std() - s.std()), 4),
            "median_delta": round(float(r.median() - s.median()), 4),
        }

    for col in cat_cols:
        if col not in synth_df.columns:
            continue
        results["categorical"][col] = {
            "tvd": round(_compute_tvd(real_df[col], synth_df[col]), 6),
        }

    # Сводные агрегаты для быстрого взгляда
    jsd_values = [v["jsd"] for v in results["numerical"].values()]
    tvd_values = [v["tvd"] for v in results["categorical"].values()]
    results["summary"] = {
        "mean_jsd": round(float(np.mean(jsd_values)), 6) if jsd_values else None,
        "mean_tvd": round(float(np.mean(tvd_values)), 6) if tvd_values else None,
    }

    return results


def compute_correlation_delta(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> Dict:
    """
    Сравнение матриц корреляций.
    Для числовых пар — Pearson, для категориальных — Cramér's V.
    Возвращает среднее абсолютное отклонение (MAE) между матрицами.
    """
    num_cols, cat_cols = _detect_column_types(real_df)
    common_num = [c for c in num_cols if c in synth_df.columns]
    common_cat = [c for c in cat_cols if c in synth_df.columns]

    pearson_delta = None
    if len(common_num) >= 2:
        real_corr = real_df[common_num].corr(method="pearson")
        synth_corr = synth_df[common_num].corr(method="pearson")
        # MAE между матрицами — чем меньше, тем лучше сохранены зависимости
        pearson_delta = float(
            (real_corr - synth_corr).abs().values[np.triu_indices_from(real_corr.values, k=1)].mean()
        )

    cramers_delta = None
    if len(common_cat) >= 2:
        deltas = []
        for i, col_a in enumerate(common_cat):
            for col_b in common_cat[i + 1:]:
                real_v = _cramers_v(real_df[col_a], real_df[col_b])
                synth_v = _cramers_v(synth_df[col_a], synth_df[col_b])
                deltas.append(abs(real_v - synth_v))
        cramers_delta = float(np.mean(deltas)) if deltas else None

    return {
        "pearson_corr_mae": round(pearson_delta, 6) if pearson_delta is not None else None,
        "cramers_v_mae": round(cramers_delta, 6) if cramers_delta is not None else None,
    }

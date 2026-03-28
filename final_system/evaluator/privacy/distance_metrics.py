"""
distance_metrics.py

Метрики расстояний между реальными и синтетическими записями.
Основная цель: убедиться, что генератор не "запомнил" реальные данные
и не воспроизводит их почти дословно (memorization risk).

DCR  — Distance to Closest Record: расстояние от синтетической записи до ближайшей реальной.
NNDR — Nearest Neighbor Distance Ratio: отношение расстояний до 1-го и 2-го ближайших соседей.

Интерпретация:
- Если DCR(synth→real) ≈ DCR(holdout→real), то синтетика не "ближе" к реальным данным,
  чем другие реальные данные. Это хороший знак.
- Если NNDR → 0, записи почти идентичны ближайшему соседу → риск утечки.

Кодирование признаков:
    Категориальные колонки кодируются через one-hot (pd.get_dummies), а числовые
    нормируются в [0, 1] через MinMaxScaler. Оба типа признаков оказываются
    в единой нормированной системе координат, где евклидово расстояние
    имеет корректную геометрическую интерпретацию.

    Это важно: LabelEncoder присваивает категориям целые числа (0, 1, 2, ...),
    создавая искусственный порядок и метрику между категориями. Для "Moscow",
    "SPb", "Kazan" расстояние 1 vs 2 ничем не хуже 0 vs 1 — это артефакт
    кодирования, а не семантика данных. One-hot лишён этого недостатка.

    Трансформация обучается на reference_df (реальные train-данные),
    затем применяется к query_df (синтетика или holdout). Категории,
    отсутствующие в reference, заполняются нулём через reindex.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Preprocessing для расстояний
# ─────────────────────────────────────────────

def _encode_and_normalize(
    reference_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Приводит оба датафрейма к числовому виду для расчета расстояний.

    Категориальные колонки → one-hot (pd.get_dummies, обучаем на reference).
    Числовые колонки       → MinMaxScaler [0, 1] (обучаем на reference).

    Трансформация всегда обучается на reference_df, к query_df только применяется.
    Категории из query, которых нет в reference, заполняются нулём (reindex).
    Общие колонки обоих датафреймов используются для расчета.
    """
    common_cols = [c for c in reference_df.columns if c in query_df.columns]
    ref = reference_df[common_cols].copy()
    qry = query_df[common_cols].copy()

    num_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ref.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot кодирование категориальных признаков.
    # Обучаем на reference, чтобы query получил те же колонки.
    # Категории из query, отсутствующие в reference → столбец с нулями (reindex).
    ref_dummies = pd.get_dummies(ref[cat_cols], prefix_sep="__") if cat_cols else pd.DataFrame(index=ref.index)
    qry_dummies = pd.get_dummies(qry[cat_cols], prefix_sep="__") if cat_cols else pd.DataFrame(index=qry.index)
    qry_dummies = qry_dummies.reindex(columns=ref_dummies.columns, fill_value=0)

    # Нормализация числовых признаков в [0, 1].
    # Обучаем scaler на reference, применяем к обоим — одинаковая шкала.
    if num_cols:
        scaler = MinMaxScaler()
        ref_num = pd.DataFrame(
            scaler.fit_transform(ref[num_cols].fillna(0)),
            columns=num_cols,
            index=ref.index,
        )
        qry_num = pd.DataFrame(
            scaler.transform(qry[num_cols].fillna(0)),
            columns=num_cols,
            index=qry.index,
        )
    else:
        ref_num = pd.DataFrame(index=ref.index)
        qry_num = pd.DataFrame(index=qry.index)

    ref_arr = np.hstack([ref_num.values, ref_dummies.values.astype(float)])
    qry_arr = np.hstack([qry_num.values, qry_dummies.values.astype(float)])

    return ref_arr, qry_arr


def _compute_distances_batched(
    query_arr: np.ndarray,
    reference_arr: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Вычисляет евклидово расстояние от каждой строки query до всех строк reference.
    Возвращает массив расстояний до ближайшего соседа (shape: [len(query)]).
    Батчинг нужен, чтобы не грузить RAM при больших датасетах.
    """
    n_query = query_arr.shape[0]
    min_distances = np.empty(n_query)

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        batch = query_arr[start:end]  # [batch, features]

        # Векторизованный расчет расстояний: [batch, n_reference]
        diff = batch[:, np.newaxis, :] - reference_arr[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        min_distances[start:end] = dists.min(axis=1)

    return min_distances


def _compute_nndr_batched(
    query_arr: np.ndarray,
    reference_arr: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Nearest Neighbor Distance Ratio: dist_1st / dist_2nd.
    Если NNDR → 0, запись почти совпадает с ближайшим соседом.
    Если NNDR → 1, первый и второй сосед одинаково далеки — запись уникальна.
    """
    n_query = query_arr.shape[0]
    nndr_values = np.empty(n_query)

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        batch = query_arr[start:end]

        diff = batch[:, np.newaxis, :] - reference_arr[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        # Сортируем и берем двух ближайших соседей
        dists.sort(axis=1)
        d1 = dists[:, 0]
        d2 = dists[:, 1]

        # Защита от деления на 0: если d2=0, NNDR считаем 0
        with np.errstate(divide='ignore', invalid='ignore'):
            nndr = np.where(d2 > 0, d1 / d2, 0.0)

        nndr_values[start:end] = nndr

    return nndr_values


# ─────────────────────────────────────────────
# Публичная функция
# ─────────────────────────────────────────────

def compute_distance_metrics(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    sample_size: Optional[int] = 2000,
) -> Dict:
    """
    Считает DCR и NNDR для синтетики и для holdout-выборки реальных данных.

    Ключевая идея сравнения:
    - DCR_synth: расстояния от синтетических записей до реального train-сета.
    - DCR_holdout: расстояния от holdout-записей до того же train-сета.
    - Если медиана DCR_synth >= медианы DCR_holdout, синтетика не "ближе"
      к обучающим данным, чем отложенные реальные данные. Это целевое поведение.

    sample_size: ограничиваем выборку для скорости (None = весь датасет).
    """
    # Сэмплируем для ускорения на больших датасетах
    if sample_size and len(synth_df) > sample_size:
        synth_sample = synth_df.sample(sample_size, random_state=42)
    else:
        synth_sample = synth_df

    if sample_size and len(real_holdout_df) > sample_size:
        holdout_sample = real_holdout_df.sample(sample_size, random_state=42)
    else:
        holdout_sample = real_holdout_df

    logger.info(
        f"[distance] Кодирование и нормализация признаков... "
        f"train={len(real_train_df)}, synth={len(synth_sample)}, holdout={len(holdout_sample)}"
    )

    # Кодируем синтетику и holdout относительно real_train.
    # real_train — эталон: на нём обучаем scaler и one-hot словарь,
    # затем применяем ту же трансформацию к synth и holdout.
    ref_arr, synth_arr = _encode_and_normalize(real_train_df, synth_sample)
    _, holdout_arr = _encode_and_normalize(real_train_df, holdout_sample)

    # DCR
    logger.info("[distance] Считаем DCR (synth → real_train)...")
    dcr_synth = _compute_distances_batched(synth_arr, ref_arr)

    logger.info("[distance] Считаем DCR (holdout → real_train)...")
    dcr_holdout = _compute_distances_batched(holdout_arr, ref_arr)

    # NNDR
    logger.info("[distance] Считаем NNDR (synth → real_train)...")
    nndr_synth = _compute_nndr_batched(synth_arr, ref_arr)

    # Интерпретация: если синтетика не ближе к обучающим данным, чем holdout — всё ок
    dcr_synth_median = float(np.median(dcr_synth))
    dcr_holdout_median = float(np.median(dcr_holdout))
    privacy_preserved = dcr_synth_median >= dcr_holdout_median

    return {
        "dcr": {
            "synth_to_real": {
                "min":    round(float(dcr_synth.min()), 6),
                "median": round(dcr_synth_median, 6),
                "mean":   round(float(dcr_synth.mean()), 6),
                "p5":     round(float(np.percentile(dcr_synth, 5)), 6),
            },
            "holdout_to_real": {
                "min":    round(float(dcr_holdout.min()), 6),
                "median": round(dcr_holdout_median, 6),
                "mean":   round(float(dcr_holdout.mean()), 6),
                "p5":     round(float(np.percentile(dcr_holdout, 5)), 6),
            },
            "privacy_preserved": privacy_preserved,
            "interpretation": (
                "ok: синтетика не ближе к обучающим данным, чем holdout"
                if privacy_preserved
                else "risk: синтетика аномально близка к обучающим данным"
            ),
        },
        "nndr": {
            "synth_mean":      round(float(nndr_synth.mean()), 6),
            "synth_median":    round(float(np.median(nndr_synth)), 6),
            "share_below_0.1": round(float((nndr_synth < 0.1).mean()), 6),
        },
    }
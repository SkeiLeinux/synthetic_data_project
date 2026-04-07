"""
attack_simulation.py

Симуляция Membership Inference Attack (MIA) на генеративную модель.

Логика атаки (distance-based proxy MIA):
1. Разбиваем реальные данные на train (использовался для обучения генератора) и holdout.
2. Для каждой записи из train и holdout считаем DCR — расстояние до ближайшей
   синтетической записи.
3. Обучаем атакующий классификатор: "эта запись была в train или нет?"
   Признак атаки — DCR. Гипотеза: если запись была в train, генератор мог
   "запомнить" её → она окажется ближе к синтетике → меньше DCR.
4. Если AUC атакующей модели ≈ 0.5 → DP работает, генератор не запоминает данные.
   Если AUC >> 0.5 → есть риск утечки.

Ограничение: это упрощённая прокси-реализация (distance-based proxy MIA),
а не полноценная shadow-model атака по методу Shokri et al. (2017).
Полная shadow-model MIA требует обучения нескольких «теневых» моделей на
отдельных независимых датасетах — это выходит за рамки данного инструмента.
Proxy MIA даёт консервативную нижнюю оценку риска: если AUC высокий даже
здесь, это сильный сигнал меморизации. Низкий AUC — необходимое, но не
достаточное условие отсутствия утечки.

Кодирование признаков:
    Используется та же схема, что и в distance_metrics.py:
    категориальные → one-hot (обучаем на real_train_df),
    числовые → MinMaxScaler [0, 1].
    Это гарантирует, что расстояния в признаковом пространстве не искажены
    артефактами порядкового кодирования LabelEncoder.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def _encode(df: pd.DataFrame, reference_df: pd.DataFrame) -> np.ndarray:
    """
    Кодирует df в числовой массив, используя словарь признаков из reference_df.

    Категориальные колонки → one-hot (pd.get_dummies, обучаем на reference).
    Числовые колонки       → MinMaxScaler [0, 1] (обучаем на reference).

    Категории из df, отсутствующие в reference, заполняются нулём (reindex).
    Этот метод намеренно дублирует логику distance_metrics._encode_and_normalize,
    чтобы attack_simulation оставался самодостаточным модулем.
    """
    common = [c for c in reference_df.columns if c in df.columns]
    ref = reference_df[common].copy()
    qry = df[common].copy()

    num_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ref.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot кодирование категориальных признаков
    ref_dummies = pd.get_dummies(ref[cat_cols], prefix_sep="__") if cat_cols else pd.DataFrame(index=ref.index)
    qry_dummies = pd.get_dummies(qry[cat_cols], prefix_sep="__") if cat_cols else pd.DataFrame(index=qry.index)
    qry_dummies = qry_dummies.reindex(columns=ref_dummies.columns, fill_value=0)

    # Нормализация числовых признаков
    if num_cols:
        scaler = MinMaxScaler()
        ref_num = scaler.fit_transform(ref[num_cols].fillna(0))
        qry_num = scaler.transform(qry[num_cols].fillna(0))
    else:
        ref_num = np.empty((len(ref), 0))
        qry_num = np.empty((len(qry), 0))

    return np.hstack([qry_num, qry_dummies.values.astype(float)])


def _compute_min_distances_to_synth(
    query_arr: np.ndarray,
    synth_arr: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """Расстояние от каждой реальной записи до ближайшей синтетической."""
    n = query_arr.shape[0]
    min_dists = np.empty(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = query_arr[start:end]
        diff = batch[:, np.newaxis, :] - synth_arr[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        min_dists[start:end] = dists.min(axis=1)
    return min_dists


def evaluate_membership_inference(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
    sample_size: int = 1000,
) -> Dict:
    """
    Запускает proxy MIA и возвращает метрики атаки.

    Параметры:
        real_train_df   — данные, на которых обучался генератор (метка: 1 = "в train")
        real_holdout_df — данные, которые генератор НЕ видел  (метка: 0 = "не в train")
        synth_df        — синтетические данные от генератора
        sample_size     — ограничение выборки для скорости

    Кодирование выполняется относительно real_train_df — это эталон
    признакового пространства для всех трёх датасетов.

    Интерпретация результатов:
        attack_auc ≈ 0.5 → атака не работает, генератор защищён (хороший результат для DP)
        attack_auc > 0.7 → атака эффективна, есть риск утечки membership info
    """
    # Сэмплируем для баланса и скорости
    n = min(sample_size, len(real_train_df), len(real_holdout_df))
    train_sample   = real_train_df.sample(n, random_state=random_state)
    holdout_sample = real_holdout_df.sample(n, random_state=random_state)

    if len(synth_df) > sample_size:
        synth_sample = synth_df.sample(sample_size, random_state=random_state)
    else:
        synth_sample = synth_df

    logger.info(
        f"[MIA] Запуск атаки. train_members={n}, non_members={n}, synth={len(synth_sample)}"
    )

    # Кодируем все три датасета относительно real_train_df.
    # Важно: один и тот же эталон признакового пространства для всех —
    # только так расстояния между train/holdout/synth сопоставимы.
    synth_encoded   = _encode(synth_sample,   real_train_df)
    train_encoded   = _encode(train_sample,   real_train_df)
    holdout_encoded = _encode(holdout_sample, real_train_df)

    # Признак атаки: расстояние от реальной записи до ближайшей синтетической.
    # Гипотеза: train-записи "отпечатались" в синтетике → меньше расстояние.
    dist_train   = _compute_min_distances_to_synth(train_encoded,   synth_encoded)
    dist_holdout = _compute_min_distances_to_synth(holdout_encoded, synth_encoded)

    # Формируем датасет для атакующего классификатора
    X_attack = np.concatenate([dist_train, dist_holdout]).reshape(-1, 1)
    y_attack = np.concatenate([np.ones(n), np.zeros(n)])

    # Атакующий классификатор: простой RF на одном признаке (DCR).
    # Кросс-валидация даёт более честную оценку, чем простой train/test.
    attacker = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(attacker, X_attack, y_attack, cv=5, scoring="roc_auc")
    attack_auc = float(np.mean(cv_scores))

    logger.info(
        f"[MIA] Attack AUC = {attack_auc:.4f} "
        f"(std={np.std(cv_scores):.4f}). "
        f"{'Защита эффективна' if attack_auc < 0.6 else 'РИСК: атака эффективна'}"
    )

    return {
        "attack_auc": round(attack_auc, 4),
        "attack_auc_std": round(float(np.std(cv_scores)), 4),
        "interpretation": (
            "protected: атака не лучше случайного угадывания (AUC < 0.6)"
            if attack_auc < 0.6
            else "warning: атака частично эффективна (0.6 ≤ AUC < 0.75)"
            if attack_auc < 0.75
            else "risk: высокая эффективность атаки (AUC ≥ 0.75)"
        ),
        "n_members_tested": n,
        "n_non_members_tested": n,
        "note": (
            "Distance-based proxy MIA. "
            "AUC ≈ 0.5 означает отсутствие утечки membership-информации. "
            "Это консервативная нижняя оценка риска — не полная shadow-model MIA."
        ),
    }
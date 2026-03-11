"""
attack_simulation.py

Симуляция Membership Inference Attack (MIA) на генеративную модель.

Логика атаки (Shadow Model approach, упрощенная версия):
1. Разбиваем реальные данные на train (использовался для обучения генератора) и holdout.
2. Обучаем атакующий классификатор: "эта запись была в обучающих данных или нет?"
3. Признаки для атаки: DCR (расстояние до ближайшей синтетической записи).
   Гипотеза: если запись была в train, она "оставляет след" в синтетике → меньше DCR.
4. Если точность атакующей модели ≈ 0.5 → DP работает, генератор не запоминает данные.
   Если точность >> 0.5 → есть риск утечки.

Ограничение: это упрощенная (proxy) MIA, а не полная shadow-model атака.
Она дает нижнюю оценку риска. Достаточна для дипломной работы и соответствует
требованиям ПНСТ к практической оценке рисков (симуляция атак).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


def _encode(df: pd.DataFrame, reference_df: pd.DataFrame) -> np.ndarray:
    """Кодирует df используя словарь из reference_df (аналог distance_metrics)."""
    common = [c for c in reference_df.columns if c in df.columns]
    ref = reference_df[common].copy()
    qry = df[common].copy()

    for col in ref.select_dtypes(exclude=[np.number]).columns:
        le = LabelEncoder()
        combined = pd.concat([ref[col], qry[col]], axis=0).astype(str)
        le.fit(combined)
        ref[col] = le.transform(ref[col].astype(str))
        qry[col] = le.transform(qry[col].astype(str))

    scaler = MinMaxScaler()
    scaler.fit(ref.select_dtypes(include=[np.number]).fillna(0))
    return scaler.transform(qry.select_dtypes(include=[np.number]).fillna(0))


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


def run_membership_inference(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
    sample_size: int = 1000,
) -> Dict:
    """
    Запускает упрощенную MIA и возвращает метрики атаки.

    Параметры:
        real_train_df   — данные, на которых обучался генератор (метка: 1 = "в train")
        real_holdout_df — данные, которые генератор НЕ видел (метка: 0 = "не в train")
        synth_df        — синтетические данные от генератора
        sample_size     — ограничение выборки для скорости

    Интерпретация результатов:
        attack_auc ≈ 0.5 → атака не работает, генератор защищен (хороший результат для DP)
        attack_auc > 0.7 → атака эффективна, есть риск утечки membership info
    """
    # Сэмплируем для баланса и скорости
    n = min(sample_size, len(real_train_df), len(real_holdout_df))
    train_sample = real_train_df.sample(n, random_state=random_state)
    holdout_sample = real_holdout_df.sample(n, random_state=random_state)

    if len(synth_df) > sample_size:
        synth_sample = synth_df.sample(sample_size, random_state=random_state)
    else:
        synth_sample = synth_df

    logger.info(
        f"[MIA] Запуск атаки. train_members={n}, non_members={n}, synth={len(synth_sample)}"
    )

    # Кодируем синтетику относительно train (эталон для признакового пространства)
    ref_encoded = _encode(real_train_df, real_train_df)
    synth_encoded = _encode(synth_sample, real_train_df)
    train_encoded = _encode(train_sample, real_train_df)
    holdout_encoded = _encode(holdout_sample, real_train_df)

    # Признак атаки: расстояние от реальной записи до ближайшей синтетической.
    # Гипотеза: train-записи "отпечатались" в синтетике → меньше расстояние.
    dist_train = _compute_min_distances_to_synth(train_encoded, synth_encoded)
    dist_holdout = _compute_min_distances_to_synth(holdout_encoded, synth_encoded)

    # Формируем датасет для атакующего классификатора
    X_attack = np.concatenate([dist_train, dist_holdout]).reshape(-1, 1)
    y_attack = np.concatenate([np.ones(n), np.zeros(n)])

    # Атакующий классификатор: простой RF на одном признаке (DCR)
    attacker = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    # Кросс-валидация дает более честную оценку, чем простой train/test
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
            "Proxy MIA на основе DCR. "
            "AUC ≈ 0.5 означает отсутствие утечки membership-информации."
        ),
    }

"""
ml_efficacy.py

Оценка ML-полезности синтетических данных через TSTR и TRTR.

Схема:
  TRTR: обучаем на real_train → тестируем на real_test  (потолок качества)
  TSTR: обучаем на synth      → тестируем на real_test  (качество синтетики)
  Utility Loss = TRTR_score - TSTR_score  (чем меньше, тем лучше)

ВАЖНО: real_train, real_test и synth_df передаются снаружи — разделение
на holdout выполняется в run_pipeline() до обучения генератора, а не здесь.
Это принципиально: генератор не должен видеть real_test ни на каком этапе.

Поддерживаемые задачи: classification (F1, ROC-AUC), regression (MAE, R²).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "regression"]


@dataclass
class MLEfficacyConfig:
    """Конфигурация ML-оценки полезности."""
    target_column: str
    task_type: TaskType = "classification"

    # Random Forest — стабильная baseline-модель для сравнения.
    # Не требует нормализации и хорошо работает "из коробки".
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42

    # Дополнительные колонки, которые нужно исключить из признаков
    drop_columns: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
) -> tuple:
    """
    Готовит X_train, y_train, X_test, y_test.
    Категориальные признаки кодируются через LabelEncoder.
    Пропуски заполняются медианой (числовые) или модой (категориальные).
    """
    exclude = set(drop_cols + [target])
    feature_cols = [c for c in train_df.columns if c not in exclude and c in test_df.columns]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target].copy()
    y_test = test_df[target].copy()

    # Кодирование категориальных признаков
    for col in X_train.select_dtypes(exclude=[np.number]).columns:
        le = LabelEncoder()
        # Обучаем на объединенных данных, чтобы тест не содержал unseen-меток
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Заполнение пропусков: проверяем оба датафрейма.
    # fill_val всегда берём из X_train, чтобы не допускать утечки test-статистик.
    for col in X_train.columns:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                fill_val = X_train[col].median()
            else:
                mode = X_train[col].mode()
                fill_val = mode[0] if len(mode) > 0 else ""
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

    return X_train, y_train, X_test, y_test


def _compute_scores(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: TaskType,
) -> Dict:
    """Вычисляет метрики для конкретного предсказания."""
    if task_type == "classification":
        scores = {
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        }
        # ROC-AUC считаем только если есть вероятности (бинарная или мультикласс OvR)
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    scores["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
                else:
                    scores["roc_auc"] = round(float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )), 4)
            except Exception as e:
                logger.warning(f"[MLEfficacy] ROC-AUC не посчитан: {e}")
    else:
        scores = {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
    return scores


def _run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: MLEfficacyConfig,
    label: str,
) -> Dict:
    """
    Запускает один эксперимент: обучение на train_df, тест на test_df.
    label — имя для логов ('TRTR' или 'TSTR').
    """
    X_train, y_train, X_test, y_test = _prepare_features(
        train_df, test_df, config.target_column, config.drop_columns
    )

    if config.task_type == "classification":
        # Кодируем целевую переменную, если она строковая
        if y_train.dtype == object:
            le_target = LabelEncoder()
            y_train = le_target.fit_transform(y_train.astype(str))
            y_test = le_target.transform(y_test.astype(str))
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if config.task_type == "classification" else None

    scores = _compute_scores(y_test, y_pred, y_proba, config.task_type)
    logger.info(f"[MLEfficacy] {label}: {scores}")
    return scores


# ─────────────────────────────────────────────
# Публичная функция
# ─────────────────────────────────────────────

def evaluate_ml_efficacy(
    real_train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: MLEfficacyConfig,
) -> Dict:
    """
    Запускает TRTR и TSTR и считает Utility Loss.

    Разделение реальных данных на train/test выполняется в run_pipeline()
    до обучения генератора и передаётся сюда готовым:
        real_train_df — обучающая часть реальных данных (генератор её видел)
        synth_df      — синтетические данные от генератора
        real_test_df  — отложенная тестовая выборка (генератор её НЕ видел)

    TRTR обучается на real_train_df, тестируется на real_test_df.
    TSTR обучается на synth_df, тестируется на том же real_test_df.
    real_test_df — единый holdout для обоих экспериментов.
    """
    logger.info(
        f"[MLEfficacy] real_train={len(real_train_df)}, "
        f"synth_train={len(synth_df)}, real_test={len(real_test_df)}"
    )

    trtr_scores = _run_single_experiment(real_train_df, real_test_df, config, label="TRTR")
    tstr_scores = _run_single_experiment(synth_df, real_test_df, config, label="TSTR")

    # Utility Loss: по основной метрике (f1 или r2).
    # Положительное значение → синтетика хуже реальных данных.
    # Отрицательное → синтетика случайно "лучше" (бывает на малых выборках).
    main_metric = "f1_weighted" if config.task_type == "classification" else "r2"
    utility_loss = None
    if main_metric in trtr_scores and main_metric in tstr_scores:
        utility_loss = round(trtr_scores[main_metric] - tstr_scores[main_metric], 4)

    return {
        "trtr": trtr_scores,
        "tstr": tstr_scores,
        "utility_loss": {
            "metric": main_metric,
            "value": utility_loss,
            "interpretation": (
                "ok (loss < 10%)" if utility_loss is not None and abs(utility_loss) < 0.1
                else "degraded" if utility_loss is not None
                else "unknown"
            ),
        },
    }
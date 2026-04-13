# synthesizer/sdv_generators.py
#
# Non-DP генераторы на базе SDV: CTGAN, TVAE, CopulaGAN.
# Используются как baseline для сравнения с DP-версиями.
#
# Зависимости:
#   pip install sdv

from __future__ import annotations

import importlib.metadata
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    TVAESynthesizer,
)

from synthesizer.base import BaseGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция: построение SDV Metadata из схемы колонок
# ──────────────────────────────────────────────────────────────────────────────

def _build_metadata(
    df: pd.DataFrame,
    categorical_columns: List[str],
    continuous_columns: List[str],
    table_name: str = "table",
) -> SingleTableMetadata:
    """
    Строит SDV SingleTableMetadata из явно заданных списков колонок.

    SDV Metadata определяет тип каждой колонки:
      - categorical → sdtype='categorical'
      - continuous  → sdtype='numerical'
      - остальные   → автодетекция по dtype

    В SDV 1.x используется SingleTableMetadata (не Metadata/MultiTableMetadata).
    """
    metadata = SingleTableMetadata()

    cat_set = set(categorical_columns)
    cont_set = set(continuous_columns)

    for col in df.columns:
        if col in cat_set:
            metadata.add_column(col, sdtype="categorical")
        elif col in cont_set:
            metadata.add_column(col, sdtype="numerical")
        else:
            metadata.add_column(
                col,
                sdtype="categorical" if df[col].dtype == object else "numerical",
            )

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
# CTGAN
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CTGANConfig:
    """Конфигурация CTGAN (SDV) без дифференциальной приватности."""
    epochs: int = 300
    batch_size: int = 500
    discriminator_steps: int = 1
    pac: int = 10
    embedding_dim: int = 128
    generator_dim: Sequence[int] = field(default_factory=lambda: (256, 256))
    discriminator_dim: Sequence[int] = field(default_factory=lambda: (256, 256))
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    log_frequency: bool = True
    cuda: bool = True
    verbose: bool = True
    random_seed: Optional[int] = 42


class CTGANGenerator(BaseGenerator):
    """
    Обёртка над SDV CTGANSynthesizer без DP.

    Используется как качественный baseline для сравнения с DP-CTGAN:
    при одинаковой архитектуре разница в качестве показывает DP-penalty.
    """

    def __init__(self, config: CTGANConfig) -> None:
        self.config = config
        self._synth: Optional[CTGANSynthesizer] = None
        self._table_name = "table"
        self._sample_size: Optional[int] = None
        self._fit_duration_sec: Optional[float] = None
        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        if self.config.random_seed is not None:
            import numpy as np
            import random
            import torch
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        self._sample_size = len(data)
        metadata = _build_metadata(
            data,
            categorical_columns or [],
            continuous_columns or [],
            self._table_name,
        )

        self._synth = CTGANSynthesizer(
            metadata,
            embedding_dim=self.config.embedding_dim,
            generator_dim=tuple(self.config.generator_dim),
            discriminator_dim=tuple(self.config.discriminator_dim),
            generator_lr=self.config.generator_lr,
            generator_decay=self.config.generator_decay,
            discriminator_lr=self.config.discriminator_lr,
            discriminator_decay=self.config.discriminator_decay,
            batch_size=self.config.batch_size,
            discriminator_steps=self.config.discriminator_steps,
            log_frequency=self.config.log_frequency,
            verbose=self.config.verbose,
            epochs=self.config.epochs,
            pac=self.config.pac,
            cuda=self.config.cuda,
        )

        logger.info(
            f"[CTGAN] Обучение: epochs={self.config.epochs}, "
            f"batch={self.config.batch_size}, rows={self._sample_size}"
        )
        t0 = time.monotonic()
        self._synth.fit(data)
        self._fit_duration_sec = time.monotonic() - t0
        self._is_fitted = True
        logger.info(f"[CTGAN] Готово за {self._fit_duration_sec:.1f}с.")

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")
        return self._synth.sample(num_rows=n_rows)

    def privacy_report(self) -> Dict[str, Any]:
        return {
            "synthesizer": "sdv_ctgan",
            "sdv_version": _get_version("sdv"),
            "status": "fitted" if self._is_fitted else "not_fitted",
            "dp_config": None,
            "dp_spent": None,
            "dp_guarantees": None,
            "data": {"sample_size": self._sample_size},
            "training": {
                "fit_duration_sec": (
                    round(self._fit_duration_sec, 2)
                    if self._fit_duration_sec else None
                ),
            },
        }

    def save(self, path: str) -> None:
        self._pickle_save(path, {
            "config": self.config,
            "synth": self._synth,
            "sample_size": self._sample_size,
            "fit_duration_sec": self._fit_duration_sec,
            "is_fitted": self._is_fitted,
            "table_name": self._table_name,
        })
        logger.info(f"[CTGAN] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "CTGANGenerator":
        payload = cls._pickle_load(path)
        obj = cls(config=payload["config"])
        obj._synth = payload["synth"]
        obj._sample_size = payload.get("sample_size")
        obj._fit_duration_sec = payload.get("fit_duration_sec")
        obj._is_fitted = payload["is_fitted"]
        obj._table_name = payload.get("table_name", "table")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# TVAE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TVAEConfig:
    """Конфигурация TVAE (SDV) без дифференциальной приватности."""
    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    compress_dims: Sequence[int] = field(default_factory=lambda: (128, 128))
    decompress_dims: Sequence[int] = field(default_factory=lambda: (128, 128))
    l2scale: float = 1e-5
    loss_factor: int = 2
    cuda: bool = True
    verbose: bool = True
    random_seed: Optional[int] = 42


class TVAEGenerator(BaseGenerator):
    """
    Обёртка над SDV TVAESynthesizer без DP.

    TVAE использует вариационный автоэнкодер вместо GAN, что даёт
    более стабильное обучение, но обычно чуть хуже воспроизводит
    распределение числовых признаков.
    """

    def __init__(self, config: TVAEConfig) -> None:
        self.config = config
        self._synth: Optional[TVAESynthesizer] = None
        self._table_name = "table"
        self._sample_size: Optional[int] = None
        self._fit_duration_sec: Optional[float] = None
        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        if self.config.random_seed is not None:
            import numpy as np
            import torch
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        self._sample_size = len(data)
        metadata = _build_metadata(
            data,
            categorical_columns or [],
            continuous_columns or [],
            self._table_name,
        )

        self._synth = TVAESynthesizer(
            metadata,
            embedding_dim=self.config.embedding_dim,
            compress_dims=tuple(self.config.compress_dims),
            decompress_dims=tuple(self.config.decompress_dims),
            l2scale=self.config.l2scale,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            loss_factor=self.config.loss_factor,
            cuda=self.config.cuda,
            verbose=self.config.verbose,
        )

        logger.info(
            f"[TVAE] Обучение: epochs={self.config.epochs}, "
            f"batch={self.config.batch_size}, rows={self._sample_size}"
        )
        t0 = time.monotonic()
        self._synth.fit(data)
        self._fit_duration_sec = time.monotonic() - t0
        self._is_fitted = True
        logger.info(f"[TVAE] Готово за {self._fit_duration_sec:.1f}с.")

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")
        return self._synth.sample(num_rows=n_rows)

    def privacy_report(self) -> Dict[str, Any]:
        return {
            "synthesizer": "sdv_tvae",
            "sdv_version": _get_version("sdv"),
            "status": "fitted" if self._is_fitted else "not_fitted",
            "dp_config": None,
            "dp_spent": None,
            "dp_guarantees": None,
            "data": {"sample_size": self._sample_size},
            "training": {
                "fit_duration_sec": (
                    round(self._fit_duration_sec, 2)
                    if self._fit_duration_sec else None
                ),
            },
        }

    def save(self, path: str) -> None:
        self._pickle_save(path, {
            "config": self.config,
            "synth": self._synth,
            "sample_size": self._sample_size,
            "fit_duration_sec": self._fit_duration_sec,
            "is_fitted": self._is_fitted,
            "table_name": self._table_name,
        })
        logger.info(f"[TVAE] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "TVAEGenerator":
        payload = cls._pickle_load(path)
        obj = cls(config=payload["config"])
        obj._synth = payload["synth"]
        obj._sample_size = payload.get("sample_size")
        obj._fit_duration_sec = payload.get("fit_duration_sec")
        obj._is_fitted = payload["is_fitted"]
        obj._table_name = payload.get("table_name", "table")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# CopulaGAN
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CopulaGANConfig:
    """
    Конфигурация CopulaGAN (SDV) без дифференциальной приватности.

    CopulaGAN = CTGAN + предварительная нормализация числовых колонок
    через гауссовые копулы, что улучшает воспроизведение маргинальных
    распределений числовых признаков по сравнению с CTGAN.
    """
    epochs: int = 300
    batch_size: int = 500
    discriminator_steps: int = 1
    pac: int = 10
    embedding_dim: int = 128
    generator_dim: Sequence[int] = field(default_factory=lambda: (256, 256))
    discriminator_dim: Sequence[int] = field(default_factory=lambda: (256, 256))
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    log_frequency: bool = True
    cuda: bool = True
    verbose: bool = True
    random_seed: Optional[int] = 42


class CopulaGANGenerator(BaseGenerator):
    """
    Обёртка над SDV CopulaGANSynthesizer без DP.
    """

    def __init__(self, config: CopulaGANConfig) -> None:
        self.config = config
        self._synth: Optional[CopulaGANSynthesizer] = None
        self._table_name = "table"
        self._sample_size: Optional[int] = None
        self._fit_duration_sec: Optional[float] = None
        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        if self.config.random_seed is not None:
            import numpy as np
            import torch
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        self._sample_size = len(data)
        metadata = _build_metadata(
            data,
            categorical_columns or [],
            continuous_columns or [],
            self._table_name,
        )

        self._synth = CopulaGANSynthesizer(
            metadata,
            embedding_dim=self.config.embedding_dim,
            generator_dim=tuple(self.config.generator_dim),
            discriminator_dim=tuple(self.config.discriminator_dim),
            generator_lr=self.config.generator_lr,
            generator_decay=self.config.generator_decay,
            discriminator_lr=self.config.discriminator_lr,
            discriminator_decay=self.config.discriminator_decay,
            batch_size=self.config.batch_size,
            discriminator_steps=self.config.discriminator_steps,
            log_frequency=self.config.log_frequency,
            verbose=self.config.verbose,
            epochs=self.config.epochs,
            pac=self.config.pac,
            cuda=self.config.cuda,
        )

        logger.info(
            f"[CopulaGAN] Обучение: epochs={self.config.epochs}, "
            f"batch={self.config.batch_size}, rows={self._sample_size}"
        )
        t0 = time.monotonic()
        self._synth.fit(data)
        self._fit_duration_sec = time.monotonic() - t0
        self._is_fitted = True
        logger.info(f"[CopulaGAN] Готово за {self._fit_duration_sec:.1f}с.")

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")
        return self._synth.sample(num_rows=n_rows)

    def privacy_report(self) -> Dict[str, Any]:
        return {
            "synthesizer": "sdv_copulagan",
            "sdv_version": _get_version("sdv"),
            "status": "fitted" if self._is_fitted else "not_fitted",
            "dp_config": None,
            "dp_spent": None,
            "dp_guarantees": None,
            "data": {"sample_size": self._sample_size},
            "training": {
                "fit_duration_sec": (
                    round(self._fit_duration_sec, 2)
                    if self._fit_duration_sec else None
                ),
            },
        }

    def save(self, path: str) -> None:
        self._pickle_save(path, {
            "config": self.config,
            "synth": self._synth,
            "sample_size": self._sample_size,
            "fit_duration_sec": self._fit_duration_sec,
            "is_fitted": self._is_fitted,
            "table_name": self._table_name,
        })
        logger.info(f"[CopulaGAN] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "CopulaGANGenerator":
        payload = cls._pickle_load(path)
        obj = cls(config=payload["config"])
        obj._synth = payload["synth"]
        obj._sample_size = payload.get("sample_size")
        obj._fit_duration_sec = payload.get("fit_duration_sec")
        obj._is_fitted = payload["is_fitted"]
        obj._table_name = payload.get("table_name", "table")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _get_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

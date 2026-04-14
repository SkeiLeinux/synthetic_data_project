# synthesizer/dp_tvae.py
#
# DP-TVAE генератор: Opacus (DP-SGD) поверх TVAE (вариационного автоэнкодера).
#
# Архитектура:
#   - Препроцессинг данных через ctgan.DataTransformer
#     (аналогично тому, как это делает SDV TVAE внутри)
#   - Encoder / Decoder берутся из ctgan.synthesizers.tvae
#   - Объединяются в один nn.Module для совместимости с Opacus
#   - PrivacyEngine.make_private_with_epsilon() обеспечивает DP-SGD
#   - Постпроцессинг (inverse_transform) возвращает исходное пространство признаков
#
# Ключевое отличие от подхода в archive/generator.py:
#   Там DP применялся post-hoc к дискриминатору после полного обучения GAN.
#   Здесь DP применяется с самого начала к полной TVAE-модели (encoder + decoder),
#   что обеспечивает формальные DP-гарантии с первой итерации.
#
# Зависимости:
#   pip install ctgan opacus torch

from __future__ import annotations

import importlib.metadata
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from synthesizer.base import BaseGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DPTVAEConfig:
    """
    Конфигурация DP-TVAE генератора.

    Параметры приватности (DP) — аналогичны DPCTGANConfig:
        sigma            -- noise_multiplier для DP-SGD; больше = строже DP, хуже качество
                            Типичные значения: 1.0–5.0. Аналог sigma в DP-CTGAN.
        delta            -- δ-вероятность нарушения DP-гарантии;
                            None = авторасчёт как 1/(n * sqrt(n))
        max_grad_norm    -- клиппинг нормы градиентов (C в DP-SGD)

    Примечание о epsilon:
        В Opacus 0.14 нельзя задать целевой epsilon напрямую — задаётся sigma.
        Фактически потраченный epsilon вычисляется ПОСЛЕ обучения через
        privacy_engine.get_privacy_spent(delta) и сохраняется в privacy_report().

    Параметры архитектуры:
        compress_dims    -- размеры скрытых слоёв энкодера
        decompress_dims  -- размеры скрытых слоёв декодера
        embedding_dim    -- размер латентного пространства
    """
    # DP-параметры
    sigma: float = 5.0          # noise_multiplier (аналог sigma в DP-CTGAN)
    delta: Optional[float] = None
    max_grad_norm: float = 1.0

    # Архитектура
    embedding_dim: int = 128
    compress_dims: Sequence[int] = (128, 128)
    decompress_dims: Sequence[int] = (128, 128)

    # Обучение
    epochs: int = 300
    batch_size: int = 500
    l2scale: float = 1e-5
    loss_factor: int = 2

    # Режимы
    cuda: bool = True
    random_seed: Optional[int] = 42


# ──────────────────────────────────────────────────────────────────────────────
# Архитектура TVAE (адаптировано из ctgan.synthesizers.tvae)
# ──────────────────────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    """Энкодер VAE: отображает x → (mu, logvar) в латентное пространство."""

    def __init__(self, data_dim: int, compress_dims: Sequence[int], embedding_dim: int) -> None:
        super().__init__()
        layers = []
        prev_dim = data_dim
        for dim in compress_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        self.seq = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, embedding_dim)
        self.fc_logvar = nn.Linear(prev_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.seq(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Декодер VAE: отображает z → реконструкцию в пространство данных."""

    def __init__(self, embedding_dim: int, decompress_dims: Sequence[int], data_dim: int) -> None:
        super().__init__()
        layers = []
        prev_dim = embedding_dim
        for dim in decompress_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        layers += [nn.Linear(prev_dim, data_dim)]
        self.seq = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.seq(z)


class _VAE(nn.Module):
    """
    Объединённый VAE-модуль для совместимости с Opacus.

    Opacus требует единый nn.Module с единым оптимизатором.
    Encoder и Decoder объединены в один модуль, чтобы DP-гарантии
    распространялись на всю модель.
    """

    def __init__(
        self,
        data_dim: int,
        compress_dims: Sequence[int],
        decompress_dims: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = _Encoder(data_dim, compress_dims, embedding_dim)
        self.decoder = _Decoder(embedding_dim, decompress_dims, data_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# ELBO Loss (аналог ctgan TVAE loss)
# ──────────────────────────────────────────────────────────────────────────────

def _elbo_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    output_info: List[Any],
    loss_factor: float,
) -> torch.Tensor:
    """
    Evidence Lower BOund (ELBO) для смешанных данных.

    reconstruction_loss:
        - Категориальные колонки: cross-entropy (после softmax из DataTransformer)
        - Числовые колонки: MSE * loss_factor

    kl_divergence:
        KL(N(mu, std) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    output_info — список SpanInfo из ctgan.DataTransformer, описывает
    структуру трансформированных данных (сколько колонок на каждую переменную).
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    recon_loss = torch.tensor(0.0, device=recon.device)
    col_idx = 0
    for span_info_list in output_info:
        for span_info in span_info_list:
            n_dims = span_info.dim
            activation = span_info.activation_fn

            cols = slice(col_idx, col_idx + n_dims)
            if activation == "softmax":
                # Категориальная переменная: log_softmax + NLL
                log_prob = torch.nn.functional.log_softmax(recon[:, cols], dim=1)
                target = x[:, cols].argmax(dim=1)
                recon_loss += torch.nn.functional.nll_loss(log_prob, target, reduction="sum")
            elif activation == "tanh":
                # Числовая переменная (в режиме continuous): MSE
                recon_loss += loss_factor * torch.nn.functional.mse_loss(
                    recon[:, cols], x[:, cols], reduction="sum"
                )
            else:
                # Fallback: MSE
                recon_loss += torch.nn.functional.mse_loss(
                    recon[:, cols], x[:, cols], reduction="sum"
                )

            col_idx += n_dims

    return (kl + recon_loss) / len(x)


# ──────────────────────────────────────────────────────────────────────────────
# Генератор
# ──────────────────────────────────────────────────────────────────────────────

class DPTVAEGenerator(BaseGenerator):
    """
    DP-TVAE генератор: Opacus (DP-SGD) поверх вариационного автоэнкодера.

    Публичный интерфейс совместим с BaseGenerator и DPCTGANGenerator:
        fit()            -- обучение с DP-гарантиями
        sample()         -- генерация синтетических строк
        privacy_report() -- отчёт о DP-бюджете
        save() / load()  -- сериализация

    Внутреннее устройство:
        1. ctgan.DataTransformer: препроцессинг (one-hot, mode-specific normalization)
        2. _VAE(nn.Module): encoder + decoder
        3. opacus.PrivacyEngine: DP-SGD с заданным epsilon
        4. Цикл обучения: ELBO loss → приватные градиенты → шаг оптимизатора
        5. sample(): sample z ~ N(0,I) → decoder → inverse_transform
    """

    def __init__(self, config: DPTVAEConfig) -> None:
        self.config = config
        self._model: Optional[_VAE] = None
        self._transformer = None        # ctgan.DataTransformer
        self._output_info: Optional[List[Any]] = None
        self._data_dim: Optional[int] = None
        self._device: Optional[torch.device] = None

        self._delta_used: Optional[float] = None
        self._spent_epsilon: Optional[float] = None
        self._epochs_completed: Optional[int] = None
        self._sample_size: Optional[int] = None
        self._fit_duration_sec: Optional[float] = None
        self._is_fitted: bool = False

    # ── Публичный API ──────────────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Обучает DP-TVAE на реальных данных с гарантиями дифференциальной приватности.

        Использует Opacus 0.14 API (совместимо с smartnoise-synth):
            - PrivacyEngine(model, batch_size, sample_size, noise_multiplier, ...)
            - privacy_engine.attach(optimizer)
            - Фактический epsilon вычисляется ПОСЛЕ обучения через get_privacy_spent()

        Параметр sigma (noise_multiplier) задаётся в конфиге напрямую — аналогично
        DPCTGANConfig.sigma. Чем больше sigma, тем строже DP и хуже качество.
        """
        try:
            from ctgan.data_transformer import DataTransformer
        except ImportError as e:
            raise ImportError(
                "Для DPTVAEGenerator необходим пакет ctgan. "
                "Установите: pip install ctgan"
            ) from e

        try:
            from opacus import PrivacyEngine
        except ImportError as e:
            raise ImportError(
                "Для DPTVAEGenerator необходим пакет opacus~=0.14. "
                "Установите: pip install opacus~=0.14.0"
            ) from e

        if self.config.sigma <= 0:
            raise ValueError(f"sigma (noise_multiplier) должна быть > 0, получено: {self.config.sigma}")

        if self.config.random_seed is not None:
            import random
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        n_rows = len(data)
        if n_rows < 10:
            raise ValueError(f"Слишком мало строк ({n_rows}). Минимум ~50.")

        self._sample_size = n_rows

        # delta: стандартная практика delta < 1/n
        if self.config.delta is not None:
            self._delta_used = self.config.delta
        else:
            self._delta_used = 1.0 / (n_rows * np.sqrt(n_rows))
            logger.info(f"[DP-TVAE] Delta авторасчёт: {self._delta_used:.2e} (n={n_rows})")

        # Устройство
        use_cuda = self.config.cuda and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"[DP-TVAE] Устройство: {self._device}")

        # ── 1. Препроцессинг данных ────────────────────────────────────────────
        discrete_columns = categorical_columns or []
        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns=discrete_columns)
        train_data = self._transformer.transform(data)

        self._output_info = self._transformer.output_info_list
        self._data_dim = train_data.shape[1]

        # DataLoader — shuffle=True, drop_last=False (Opacus 0.14 не требует drop_last)
        train_tensor = torch.FloatTensor(train_data)
        dataset = TensorDataset(train_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        if len(loader) == 0:
            raise ValueError(
                f"batch_size ({self.config.batch_size}) >= n_rows ({n_rows}). "
                "Уменьшите batch_size."
            )

        # ── 2. Модель ──────────────────────────────────────────────────────────
        model = _VAE(
            data_dim=self._data_dim,
            compress_dims=self.config.compress_dims,
            decompress_dims=self.config.decompress_dims,
            embedding_dim=self.config.embedding_dim,
        )
        # Перемещаем на устройство ДО создания PrivacyEngine —
        # иначе Opacus 0.14 создаёт шумовые тензоры на CPU, что
        # вызывает device mismatch при optimizer.step() на CUDA.
        model.to(self._device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=self.config.l2scale,
        )

        # ── 3. Применяем DP-SGD через Opacus 0.14 API ─────────────────────────
        # В Opacus 0.14: PrivacyEngine создаётся с моделью и параметрами,
        # затем attach() привязывает его к оптимизатору.
        # RDP accountant использует стандартный набор порядков α.
        rdp_alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        privacy_engine = PrivacyEngine(
            model,
            batch_size=self.config.batch_size,
            sample_size=n_rows,
            alphas=rdp_alphas,
            noise_multiplier=self.config.sigma,
            max_grad_norm=self.config.max_grad_norm,
        )
        privacy_engine.attach(optimizer)

        logger.info(
            f"[DP-TVAE] Обучение: σ={self.config.sigma}, "
            f"δ={self._delta_used:.2e}, C={self.config.max_grad_norm}, "
            f"epochs={self.config.epochs}, batch={self.config.batch_size}, rows={n_rows}"
        )

        # ── 4. Цикл обучения ──────────────────────────────────────────────────
        model.train()
        t0 = time.monotonic()

        try:
            from tqdm import tqdm
            epoch_iter = tqdm(
                range(self.config.epochs),
                desc="DP-TVAE",
                unit="epoch",
                dynamic_ncols=True,
            )
        except ImportError:
            epoch_iter = range(self.config.epochs)

        for epoch in epoch_iter:
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss = _elbo_loss(
                    recon, batch, mu, logvar,
                    self._output_info, self.config.loss_factor,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if hasattr(epoch_iter, "set_postfix"):
                try:
                    eps_now, _ = privacy_engine.get_privacy_spent(self._delta_used)
                    epoch_iter.set_postfix({"ε": f"{eps_now:.4f}", "loss": f"{epoch_loss:.2f}"})
                except Exception:
                    pass

        self._epochs_completed = self.config.epochs
        self._fit_duration_sec = time.monotonic() - t0

        # Финальный фактически потраченный ε (вычисляется через RDP accountant)
        try:
            self._spent_epsilon, _ = privacy_engine.get_privacy_spent(self._delta_used)
        except Exception:
            self._spent_epsilon = None

        model.eval()
        self._model = model
        self._is_fitted = True

        logger.info(
            f"[DP-TVAE] Завершено за {self._fit_duration_sec:.1f}с. "
            f"Spent ε={self._spent_epsilon}, epochs={self._epochs_completed}"
        )

    def sample(self, n_rows: int) -> pd.DataFrame:
        """
        Генерирует синтетические строки через сэмплинг из латентного пространства.

        z ~ N(0, I) → decoder → inverse_transform → DataFrame
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")
        if n_rows <= 0:
            raise ValueError("n_rows должен быть положительным.")

        self._model.eval()
        steps = n_rows // self.config.batch_size + 1
        chunks = []

        with torch.no_grad():
            for _ in range(steps):
                z = torch.randn(self.config.batch_size, self.config.embedding_dim).to(self._device)
                decoded = self._model.decode(z).cpu().numpy()
                chunks.append(decoded)

        generated = np.concatenate(chunks, axis=0)[:n_rows]
        return self._transformer.inverse_transform(generated)

    def privacy_report(self) -> Dict[str, Any]:
        return {
            "synthesizer": "dp_tvae_opacus",
            "opacus_version": _get_version("opacus"),
            "ctgan_version": _get_version("ctgan"),
            "status": "fitted" if self._is_fitted else "not_fitted",
            "reproducibility": {
                "random_seed": self.config.random_seed,
            },
            "data": {
                "sample_size": self._sample_size,
            },
            "dp_config": {
                "sigma": self.config.sigma,
                "epsilon_initial": None,  # sigma-based: epsilon вычисляется постфактум
                "delta": self._delta_used,
                "max_grad_norm": self.config.max_grad_norm,
                "batch_size": self.config.batch_size,
                "is_dp_enabled": True,
            },
            "dp_spent": {
                "spent_epsilon_final": self._spent_epsilon,
                "epochs_completed": self._epochs_completed,
            },
            # Поле dp_guarantees используется PrivacyEvaluator при сборке отчёта
            "dp_guarantees": {
                "spent_epsilon_final": self._spent_epsilon,
                "epochs_completed": self._epochs_completed,
                "is_dp_enabled": True,
            },
            "training": {
                "fit_duration_sec": (
                    round(self._fit_duration_sec, 2)
                    if self._fit_duration_sec else None
                ),
            },
        }

    def save(self, path: str) -> None:
        payload = {
            "config": self.config,
            # Сохраняем unwrapped модель (без Opacus-обёрток)
            "model_state_dict": self._model.state_dict(),
            "data_dim": self._data_dim,
            "transformer": self._transformer,
            "output_info": self._output_info,
            "delta_used": self._delta_used,
            "spent_epsilon": self._spent_epsilon,
            "epochs_completed": self._epochs_completed,
            "sample_size": self._sample_size,
            "fit_duration_sec": self._fit_duration_sec,
            "is_fitted": self._is_fitted,
        }
        self._pickle_save(path, payload)
        logger.info(f"[DP-TVAE] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "DPTVAEGenerator":
        payload = cls._pickle_load(path)
        obj = cls(config=payload["config"])

        # Восстанавливаем архитектуру и веса
        cfg = payload["config"]
        data_dim = payload["data_dim"]
        obj._data_dim = data_dim
        obj._transformer = payload["transformer"]
        obj._output_info = payload["output_info"]

        use_cuda = cfg.cuda and torch.cuda.is_available()
        obj._device = torch.device("cuda" if use_cuda else "cpu")

        model = _VAE(
            data_dim=data_dim,
            compress_dims=cfg.compress_dims,
            decompress_dims=cfg.decompress_dims,
            embedding_dim=cfg.embedding_dim,
        ).to(obj._device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        obj._model = model

        obj._delta_used = payload.get("delta_used")
        obj._spent_epsilon = payload.get("spent_epsilon")
        obj._epochs_completed = payload.get("epochs_completed")
        obj._sample_size = payload.get("sample_size")
        obj._fit_duration_sec = payload.get("fit_duration_sec")
        obj._is_fitted = payload["is_fitted"]
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _get_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

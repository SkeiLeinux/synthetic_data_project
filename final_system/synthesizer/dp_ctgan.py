"""
generator_dpctgan.py

DP-генератор синтетических табличных данных на базе SmartNoise Synth DPCTGAN.

Ключевые отличия от Opacus-подхода (предыдущая версия):
  - DP-SGD встроен в цикл обучения SmartNoise нативно с первой эпохи,
    что соответствует математически корректной схеме DP-гарантий.
  - Privacy accountant отслеживает фактический расход бюджета,
    а не только его арифметическое распределение.

Зависимости:
    pip install smartnoise-synth pandas numpy
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import logging
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from snsynth import Synthesizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DPCTGANConfig:
    """
    Полная конфигурация DP-CTGAN генератора.

    Параметры приватности (DP):
        epsilon           -- бюджет приватности; меньше значение = строже защита
        delta             -- вероятность "сбоя" DP-гарантии;
                             None = авторасчёт как 1 / (n * sqrt(n))
        sigma             -- множитель шума для DP-SGD;
                             больше sigma = больше шума = строже DP, но хуже качество
        max_per_sample_grad_norm -- клиппинг нормы градиентов (C в DP-SGD);
                             ограничивает чувствительность каждого шага
        preprocessor_eps  -- доля бюджета на приватное вычисление границ
                             числовых колонок при трансформации данных;
                             должна быть строго < epsilon; 0.0 = без приватного препроцессинга

    Параметры воспроизводимости:
        random_seed       -- фиксируем для повторяемости экспериментов;
                             обязателен для сравнительных экспериментов non-DP vs DP

    Отладочный режим:
        disabled_dp       -- True отключает DP (обычный GAN без шума); используется
                             как потолок качества при сравнении режимов;
                             verbose принудительно отключается из-за бага SmartNoise 1.0.6
    """
    # DP-параметры
    epsilon: float = 3.0
    delta: Optional[float] = None
    sigma: float = 5.0
    max_per_sample_grad_norm: float = 1.0
    preprocessor_eps: float = 0.0

    # Гиперпараметры обучения
    epochs: int = 300
    batch_size: int = 500
    discriminator_steps: int = 1
    pac: int = 1  # PacGAN: объединение нескольких сэмплов для дискриминатора

    # Архитектура нейронных сетей
    embedding_dim: int = 128
    generator_dim: Sequence[int] = (256, 256)
    discriminator_dim: Sequence[int] = (256, 256)

    # Оптимизатор
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6

    # Режимы
    cuda: bool = True
    verbose: bool = True
    disabled_dp: bool = False  # True = non-DP режим для baseline-сравнений
    loss: str = "cross_entropy"
    nullable: bool = False

    # Воспроизводимость
    random_seed: Optional[int] = 42


# ──────────────────────────────────────────────────────────────────────────────
# Генератор
# ──────────────────────────────────────────────────────────────────────────────

class DPCTGANGenerator:
    """
    Обёртка над SmartNoise Synth DPCTGAN с полным аудитом DP-бюджета.

    Публичный интерфейс:
        fit()                -- обучение с DP-гарантиями
        sample()             -- генерация синтетических строк
        privacy_report()     -- полный отчёт о параметрах и фактическом расходе бюджета
        estimate_max_epochs()-- оценка максимальных эпох через dry run
        save() / load()      -- сериализация обученной модели
    """

    def __init__(self, config: DPCTGANConfig) -> None:
        self.config = config
        self._synth: Optional[Any] = None

        # Трекинг бюджета; заполняется после fit()
        self._epsilon_initial: float = float(config.epsilon)
        self._delta_used: Optional[float] = config.delta
        self._epsilon_target_after_preprocess: Optional[float] = None
        self._spent_epsilon: Optional[float] = None
        self._spent_delta: Optional[float] = None
        self._epochs_completed: Optional[int] = None
        self._sample_size: Optional[int] = None
        self._fit_duration_sec: Optional[float] = None
        self._is_fitted: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
        transformer: Optional[Any] = None,
    ) -> None:
        """
        Обучает DPCTGAN на реальных данных с DP-гарантиями.

        DP-SGD встроен в цикл обучения: шум добавляется к градиентам на каждом шаге,
        privacy accountant отслеживает накопленный расход ε.
        При исчерпании бюджета обучение останавливается раньше заданных epochs --
        фактическое число выполненных эпох фиксируется в privacy_report().
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data должен быть pandas.DataFrame")

        n_rows = data.shape[0]
        if n_rows < 10:
            raise ValueError(
                f"Слишком мало строк ({n_rows}). Минимум ~50 для сходимости GAN."
            )

        # Фиксируем seed до любых операций с данными
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        self._sample_size = n_rows

        # Авторасчёт delta: стандартная практика delta < 1/n.
        # 1 / (n * sqrt(n)) -- безопасная консервативная оценка.
        if self._delta_used is None:
            self._delta_used = 1.0 / (n_rows * np.sqrt(n_rows))
            logger.info(
                f"[DP] Delta рассчитана автоматически: {self._delta_used:.2e} (n={n_rows})"
            )

        # preprocessor_eps -- часть общего бюджета, тратится на приватное
        # вычисление диапазонов числовых колонок до обучения модели.
        if not (0.0 <= self.config.preprocessor_eps < self.config.epsilon):
            raise ValueError(
                f"preprocessor_eps={self.config.preprocessor_eps} должен быть "
                f"в [0, epsilon={self.config.epsilon})"
            )

        # Плановый epsilon на само обучение = общий бюджет минус препроцессинг
        self._epsilon_target_after_preprocess = (
            self.config.epsilon - self.config.preprocessor_eps
        )

        # При disabled_dp=True SmartNoise 1.0.6 не инициализирует переменную epsilon,
        # но verbose=True всё равно пытается её напечатать → UnboundLocalError.
        # Отключаем verbose принудительно, чтобы обойти баг библиотеки.
        effective_verbose = False if self.config.disabled_dp else bool(self.config.verbose)

        try:
            self._synth = Synthesizer.create(
                "dpctgan",
                epsilon=float(self.config.epsilon),
                delta=float(self._delta_used),
                sigma=float(self.config.sigma),
                max_per_sample_grad_norm=float(self.config.max_per_sample_grad_norm),
                disabled_dp=bool(self.config.disabled_dp),
                epochs=int(self.config.epochs),
                batch_size=int(self.config.batch_size),
                discriminator_steps=int(self.config.discriminator_steps),
                pac=int(self.config.pac),
                embedding_dim=int(self.config.embedding_dim),
                generator_dim=tuple(int(x) for x in self.config.generator_dim),
                discriminator_dim=tuple(int(x) for x in self.config.discriminator_dim),
                generator_lr=float(self.config.generator_lr),
                generator_decay=float(self.config.generator_decay),
                discriminator_lr=float(self.config.discriminator_lr),
                discriminator_decay=float(self.config.discriminator_decay),
                cuda=bool(self.config.cuda),
                verbose=effective_verbose,
                loss=str(self.config.loss),
            )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка при инициализации SmartNoise Synthesizer: {e}"
            ) from e

        logger.info(
            f"[DP] Обучение: ε={self.config.epsilon}, "
            f"preprocessor_eps={self.config.preprocessor_eps}, "
            f"δ={self._delta_used:.2e}, σ={self.config.sigma}, "
            f"disabled_dp={self.config.disabled_dp}"
        )

        fit_start = time.monotonic()

        # Перехватываем stdout, чтобы поймать строки вида:
        # "epsilon is 1.484..., alpha is 10.2" и "Epoch 50, Loss G: ..."
        # SmartNoise 1.0.6 не сохраняет privacy_engine на объекте после fit() --
        # это локальная переменная внутри цикла обучения, поэтому единственный
        # способ получить spent epsilon -- распарсить stdout.
        # ВАЖНО: требует verbose=True в конфиге, иначе SmartNoise ничего не печатает.
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            self._synth.fit(
                data,
                transformer=transformer,
                categorical_columns=categorical_columns or [],
                ordinal_columns=ordinal_columns or [],
                continuous_columns=continuous_columns or [],
                preprocessor_eps=float(self.config.preprocessor_eps),
                nullable=bool(self.config.nullable),
            )

        self._fit_duration_sec = time.monotonic() - fit_start

        if self.config.disabled_dp:
            # В non-DP режиме бюджет приватности не тратится.
            # Все эпохи выполняются до конца -- бюджет не ограничивает обучение.
            self._spent_epsilon = None
            self._spent_delta = None
            self._epochs_completed = self.config.epochs
        else:
            # Парсим spent epsilon и количество выполненных эпох из перехваченного вывода
            self._spent_epsilon, self._spent_delta, self._epochs_completed = (
                _parse_privacy_from_stdout(
                    captured_output.getvalue(),
                    self._delta_used,
                    self.config.epochs,
                )
            )

        self._is_fitted = True
        logger.info(
            f"[DP] Завершено за {self._fit_duration_sec:.1f}с. "
            f"Spent ε={self._spent_epsilon}, epochs={self._epochs_completed}"
        )

    def sample(
        self,
        n_rows: int,
        condition: Optional[str] = None,
        max_tries: int = 100,
    ) -> pd.DataFrame:
        """
        Генерирует синтетические строки.

        После обучения реальные данные не используются -- генератор работает
        только через веса модели (hands-off property из ПНСТ).
        """
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")
        if n_rows <= 0:
            raise ValueError("n_rows должен быть положительным.")

        if condition is None:
            return self._synth.sample(int(n_rows))

        # Условная генерация с rejection sampling внутри SmartNoise
        return self._synth.sample_conditional(
            int(n_rows), condition=str(condition), max_tries=int(max_tries)
        )

    def privacy_report(self) -> Dict[str, Any]:
        """
        Возвращает полный отчёт о параметрах и фактическом расходе DP-бюджета.

        Ключевое различие полей:
            epsilon_target_after_preprocess -- ПЛАНОВЫЙ epsilon на обучение
                                               (арифметика: epsilon - preprocessor_eps)
            spent_epsilon_final             -- ФАКТИЧЕСКИ потраченный epsilon
                                               по данным privacy accountant после обучения;
                                               может быть меньше target, если бюджет не исчерпан,
                                               или равен ему, если обучение остановилось досрочно;
                                               None при disabled_dp=True

        Отчёт пригоден для аудита согласно ПНСТ Часть 3 (раздел документирования).
        """
        return {
            "synthesizer": "smartnoise_dpctgan",
            "snsynth_version": _get_package_version("smartnoise-synth"),
            "status": "fitted" if self._is_fitted else "not_fitted",

            # Воспроизводимость: seed и версия обязательны для сравнительных экспериментов
            "reproducibility": {
                "random_seed": self.config.random_seed,
            },

            "data": {
                "sample_size": self._sample_size,
            },

            # Конфигурация DP: все параметры, влияющие на силу гарантий
            "dp_config": {
                "epsilon_initial": self._epsilon_initial,
                "preprocessor_eps": self.config.preprocessor_eps,
                "sigma": self.config.sigma,
                "max_grad_norm": self.config.max_per_sample_grad_norm,
                "delta": self._delta_used,
                "batch_size": self.config.batch_size,
                "is_dp_enabled": not self.config.disabled_dp,
            },

            # Фактический расход бюджета -- ключевые поля для аудита
            "dp_spent": {
                "epsilon_target_after_preprocess": self._epsilon_target_after_preprocess,
                "spent_epsilon_final": self._spent_epsilon,
                "spent_delta_final": self._spent_delta,
                "epochs_requested": self.config.epochs,
                "epochs_completed": self._epochs_completed,
            },

            "training": {
                "fit_duration_sec": (
                    round(self._fit_duration_sec, 2)
                    if self._fit_duration_sec is not None else None
                ),
            },
        }

    def estimate_max_epochs(
        self,
        data: pd.DataFrame,
        probe_epochs: int = 5,
    ) -> Optional[int]:
        """
        Оценивает максимальное число эпох при текущих DP-параметрах через dry run.

        Логика:
            1. Запускаем короткое пробное обучение на probe_epochs
            2. Смотрим, сколько epsilon потрачено за эти эпохи
            3. Линейно экстраполируем на оставшийся бюджет

        Ограничения:
            - Результат приблизительный: реальная кривая расхода нелинейна
              из-за RDP-композиции в privacy accountant.
            - Это ориентир, а не точный расчёт. Правильный production-подход:
              задавать epochs с запасом и доверять встроенной остановке SmartNoise.

        Вычисление формулы расхода эпох самостоятельно не рекомендуется --
        слишком легко ошибиться в accounting (совет научного руководителя).
        """
        # В non-DP режиме бюджет не ограничивает обучение,
        # оценка максимальных эпох не имеет смысла.
        if self.config.disabled_dp:
            logger.info(
                "[estimate_max_epochs] disabled_dp=True, ограничений по бюджету нет."
            )
            return None

        if probe_epochs <= 0:
            raise ValueError("probe_epochs должен быть > 0")

        n_rows = data.shape[0]
        # Используем текущее _delta_used если уже рассчитано, иначе пересчитываем
        delta = self._delta_used or (1.0 / (n_rows * np.sqrt(n_rows)))

        try:
            # Создаём облегчённый probe-синтезатор: минимальные эпохи, без GPU.
            # verbose=True обязателен -- иначе stdout пустой и парсить нечего.
            probe_synth = Synthesizer.create(
                "dpctgan",
                epsilon=float(self.config.epsilon),
                delta=float(delta),
                sigma=float(self.config.sigma),
                max_per_sample_grad_norm=float(self.config.max_per_sample_grad_norm),
                epochs=probe_epochs,
                batch_size=int(self.config.batch_size),
                verbose=True,
                cuda=False,
            )
            # Перехватываем stdout по той же схеме, что и в fit() --
            # probe_synth тоже не хранит privacy_engine после обучения
            captured_probe = io.StringIO()
            with contextlib.redirect_stdout(captured_probe):
                probe_synth.fit(
                    data,
                    preprocessor_eps=float(self.config.preprocessor_eps),
                    nullable=bool(self.config.nullable),
                )
        except Exception as e:
            logger.warning(f"[estimate_max_epochs] Dry run не удался: {e}")
            return None

        spent_probe, _, _ = _parse_privacy_from_stdout(
            captured_probe.getvalue(), delta, probe_epochs
        )
        if spent_probe is None or spent_probe <= 0:
            logger.warning(
                "[estimate_max_epochs] Не удалось получить spent epsilon из dry run"
            )
            return None

        # Линейная экстраполяция по epsilon_per_epoch.
        # Намеренно занижает результат из-за нелинейности RDP -- используй как нижнюю оценку.
        budget = self.config.epsilon - self.config.preprocessor_eps
        eps_per_epoch = spent_probe / probe_epochs
        estimated = int(budget / eps_per_epoch)

        logger.info(
            f"[estimate_max_epochs] probe spent={spent_probe:.4f} за {probe_epochs} эпох, "
            f"ε/epoch≈{eps_per_epoch:.4f}, оценка max_epochs≈{estimated}"
        )
        return estimated

    def save(self, path: str) -> None:
        """
        Сохраняет обученный генератор через pickle.

        ВАЖНО: pickle небезопасен при загрузке из недоверенных источников.
        Использовать только внутри закрытого контура системы.
        В production-версии планируется заменить на безопасный формат сериализации.
        """
        if not self._is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")

        payload = {
            "config": self.config,
            "epsilon_initial": self._epsilon_initial,
            "epsilon_target_after_preprocess": self._epsilon_target_after_preprocess,
            "spent_epsilon": self._spent_epsilon,
            "spent_delta": self._spent_delta,
            "delta_used": self._delta_used,
            "epochs_completed": self._epochs_completed,
            "sample_size": self._sample_size,
            "fit_duration_sec": self._fit_duration_sec,
            "synth": self._synth,
            "is_fitted": self._is_fitted,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            logger.info(f"[DP-GAN] Модель сохранена: {path}")
        except Exception as e:
            raise IOError(f"Ошибка при сохранении модели: {e}") from e

    @classmethod
    def load(cls, path: str) -> "DPCTGANGenerator":
        """
        Загружает генератор из pickle-файла.
        Совместим со старыми сохранёнными версиями через .get() с fallback.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            raise IOError(f"Ошибка при загрузке модели: {e}") from e

        obj = cls(config=payload["config"])
        obj._epsilon_initial = payload["epsilon_initial"]
        obj._epsilon_target_after_preprocess = payload.get("epsilon_target_after_preprocess")
        obj._spent_epsilon = payload.get("spent_epsilon")
        obj._spent_delta = payload.get("spent_delta")
        obj._delta_used = payload.get("delta_used")
        obj._epochs_completed = payload.get("epochs_completed")
        obj._sample_size = payload.get("sample_size")
        obj._fit_duration_sec = payload.get("fit_duration_sec")
        obj._synth = payload["synth"]
        obj._is_fitted = payload["is_fitted"]
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции (внутренние)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_privacy_from_stdout(
    output: str,
    delta: Optional[float],
    epochs_requested: int,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Парсит spent epsilon и число выполненных эпох из перехваченного stdout SmartNoise.

    SmartNoise 1.0.6 не хранит privacy_engine на объекте после обучения,
    но на каждой эпохе печатает строку вида:
        "epsilon is 1.4843, alpha is 10.2"
    и строку вида:
        "Epoch 50, Loss G: 0.706, Loss D: 1.395"

    Берём значения из последних найденных строк -- это финальное состояние
    после завершения обучения (либо по числу эпох, либо по исчерпанию бюджета).

    Ограничение: зависит от формата вывода SmartNoise. При смене версии библиотеки
    паттерны могут потребовать обновления.
    """
    spent_epsilon: Optional[float] = None
    epochs_completed: Optional[int] = None

    epsilon_pattern = re.compile(r"epsilon is ([0-9.]+(?:e[+-]?\d+)?)")
    epoch_pattern   = re.compile(r"Epoch (\d+),")

    # Перебираем строки и обновляем значения -- в итоге остаётся последнее
    for line in output.splitlines():
        eps_match = epsilon_pattern.search(line)
        if eps_match:
            spent_epsilon = float(eps_match.group(1))

        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epochs_completed = int(epoch_match.group(1))

    if spent_epsilon is None:
        logger.warning(
            "[parse_privacy] Не удалось распарсить spent epsilon из stdout. "
            "Убедитесь, что verbose=True в конфиге."
        )
    else:
        logger.info(
            f"[parse_privacy] Spent ε={spent_epsilon:.4f}, "
            f"epochs_completed={epochs_completed}/{epochs_requested}"
        )

    return spent_epsilon, delta, epochs_completed


def _get_package_version(package_name: str) -> str:
    """Возвращает версию пакета для фиксации в отчёте о воспроизводимости."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Тестовый блок
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print("=== ТЕСТ DP-CTGAN GENERATOR ===\n")

    # датасет для теста (500 строк для стабильной сходимости GAN)
    np.random.seed(42)
    N = 500
    df_test = pd.DataFrame({
        "age":         np.random.randint(18, 90, N),
        "income":      np.random.normal(50000, 15000, N).round(2),
        "city":        np.random.choice(["Moscow", "Saint-P", "Novosibirsk", "Kazan"], N),
        "is_customer": np.random.choice([0, 1], N),
    })

    print(f"[1] Данные: {df_test.shape}")
    print(df_test.head(), "\n")

    # Конфигурация:
    #   epsilon=3.0, preprocessor_eps=1.5 → на обучение планируется 1.5
    #   epochs=100: верхняя граница; SmartNoise остановится раньше при исчерпании бюджета
    config = DPCTGANConfig(
        epsilon=3.0,
        preprocessor_eps=1.5,
        delta=None,         # авторасчёт
        epochs=100,
        batch_size=20,
        cuda=False,
        verbose=True,
        disabled_dp=False,  # явно указываем режим для читаемости
        random_seed=42,
    )

    cat_cols  = ["city", "is_customer"]
    cont_cols = ["age", "income"]

    generator = DPCTGANGenerator(config)

    # Оценка максимальных эпох через dry run перед полным обучением
    print("[2] Оценка max_epochs (dry run, probe_epochs=3)...")
    estimated = generator.estimate_max_epochs(df_test, probe_epochs=3)
    print(f"    Оценочный max_epochs: {estimated}\n")

    # Обучение
    print(f"[3] Обучение (epochs={config.epochs}, disabled_dp={config.disabled_dp})...")
    generator.fit(df_test, categorical_columns=cat_cols, continuous_columns=cont_cols)
    print("    Готово.\n")

    # Privacy report -- главная проверка корректности бюджета
    report = generator.privacy_report()
    print("[4] Privacy Report:")
    for section, value in report.items():
        if isinstance(value, dict):
            print(f"  {section}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {section}: {value}")
    print()

    # Генерация синтетики
    print("[5] Генерация 10 строк...")
    synth_df = generator.sample(10)
    print(synth_df, "\n")

    # Тест сериализации
    print("[6] Сохранение / загрузка...")
    save_path = "../dp_ctgan_model.pkl"
    generator.save(save_path)
    loaded = DPCTGANGenerator.load(save_path)
    print("    Генерация из загруженной модели:")
    print(loaded.sample(3))

    print("\n=== ТЕСТ ЗАВЕРШЁН ===")

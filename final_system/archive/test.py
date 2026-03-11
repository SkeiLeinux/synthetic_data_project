"""
generatorDPCTGAN.py

DP-генератор синтетических табличных данных на базе SmartNoise Synth DPCTGAN.
Обертка включает автоматический расчет Delta и управление бюджетом препроцессинга.

Зависимости:
    pip install smartnoise-synth pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Union
import pickle
import numpy as np
import pandas as pd

# Библиотека SmartNoise Synth
import snsynth
from snsynth import Synthesizer


@dataclass
class DPCTGANConfig:
    """
    Конфигурация для DP-CTGAN генератора.
    """
    # Основной DP-параметр: бюджет приватности (Epsilon)
    epsilon: float = 3.0

    # DP-параметры градиентного шума
    # Если delta=None, он будет рассчитан автоматически как 1/(n * sqrt(n))
    delta: Optional[float] = None

    # Sigma (множитель шума) и клиппинг градиентов
    sigma: float = 5.0
    max_per_sample_grad_norm: float = 1.0

    # Гиперпараметры обучения
    epochs: int = 300
    batch_size: int = 500
    discriminator_steps: int = 1
    pac: int = 1  # Количество сэмплов, объединяемых для дискриминатора (PacGAN)

    # Архитектура нейросетей
    embedding_dim: int = 128
    generator_dim: Sequence[int] = (256, 256)
    discriminator_dim: Sequence[int] = (256, 256)

    # Оптимизация (Learning Rates и Decay)
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6

    # Режимы работы
    cuda: bool = True
    verbose: bool = True

    # Отладочный флаг: если True, DP отключается (обычный GAN)
    disabled_dp: bool = False

    # Функция потерь (стандартная для SmartNoise)
    loss: str = "cross_entropy"

    # Бюджет на препроцессинг (расчет границ числовых данных)
    # Вычитается из общего epsilon.
    preprocessor_eps: float = 0.0

    # Допускать ли NULL значения
    nullable: bool = False


class DPCTGANGenerator:
    """
    Обертка над SmartNoise Synth DPCTGAN с удобными методами:
      - fit(): обучение на приватных данных с авто-расчетом Delta
      - sample(): генерация синтетики
      - privacy_report(): параметры DP для аудита
      - save/load(): сохранение обученного объекта
    """

    def __init__(self, config: DPCTGANConfig):
        self.config = config
        self._synth: Optional[Synthesizer] = None

        # Фиксируем начальный бюджет
        self._epsilon_initial: float = float(config.epsilon)

        # Бюджет, оставшийся на обучение после препроцессинга
        self._epsilon_after_preprocess: Optional[float] = None

        # Фактическое значение Delta (из конфига или рассчитанное)
        self._delta_used: Optional[float] = config.delta

        self._sample_size: Optional[int] = None
        self._spent_epsilon_by_epoch: Optional[List[float]] = None
        self._best_alpha_by_epoch: Optional[List[float]] = None
        self._epochs_completed: Optional[int] = None

        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
        transformer: Optional[Any] = None,
    ) -> None:
        """
        Обучаем DPCTGAN на исходных данных.
        """
        # 1. Валидация входных данных
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data должен быть pandas.DataFrame")

        n_rows = data.shape[0]
        self._sample_size = int(n_rows)

        # GAN требует достаточного количества данных для сходимости
        if n_rows < 10:
            raise ValueError(f"Слишком мало строк для обучения ({n_rows}). Рекомендуется минимум 50+.")

        # 2. Автоматический расчет Delta, если не задан
        # Стандартная практика: delta < 1/n. Безопасное значение: 1 / (n * sqrt(n))
        if self._delta_used is None:
            self._delta_used = 1 / (n_rows * np.sqrt(n_rows))
            if self.config.verbose:
                print(f"[DP-GAN] Delta не задана. Рассчитано автоматически: {self._delta_used:.2e} (для n={n_rows})")

        # 3. Нормализация списков колонок
        categorical_columns = categorical_columns or []
        ordinal_columns = ordinal_columns or []
        continuous_columns = continuous_columns or []

        # 4. Проверка бюджета препроцессинга
        if self.config.preprocessor_eps < 0.0:
            raise ValueError("preprocessor_eps не может быть отрицательным")
        if self.config.preprocessor_eps >= self.config.epsilon:
            raise ValueError("preprocessor_eps должен быть строго меньше epsilon")

        # 5. Создание синтезатора SmartNoise
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
                verbose=bool(self.config.verbose),
                loss=str(self.config.loss),
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при инициализации SmartNoise Synthesizer: {e}")

        # 6. Запуск обучения
        if self.config.verbose:
            print(f"[DP-GAN] Начало обучения. Epsilon: {self.config.epsilon}, Preproc Budget: {self.config.preprocessor_eps}")

        self._synth.fit(
            data,
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            preprocessor_eps=float(self.config.preprocessor_eps),
            nullable=bool(self.config.nullable),
        )

        # 7. Фиксация итогового состояния
        # SmartNoise обновляет epsilon внутри объекта после трат на препроцессинг
        self._epsilon_after_preprocess = float(getattr(self._synth, "epsilon", self.config.epsilon))
        self._extract_training_privacy_stats()
        self._is_fitted = True

    def sample(self, n_rows: int, condition: Optional[str] = None, max_tries: int = 100) -> pd.DataFrame:
        """
        Генерируем синтетические данные.
        """
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Модель не обучена. Сначала вызови fit().")

        if n_rows <= 0:
            raise ValueError("n_rows должен быть положительным числом.")

        # Генерация без условий
        if condition is None:
            return self._synth.sample(int(n_rows))

        # Генерация с условием (rejection sampling внутри SmartNoise)
        return self._synth.sample_conditional(int(n_rows), condition=str(condition), max_tries=int(max_tries))

    def privacy_report(self) -> Dict[str, Any]:
        """
        Возвращает отчет о параметрах приватности для аудита.
        """
        report: Dict[str, Any] = {
            "synthesizer": "smartnoise_dpctgan",
            "config": asdict(self.config),
            "privacy_metrics": {
                "initial_epsilon": self._epsilon_initial,
                "epsilon_target_after_preprocess": self._epsilon_after_preprocess,
                "delta": self._delta_used,
                "is_dp_enabled": not self.config.disabled_dp
            },
            "status": "fitted" if self._is_fitted else "not_fitted"
        }
        return report

    def save(self, path: str) -> None:
        """
        Сохранение генератора через pickle.
        """
        if not self._is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")

        payload = {
            "config": self.config,
            "epsilon_initial": self._epsilon_initial,
            "epsilon_after_preprocess": self._epsilon_after_preprocess,
            "delta_used": self._delta_used,
            "synth": self._synth,
            "is_fitted": self._is_fitted,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            if self.config.verbose:
                print(f"[DP-GAN] Модель сохранена в {path}")
        except Exception as e:
            raise IOError(f"Ошибка при сохранении модели: {e}")

    @classmethod
    def load(cls, path: str) -> "DPCTGANGenerator":
        """
        Загрузка генератора из файла.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            raise IOError(f"Ошибка при загрузке модели: {e}")

        # Восстанавливаем состояние
        obj = cls(config=payload["config"])
        obj._epsilon_initial = payload["epsilon_initial"]
        obj._epsilon_after_preprocess = payload["epsilon_after_preprocess"]
        obj._delta_used = payload.get("delta_used") # Совместимость со старыми версиями, если будут
        obj._synth = payload["synth"]
        obj._is_fitted = payload["is_fitted"]

        return obj


# --- БЛОК ТЕСТИРОВАНИЯ ---
if __name__ == "__main__":
    # Настройка параметров отображения Pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("=== ЗАПУСК ТЕСТА DP-CTGAN GENERATOR ===")

    # 1. Генерация тестового датасета (чуть больше данных для стабильности GAN)
    # 100 строк, чтобы батчи по 10-20 проходили корректно
    np.random.seed(42)
    sample_size = 500

    df_test = pd.DataFrame({
        "age": np.random.randint(18, 90, sample_size),
        "income": np.random.normal(50000, 15000, sample_size).round(2),
        "city": np.random.choice(["Moscow", "Saint-P", "Novosibirsk", "Kazan"], sample_size),
        "is_customer": np.random.choice([0, 1], sample_size)
    })

    print(f"\n[1] Исходные данные (размер {df_test.shape}):")
    print(df_test.head())

    # 2. Настройка конфига
    # Важно: preprocessor_eps должно быть < epsilon.
    # Epsilon = 3.0, Preproc = 0.5 -> На обучение останется 2.5
    test_config = DPCTGANConfig(
        epsilon=3.0,
        preprocessor_eps=1.5,
        delta=None,         # Проверка авторасчета
        epochs=100,          # Эпохи генерации
        batch_size=20,      # Batch size < размера данных
        cuda=False,         # Использование GPU
        verbose=True
    )

    # 3. Инициализация и обучение
    try:
        generator = DPCTGANGenerator(test_config)

        # Указываем типы колонок (для улучшения качества, но SmartNoise может и сам определить)
        cat_cols = ["city", "is_customer"]
        cont_cols = ["age", "income"]
        constraints = {
            "age": [0, 100],
            "income": [0, 200000]
        }

        print(f"\n[2] Запуск обучения (Epochs={test_config.epochs})...")
        generator.fit(
            df_test,
            categorical_columns=cat_cols,
            continuous_columns=cont_cols
            # transformer=constraints
        )
        print("[OK] Обучение завершено.")

        # 4. Проверка отчета о приватности
        report = generator.privacy_report()
        print(f"\n[3] Отчет о приватности:\n{report['privacy_metrics']}")

        # Проверка логики бюджета
        eps_used = report['privacy_metrics']['epsilon_target_after_preprocess']
        expected_eps = test_config.epsilon - test_config.preprocessor_eps # 2.5

        # SmartNoise может вернуть чуть меньше или ровно, главное не больше
        if eps_used is not None and abs(eps_used - expected_eps) < 1e-5:
            print(f"[CHECK] Бюджет распределен корректно: {eps_used}")
        else:
            print(f"[WARNING] Бюджет отличается от ожидаемого: {eps_used} (ожидалось ~{expected_eps})")

        # 5. Генерация
        print(f"\n[4] Генерация 10 синтетических строк...")
        synth_data = generator.sample(10)
        print(synth_data)

        # 6. Тест сохранения/загрузки
        print(f"\n[5] Тест сохранения модели...")
        save_path = "dp_generator_test.pkl"
        generator.save(save_path)

        loaded_gen = DPCTGANGenerator.load(save_path)
        print("[OK] Модель загружена. Проверка генерации из загруженной модели:")
        print(loaded_gen.sample(3))

        print("\n=== ТЕСТ УСПЕШНО ЗАВЕРШЕН ===")

    except Exception as e:
        print(f"\n[ERROR] Произошла ошибка при выполнении теста:\n{e}")
        import traceback
        traceback.print_exc()
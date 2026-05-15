"""
generatorDPCTGAN.py

DP-генератор синтетических табличных данных на базе SmartNoise Synth DPCTGAN.

Зависимости:
    pip install smartnoise-synth pandas

Документация (основное):
    - Synthesizer.create / fit / sample
    - DPCTGAN параметры: epsilon, sigma, delta, max_per_sample_grad_norm, epochs, batch_size и т.д.

Важно:
    - DP гарантия относится к этапу обучения (fit).
    - Для передачи данных между сервисами лучше использовать ссылки на артефакты (dataset_ref),
      а не отправлять DataFrame по сети. Здесь реализован локальный класс-обертка.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# Библиотека SmartNoise Synth
from snsynth import Synthesizer


@dataclass
class DPCTGANConfig:
    # Основной DP-параметр: бюджет приватности
    epsilon: float = 3.0

    # DP-параметры градиентного шума
    sigma: float = 5.0
    delta: Optional[float] = None
    max_per_sample_grad_norm: float = 1.0

    # Гиперпараметры обучения
    epochs: int = 300
    batch_size: int = 500
    discriminator_steps: int = 1
    pac: int = 1

    # Архитектура
    embedding_dim: int = 128
    generator_dim: Sequence[int] = (256, 256)
    discriminator_dim: Sequence[int] = (256, 256)

    # Оптимизация
    generator_lr: float = 0.0002
    generator_decay: float = 1e-06
    discriminator_lr: float = 0.0002
    discriminator_decay: float = 1e-06

    # Режимы
    cuda: bool = True
    verbose: bool = True

    # Отладочный флаг: обучать без DP (для диагностики нестабильности GAN)
    disabled_dp: bool = False

    # Функция потерь (как в SmartNoise)
    loss: str = "cross_entropy"

    # Бюджет на препроцессинг
    # Если трансформеру нужно оценить границы числовых колонок, он потратит часть epsilon.
    preprocessor_eps: float = 0.0

    # Если True, трансформер будет допускать null значения как отдельное состояние
    nullable: bool = False


class DPCTGANGenerator:
    """
    Обертка над SmartNoise Synth DPCTGAN с удобными методами:
      - fit(): обучение на приватных данных
      - sample(): генерация синтетики
      - privacy_report(): параметры DP для аудита
      - save/load(): сохранение обученного объекта
    """

    def __init__(self, config: DPCTGANConfig):
        # Сохраняем конфигурацию
        self.config = config

        # Здесь будет объект SmartNoise Synthesizer
        self._synth: Optional[Synthesizer] = None

        # Фиксируем epsilon до препроцессинга
        self._epsilon_initial: float = float(config.epsilon)

        # Фиксируем epsilon, который остался на обучение после препроцессинга
        self._epsilon_after_preprocess: Optional[float] = None

        # Фиксируем признак обучения
        self._is_fitted: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[Union[str, int]]] = None,
        ordinal_columns: Optional[List[Union[str, int]]] = None,
        continuous_columns: Optional[List[Union[str, int]]] = None,
        transformer: Optional[Any] = None,
    ) -> None:
        """
        Обучаем DPCTGAN на исходных данных.

        data:
            Приватный датафрейм
        categorical_columns / ordinal_columns / continuous_columns:
            Подсказки для трансформера (если transformer не передан)
        transformer:
            Можно передать TableTransformer или dict-constraints из SmartNoise
        """

        # Проверяем входные данные
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data должен быть pandas.DataFrame")

        if data.shape[0] < 10:
            # Слишком мало строк для GAN, будет нестабильно
            raise ValueError("Слишком мало строк для обучения DPCTGAN. Нужно хотя бы 10-50+.")

        # Нормализуем списки колонок
        categorical_columns = categorical_columns or []
        ordinal_columns = ordinal_columns or []
        continuous_columns = continuous_columns or []

        # Проверяем бюджет препроцессинга
        if self.config.preprocessor_eps < 0.0:
            raise ValueError("preprocessor_eps не может быть отрицательным")
        if self.config.preprocessor_eps >= self.config.epsilon:
            raise ValueError("preprocessor_eps должен быть меньше epsilon, иначе не останется бюджета на обучение")

        # Создаем синтезатор
        # Важно: сюда передаем hyperparameters именно DPCTGAN
        self._synth = Synthesizer.create(
            "dpctgan",
            epsilon=float(self.config.epsilon),
            # DP параметры
            sigma=float(self.config.sigma),
            delta=self.config.delta,
            max_per_sample_grad_norm=float(self.config.max_per_sample_grad_norm),
            disabled_dp=bool(self.config.disabled_dp),
            # Обучение
            epochs=int(self.config.epochs),
            batch_size=int(self.config.batch_size),
            discriminator_steps=int(self.config.discriminator_steps),
            pac=int(self.config.pac),
            # Архитектура
            embedding_dim=int(self.config.embedding_dim),
            generator_dim=tuple(int(x) for x in self.config.generator_dim),
            discriminator_dim=tuple(int(x) for x in self.config.discriminator_dim),
            # Оптимизация
            generator_lr=float(self.config.generator_lr),
            generator_decay=float(self.config.generator_decay),
            discriminator_lr=float(self.config.discriminator_lr),
            discriminator_decay=float(self.config.discriminator_decay),
            # Режимы
            cuda=bool(self.config.cuda),
            verbose=bool(self.config.verbose),
            # Лосс
            loss=str(self.config.loss),
        )

        # Обучаем (fit)
        # preprocessor_eps может быть > 0.0, если трансформер должен оценить bounds приватно
        self._synth.fit(
            data,
            transformer=transformer,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns,
            preprocessor_eps=float(self.config.preprocessor_eps),
            nullable=bool(self.config.nullable),
        )

        # Сохраняем epsilon после препроцессинга (внутри SmartNoise он может уменьшиться)
        # Примечание: это "оставшийся epsilon на обучение" после траты на препроцессинг,
        # а не "остаток после обучения".
        self._epsilon_after_preprocess = float(getattr(self._synth, "epsilon", self.config.epsilon))

        # Отмечаем состояние
        self._is_fitted = True

    def sample(self, n_rows: int, condition: Optional[str] = None, max_tries: int = 100) -> pd.DataFrame:
        """
        Генерируем синтетику.

        n_rows:
            Сколько строк сгенерировать
        condition:
            Условие SQL WHERE для sample_conditional (опционально)
        max_tries:
            Количество попыток при rejection sampling (только для condition)
        """
        # Проверяем состояние
        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Сначала вызови fit(), затем sample()")

        if n_rows <= 0:
            raise ValueError("n_rows должен быть положительным")

        # Генерируем без условий
        if condition is None:
            return self._synth.sample(int(n_rows))

        # Генерируем с условием (rejection sampling)
        return self._synth.sample_conditional(int(n_rows), condition=str(condition), max_tries=int(max_tries))

    def privacy_report(self) -> Dict[str, Any]:
        """
        Возвращаем данные для аудита DP.
        Эти значения важно сохранять в метаданные процесса.
        """
        report: Dict[str, Any] = {
            "synthesizer": "smartnoise_dpctgan",
            # Исходная конфигурация
            "config": asdict(self.config),
            # Для прозрачности фиксируем ключевые значения отдельно
            "epsilon_initial": float(self._epsilon_initial),
            "epsilon_after_preprocess": None if self._epsilon_after_preprocess is None else float(self._epsilon_after_preprocess),
            "dp_enabled": not bool(self.config.disabled_dp),
        }

        # Дополнительные поля из внутреннего объекта (если доступны)
        if self._synth is not None:
            report["epsilon_internal_current"] = float(getattr(self._synth, "epsilon", self._epsilon_after_preprocess or self._epsilon_initial))

        return report

    def is_fitted(self) -> bool:
        # Возвращаем статус обучения
        return bool(self._is_fitted)

    def save(self, path: str) -> None:
        """
        Сохраняем обученный генератор на диск.
        Используем pickle, так как SmartNoise Synthesizer обычно сериализуем.
        """
        import pickle

        if not self._is_fitted or self._synth is None:
            raise RuntimeError("Нельзя сохранить: генератор еще не обучен (fit не вызывался)")

        payload = {
            "config": self.config,
            "epsilon_initial": self._epsilon_initial,
            "epsilon_after_preprocess": self._epsilon_after_preprocess,
            "synth": self._synth,
            "is_fitted": self._is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "DPCTGANGenerator":
        """
        Загружаем сохраненный генератор.
        """
        import pickle

        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(config=payload["config"])
        obj._epsilon_initial = float(payload.get("epsilon_initial", obj.config.epsilon))
        obj._epsilon_after_preprocess = payload.get("epsilon_after_preprocess")
        obj._synth = payload.get("synth")
        obj._is_fitted = bool(payload.get("is_fitted", False))

        return obj


# Пример использования (локальная проверка)
if __name__ == "__main__":
    # Загружаем пример данных
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "city": ["A", "A", "B", "B", "B", "C", "C", "C", "A", "B"],
            "income": [1000, 1100, 900, 1200, 1150, 1300, 1250, 1400, 1050, 980],
        }
    )

    # Создаем конфиг
    cfg = DPCTGANConfig(
        epsilon=3.0,
        preprocessor_eps=0.5,
        epochs=50,
        batch_size=10,
        cuda=False,
        verbose=False,
    )

    # Создаем генератор
    gen = DPCTGANGenerator(cfg)

    # Обучаем
    gen.fit(df, categorical_columns=["city"])

    # Генерируем 20 строк
    syn = gen.sample(20)

    # Печатаем отчет по DP
    print(gen.privacy_report())

    # Печатаем результат
    print(syn.head())

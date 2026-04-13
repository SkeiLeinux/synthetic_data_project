# synthesizer/loader.py
#
# Универсальный загрузчик генераторов из .pkl-файлов.
# Определяет тип генератора по классу конфига, сохранённому в payload.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synthesizer.base import BaseGenerator


def load_generator(path: str) -> "BaseGenerator":
    """
    Загружает генератор из .pkl-файла, автоматически определяя его тип.

    Поддерживаемые типы (определяются по классу config в payload):
        DPCTGANConfig   → DPCTGANGenerator
        DPTVAEConfig    → DPTVAEGenerator
        CTGANConfig     → CTGANGenerator
        TVAEConfig      → TVAEGenerator
        CopulaGANConfig → CopulaGANGenerator

    Пример использования:
        generator = load_generator("models/adult.pkl")
        synth_df = generator.sample(10000)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        raise IOError(f"Не удалось прочитать файл модели {path}: {e}") from e

    config = payload.get("config")
    if config is None:
        raise ValueError(f"Файл {path} не содержит поле 'config' — возможно, повреждён.")

    config_class = type(config).__name__

    if config_class == "DPCTGANConfig":
        from synthesizer.dp_ctgan import DPCTGANGenerator
        return DPCTGANGenerator.load(path)

    elif config_class == "DPTVAEConfig":
        from synthesizer.dp_tvae import DPTVAEGenerator
        return DPTVAEGenerator.load(path)

    elif config_class == "CTGANConfig":
        from synthesizer.sdv_generators import CTGANGenerator
        return CTGANGenerator.load(path)

    elif config_class == "TVAEConfig":
        from synthesizer.sdv_generators import TVAEGenerator
        return TVAEGenerator.load(path)

    elif config_class == "CopulaGANConfig":
        from synthesizer.sdv_generators import CopulaGANGenerator
        return CopulaGANGenerator.load(path)

    else:
        raise ValueError(
            f"Неизвестный тип конфига в файле модели: {config_class!r}. "
            f"Поддерживаются: DPCTGANConfig, DPTVAEConfig, CTGANConfig, "
            f"TVAEConfig, CopulaGANConfig."
        )

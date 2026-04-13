# synthesizer/__init__.py
#
# Публичный API модуля synthesizer.
# Импортируйте отсюда для удобства:
#   from synthesizer import DPCTGANGenerator, DPTVAEGenerator, load_generator

from synthesizer.base import BaseGenerator
from synthesizer.dp_ctgan import DPCTGANConfig, DPCTGANGenerator
from synthesizer.dp_tvae import DPTVAEConfig, DPTVAEGenerator
from synthesizer.sdv_generators import (
    CTGANConfig, CTGANGenerator,
    TVAEConfig, TVAEGenerator,
    CopulaGANConfig, CopulaGANGenerator,
)
from synthesizer.loader import load_generator

__all__ = [
    "BaseGenerator",
    "DPCTGANConfig", "DPCTGANGenerator",
    "DPTVAEConfig",  "DPTVAEGenerator",
    "CTGANConfig",   "CTGANGenerator",
    "TVAEConfig",    "TVAEGenerator",
    "CopulaGANConfig", "CopulaGANGenerator",
    "load_generator",
]

# final_system/config_loader.py
#
# Загрузчик конфигурации из YAML с валидацией через Pydantic.
#
# Превращает configs/adult.yaml в типизированные dataclass-совместимые объекты,
# готовые к передаче напрямую в run_pipeline():
#
#   cfg = load_config("configs/adult.yaml")
#   synth_df, report = run_pipeline(
#       real_df=df,
#       synth_config=cfg.generator,
#       privacy_config=cfg.privacy,
#       utility_config=cfg.utility,
#       thresholds=cfg.thresholds,
#       ...
#   )
#
# Pydantic обеспечивает:
#   - Проверку типов на старте (не в середине обучения)
#   - Читаемые сообщения об ошибках при опечатках в конфиге
#   - Значения по умолчанию без дублирования логики в run_adult.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Импорты dataclass-конфигов из модулей ─────────────────────────────────────
# Загружаем оригинальные dataclass'ы, чтобы config_loader возвращал
# именно те объекты, которые ожидают synthesizer, evaluator, reporter.
from synthesizer.dp_ctgan import DPCTGANConfig
from synthesizer.dp_tvae import DPTVAEConfig
from synthesizer.sdv_generators import CTGANConfig, TVAEConfig, CopulaGANConfig
from evaluator.privacy.privacy_evaluator import PrivacyConfig
from evaluator.utility.utility_evaluator import UtilityConfig
from reporter.reporter import VerdictThresholds

# Тип конфига генератора (union всех вариантов)
GeneratorConfig = DPCTGANConfig | DPTVAEConfig | CTGANConfig | TVAEConfig | CopulaGANConfig


# ==============================================================================
# Pydantic-схемы для валидации YAML
# ==============================================================================

class DBConfig(BaseModel):
    """Параметры подключения к PostgreSQL."""
    host: str = "localhost"
    port: int = 5432
    dbname: str
    user: str
    password: str
    schema_name: str = Field("public", alias="schema")

    model_config = {"populate_by_name": True}


class PathsConfig(BaseModel):
    logs: str = "logs/app.log"
    output_dir: str = "reporter/reports"


class PipelineConfig(BaseModel):
    dataset_name: str = "dataset"

    # Источник данных: csv | db
    data_source: str = "csv"
    data_path: str = ""           # путь к CSV; нужен если data_source = csv
    db_query: Optional[str] = None  # SQL-запрос; нужен если data_source = db

    sample_size: int = 0          # 0 = все строки
    n_synth_rows: int = 0         # 0 = совпадает с размером train
    run_preprocessing: bool = True
    holdout_size: float = 0.2
    random_state: int = 42

    @field_validator("data_source")
    @classmethod
    def check_data_source(cls, v: str) -> str:
        if v not in ("csv", "db"):
            raise ValueError(
                f"data_source должен быть csv или db, получено: {v!r}"
            )
        return v

    @field_validator("holdout_size")
    @classmethod
    def check_holdout(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(
                f"holdout_size должен быть в (0, 1), получено: {v}"
            )
        return v

    @model_validator(mode="after")
    def check_source_params(self) -> "PipelineConfig":
        if self.data_source == "csv" and not self.data_path:
            raise ValueError("data_path обязателен при data_source: csv")
        if self.data_source == "db" and not self.db_query:
            raise ValueError("db_query обязателен при data_source: db")
        return self


_DP_GENERATOR_TYPES = {"dpctgan", "dptvae"}
_ALL_GENERATOR_TYPES = {"dpctgan", "dptvae", "ctgan", "tvae", "copulagan"}


class GeneratorYamlConfig(BaseModel):
    """
    Параметры генератора из YAML.

    Поле generator_type определяет, какой генератор будет создан:
        dpctgan   -- DP-CTGAN (SmartNoise Synth) — default, с DP-гарантиями
        dptvae    -- DP-TVAE (Opacus + TVAE) — с DP-гарантиями
        ctgan     -- CTGAN (SDV) — без DP, baseline
        tvae      -- TVAE (SDV) — без DP, baseline
        copulagan -- CopulaGAN (SDV) — без DP, baseline

    Поля применяются избирательно в зависимости от generator_type.
    Non-DP генераторы игнорируют epsilon/delta/sigma/max_per_sample_grad_norm.
    Поля DP-специфичны для конкретных генераторов отмечены комментариями.
    """
    generator_type: str = "dpctgan"

    # DP-параметры (используются dpctgan / dptvae)
    epsilon: float = 3.0
    delta: Optional[float] = None

    # DP-CTGAN specific
    preprocessor_eps: float = 0.5
    sigma: float = 5.0
    max_per_sample_grad_norm: float = 1.0
    disabled_dp: bool = False
    loss: str = "cross_entropy"
    nullable: bool = False

    # DP-TVAE specific
    max_grad_norm: float = 1.0

    # Общие параметры обучения
    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    cuda: bool = True
    verbose: bool = True
    random_seed: Optional[int] = 42

    # CTGAN / CopulaGAN architecture
    discriminator_steps: int = 1
    pac: int = 1
    generator_dim: List[int] = Field(default_factory=lambda: [256, 256])
    discriminator_dim: List[int] = Field(default_factory=lambda: [256, 256])
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    log_frequency: bool = True

    # TVAE / DP-TVAE architecture
    compress_dims: List[int] = Field(default_factory=lambda: [128, 128])
    decompress_dims: List[int] = Field(default_factory=lambda: [128, 128])
    l2scale: float = 1e-5
    loss_factor: int = 2

    @field_validator("generator_type")
    @classmethod
    def check_generator_type(cls, v: str) -> str:
        if v not in _ALL_GENERATOR_TYPES:
            raise ValueError(
                f"generator_type должен быть одним из {sorted(_ALL_GENERATOR_TYPES)}, "
                f"получено: {v!r}"
            )
        return v

    @field_validator("epsilon")
    @classmethod
    def check_epsilon(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"epsilon должен быть > 0, получено: {v}")
        return v

    @model_validator(mode="after")
    def check_preprocessor_eps(self) -> "GeneratorYamlConfig":
        if self.generator_type == "dpctgan" and not self.disabled_dp:
            if not (0.0 <= self.preprocessor_eps < self.epsilon):
                raise ValueError(
                    f"preprocessor_eps={self.preprocessor_eps} должен быть "
                    f"в [0, epsilon={self.epsilon})"
                )
        return self

    def to_dpctgan_config(self) -> DPCTGANConfig:
        return DPCTGANConfig(
            epsilon=self.epsilon,
            preprocessor_eps=self.preprocessor_eps,
            delta=self.delta,
            sigma=self.sigma,
            max_per_sample_grad_norm=self.max_per_sample_grad_norm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            pac=self.pac,
            embedding_dim=self.embedding_dim,
            generator_dim=tuple(self.generator_dim),
            discriminator_dim=tuple(self.discriminator_dim),
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            cuda=self.cuda,
            verbose=self.verbose,
            disabled_dp=self.disabled_dp,
            loss=self.loss,
            nullable=self.nullable,
            random_seed=self.random_seed,
        )

    def to_dptvae_config(self) -> DPTVAEConfig:
        return DPTVAEConfig(
            sigma=self.sigma,
            delta=self.delta,
            max_grad_norm=self.max_grad_norm,
            embedding_dim=self.embedding_dim,
            compress_dims=tuple(self.compress_dims),
            decompress_dims=tuple(self.decompress_dims),
            epochs=self.epochs,
            batch_size=self.batch_size,
            l2scale=self.l2scale,
            loss_factor=self.loss_factor,
            cuda=self.cuda,
            random_seed=self.random_seed,
        )

    def to_ctgan_config(self) -> CTGANConfig:
        return CTGANConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            pac=self.pac,
            embedding_dim=self.embedding_dim,
            generator_dim=tuple(self.generator_dim),
            discriminator_dim=tuple(self.discriminator_dim),
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            log_frequency=self.log_frequency,
            cuda=self.cuda,
            verbose=self.verbose,
            random_seed=self.random_seed,
        )

    def to_tvae_config(self) -> TVAEConfig:
        return TVAEConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            compress_dims=tuple(self.compress_dims),
            decompress_dims=tuple(self.decompress_dims),
            l2scale=self.l2scale,
            loss_factor=self.loss_factor,
            cuda=self.cuda,
            verbose=self.verbose,
            random_seed=self.random_seed,
        )

    def to_copulagan_config(self) -> CopulaGANConfig:
        return CopulaGANConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            pac=self.pac,
            embedding_dim=self.embedding_dim,
            generator_dim=tuple(self.generator_dim),
            discriminator_dim=tuple(self.discriminator_dim),
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            log_frequency=self.log_frequency,
            cuda=self.cuda,
            verbose=self.verbose,
            random_seed=self.random_seed,
        )


class UtilityYamlConfig(BaseModel):
    target_column: str
    task_type: str = "classification"
    drop_columns: List[str] = Field(default_factory=list)
    n_estimators: int = 100
    random_state: int = 42

    @field_validator("task_type")
    @classmethod
    def check_task_type(cls, v: str) -> str:
        if v not in ("classification", "regression"):
            raise ValueError(
                f"task_type должен быть 'classification' или 'regression', получено: '{v}'"
            )
        return v

    def to_utility_config(self) -> UtilityConfig:
        return UtilityConfig(
            target_column=self.target_column,
            task_type=self.task_type,  # type: ignore[arg-type]
            drop_columns=self.drop_columns,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )


class PrivacyYamlConfig(BaseModel):
    quasi_identifiers: List[str] = Field(default_factory=list)
    sensitive_attribute: Optional[str] = None
    compute_classical: bool = True
    compute_distance: bool = True
    compute_mia: bool = True
    distance_sample_size: int = 2000
    mia_sample_size: int = 1000

    def to_privacy_config(self) -> PrivacyConfig:
        return PrivacyConfig(
            quasi_identifiers=self.quasi_identifiers,
            sensitive_attribute=self.sensitive_attribute,
            compute_classical=self.compute_classical,
            compute_distance=self.compute_distance,
            compute_mia=self.compute_mia,
            distance_sample_size=self.distance_sample_size,
            mia_sample_size=self.mia_sample_size,
        )


class ThresholdsYamlConfig(BaseModel):
    max_utility_loss: float = 0.25
    max_mean_jsd: float = 0.40
    max_mia_auc: float = 0.60
    require_dcr_privacy_preserved: bool = True
    require_dp_enabled: bool = True
    max_spent_epsilon: Optional[float] = None

    def to_verdict_thresholds(self) -> VerdictThresholds:
        return VerdictThresholds(
            max_utility_loss=self.max_utility_loss,
            max_mean_jsd=self.max_mean_jsd,
            max_mia_auc=self.max_mia_auc,
            require_dcr_privacy_preserved=self.require_dcr_privacy_preserved,
            require_dp_enabled=self.require_dp_enabled,
            max_spent_epsilon=self.max_spent_epsilon,
        )


class DataSchemaYamlConfig(BaseModel):
    """
    Схема данных из конфига.

    Если categorical и continuous пусты — схема определяется автоматически
    через DataProcessor.detect_column_types(). Это поведение по умолчанию
    для новых датасетов без явно заданной схемы.
    """
    categorical: List[str] = Field(default_factory=list)
    continuous: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)

    # Параметры минимизации данных (data minimization по модели ПНСТ)
    direct_identifiers: List[str] = Field(default_factory=list)
    drop_high_cardinality: bool = False
    cardinality_threshold: float = 0.9

    @property
    def is_auto(self) -> bool:
        """True если схема не задана и нужна автодетекция."""
        return not self.categorical and not self.continuous


# ==============================================================================
# Корневая схема и загрузчик
# ==============================================================================

class AppConfig(BaseModel):
    """
    Полная конфигурация приложения — результат парсинга configs/adult.yaml.

    Методы .generator, .privacy, .utility, .thresholds возвращают
    готовые к использованию объекты для run_pipeline().
    """
    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: Optional[DBConfig] = None
    data_source: Optional[DBConfig] = None
    data_target: Optional[DBConfig] = None
    pipeline: PipelineConfig
    generator: GeneratorYamlConfig = Field(default_factory=GeneratorYamlConfig)
    utility: UtilityYamlConfig
    privacy: PrivacyYamlConfig = Field(default_factory=PrivacyYamlConfig)
    thresholds: ThresholdsYamlConfig = Field(default_factory=ThresholdsYamlConfig)
    data_schema: DataSchemaYamlConfig = Field(default_factory=DataSchemaYamlConfig)

    # ── Удобные методы для run_pipeline() ─────────────────────────────────────

    def get_generator_type(self) -> str:
        """Возвращает тип генератора из конфига."""
        return self.generator.generator_type

    def get_generator_config(self) -> "GeneratorConfig":
        """
        Возвращает конфиг в типе, соответствующем generator_type.
        pipeline.py использует его через build_generator().
        """
        t = self.generator.generator_type
        if t == "dpctgan":
            return self.generator.to_dpctgan_config()
        elif t == "dptvae":
            return self.generator.to_dptvae_config()
        elif t == "ctgan":
            return self.generator.to_ctgan_config()
        elif t == "tvae":
            return self.generator.to_tvae_config()
        elif t == "copulagan":
            return self.generator.to_copulagan_config()
        else:
            raise ValueError(f"Неизвестный generator_type: {t!r}")

    def get_privacy_config(self) -> PrivacyConfig:
        return self.privacy.to_privacy_config()

    def get_utility_config(self) -> UtilityConfig:
        return self.utility.to_utility_config()

    def get_thresholds(self) -> VerdictThresholds:
        return self.thresholds.to_verdict_thresholds()

    def get_n_synth_rows(self, n_train_rows: int) -> int:
        """
        Возвращает количество строк для генерации.
        Если n_synth_rows=0 в конфиге — совпадает с размером обучающей выборки.
        """
        if self.pipeline.n_synth_rows > 0:
            return self.pipeline.n_synth_rows
        return n_train_rows


# ==============================================================================
# Публичная функция загрузки
# ==============================================================================

def load_config(config_path: str = "configs/adult.yaml") -> AppConfig:
    """
    Загружает и валидирует конфигурацию из YAML-файла.

    Завершается с понятным сообщением об ошибке если:
      - файл не найден
      - YAML невалиден
      - обязательное поле отсутствует
      - значение не прошло валидацию (epsilon <= 0, holdout вне (0,1) и т.п.)

    Пример:
        cfg = load_config("configs/adult.yaml")
        synth_df, report = run_pipeline(
            real_df=df,
            synth_config=cfg.get_generator_config(),
            privacy_config=cfg.get_privacy_config(),
            utility_config=cfg.get_utility_config(),
            thresholds=cfg.get_thresholds(),
            dataset_name=cfg.pipeline.dataset_name,
            output_dir=cfg.paths.output_dir,
            holdout_size=cfg.pipeline.holdout_size,
            random_state=cfg.pipeline.random_state,
            categorical_columns=cfg.data_schema.categorical,
            continuous_columns=cfg.data_schema.continuous,
            n_synth_rows=cfg.get_n_synth_rows(n_train_rows),
        )
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {path.resolve()}\n"
            f"Убедитесь, что файл конфигурации лежит в нужной директории."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка парсинга YAML в {path}: {e}") from e

    if raw is None:
        raise ValueError(f"Файл конфигурации пуст: {path}")

    try:
        cfg = AppConfig.model_validate(raw)
    except Exception as e:
        # Pydantic даёт подробные сообщения, просто пробрасываем с контекстом
        raise ValueError(
            f"Ошибка валидации конфигурации в {path}:\n{e}"
        ) from e

    return cfg


# ==============================================================================
# Вспомогательная функция для quick_test режима
# ==============================================================================

def apply_quick_test(cfg: AppConfig) -> AppConfig:
    """
    Возвращает копию конфига с параметрами для быстрого теста.
    Не мутирует исходный объект.

    QUICK_TEST проверяет, что пайплайн проходит end-to-end без падений —
    не качество синтетики. Поэтому пороги вердикта намеренно ослаблены:
    50 эпох с DP-шумом не позволяют получить production-качество.

    Изменяет:
      pipeline:    sample_size = 5000
      generator:   epochs = 50, cuda = False
      privacy:     distance_sample_size = 500, mia_sample_size = 250
      thresholds:  max_utility_loss = 0.40, max_mean_jsd = 0.40,
                   max_mia_auc = 0.65, max_spent_epsilon = None
    """
    raw = cfg.model_dump()
    raw["pipeline"]["sample_size"] = 5000
    raw["generator"]["epochs"] = 50
    raw["generator"]["cuda"] = False
    raw["privacy"]["distance_sample_size"] = 500
    raw["privacy"]["mia_sample_size"] = 250
    # Пороги вердикта для режима smoke-test: проверяем только отсутствие краша
    raw["thresholds"]["max_utility_loss"] = 0.40
    raw["thresholds"]["max_mean_jsd"] = 0.40
    raw["thresholds"]["max_mia_auc"] = 0.65
    raw["thresholds"]["max_spent_epsilon"] = None
    return AppConfig.model_validate(raw)


# ==============================================================================
# Точка входа для отладки
# ==============================================================================

if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/adult.yaml"

    print(f"Загружаем конфиг: {config_file}\n")
    cfg = load_config(config_file)

    print("=== PIPELINE ===")
    print(f"  dataset:      {cfg.pipeline.dataset_name}")
    print(f"  data_path:    {cfg.pipeline.data_path}")
    print(f"  sample_size:  {cfg.pipeline.sample_size or 'все строки'}")
    print(f"  holdout_size: {cfg.pipeline.holdout_size}")
    print(f"  random_state: {cfg.pipeline.random_state}")

    print("\n=== GENERATOR ===")
    g = cfg.generator
    print(f"  epsilon:          {g.epsilon}")
    print(f"  preprocessor_eps: {g.preprocessor_eps}")
    print(f"  sigma:            {g.sigma}")
    print(f"  epochs:           {g.epochs}")
    print(f"  batch_size:       {g.batch_size}")
    print(f"  disabled_dp:      {g.disabled_dp}")
    print(f"  random_seed:      {g.random_seed}")

    print("\n=== PRIVACY ===")
    p = cfg.privacy
    print(f"  quasi_identifiers:    {p.quasi_identifiers}")
    print(f"  sensitive_attribute:  {p.sensitive_attribute}")
    print(f"  compute_mia:          {p.compute_mia}")

    print("\n=== UTILITY ===")
    u = cfg.utility
    print(f"  target_column: {u.target_column}")
    print(f"  task_type:     {u.task_type}")
    print(f"  drop_columns:  {u.drop_columns}")

    print("\n=== THRESHOLDS ===")
    t = cfg.thresholds
    print(f"  max_utility_loss:  {t.max_utility_loss}")
    print(f"  max_mean_jsd:      {t.max_mean_jsd}")
    print(f"  max_mia_auc:       {t.max_mia_auc}")
    print(f"  require_dp:        {t.require_dp_enabled}")

    print("\n=== DATA SCHEMA ===")
    s = cfg.data_schema
    if s.is_auto:
        print("  → Автодетекция (categorical/continuous не заданы)")
    else:
        print(f"  categorical: {s.categorical}")
        print(f"  continuous:  {s.continuous}")
        print(f"  exclude:     {s.exclude}")

    print("\n✓ Конфиг загружен успешно")
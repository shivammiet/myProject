"""
config.py — Centralised configuration management.

All settings are defined as Pydantic models so they are:
  • Strongly typed and validated at startup.
  • Loadable from environment variables (via pydantic-settings).
  • Mergeable with YAML overrides (via OmegaConf / PyYAML).

Usage
-----
# Load from environment + optional YAML override:
cfg = load_config("configs/default_config.yaml")

# Access nested settings:
print(cfg.training.learning_rate)
print(cfg.model.model_name)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(value: str | Path) -> Path:
    """Expand environment variables and make the path absolute."""
    return Path(os.path.expandvars(str(value))).expanduser()


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ProjectConfig(BaseSettings):
    """Top-level project identity settings."""

    model_config = SettingsConfigDict(env_prefix="PROJECT_", extra="ignore")

    name: str = Field(default="ai_project", description="Human-readable project name.")
    version: str = Field(default="0.1.0", description="Semantic version of the project.")
    seed: int = Field(default=42, ge=0, description="Global random seed for reproducibility.")
    description: str = Field(default="", description="Short project description.")


class DataConfig(BaseSettings):
    """Dataset paths and split ratios."""

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    data_dir: Path = Field(default=Path("./data"), description="Root directory for all datasets.")
    output_dir: Path = Field(default=Path("./outputs"), description="Directory for model outputs.")
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"), description="Directory for model checkpoints."
    )
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0, description="Fraction of data for training.")
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0, description="Fraction of data for validation.")
    test_split: float = Field(default=0.1, gt=0.0, lt=1.0, description="Fraction of data for testing.")

    # Preprocessing
    max_samples: Optional[int] = Field(
        default=None, ge=1, description="Cap dataset at this many samples (useful for debugging)."
    )
    num_workers: int = Field(default=4, ge=0, description="DataLoader worker processes.")
    prefetch_factor: int = Field(default=2, ge=1, description="DataLoader prefetch factor.")

    @model_validator(mode="after")
    def _splits_sum_to_one(self) -> "DataConfig":
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total}"
            )
        return self

    @field_validator("data_dir", "output_dir", "checkpoint_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return _resolve_path(v)


class ModelConfig(BaseSettings):
    """Model architecture settings."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model identifier or local path.",
    )
    num_labels: int = Field(default=2, ge=1, description="Number of output classes / labels.")
    max_length: int = Field(
        default=512, ge=1, le=131072, description="Maximum token sequence length."
    )
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout probability.")
    hidden_size: Optional[int] = Field(
        default=None, description="Override model hidden size (None = use model default)."
    )
    use_pretrained: bool = Field(
        default=True, description="Load pretrained weights from HuggingFace Hub."
    )
    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom modelling code from the Hub."
    )


class TrainingConfig(BaseSettings):
    """Training loop hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="TRAINING_", extra="ignore")

    batch_size: int = Field(default=32, ge=1, description="Per-device training batch size.")
    eval_batch_size: Optional[int] = Field(
        default=None, ge=1, description="Evaluation batch size (defaults to batch_size)."
    )
    learning_rate: float = Field(default=1e-4, gt=0.0, description="Peak learning rate.")
    num_epochs: int = Field(default=10, ge=1, description="Total training epochs.")
    warmup_steps: int = Field(
        default=500, ge=0, description="Linear LR warmup steps."
    )
    weight_decay: float = Field(default=0.01, ge=0.0, description="AdamW weight decay.")
    max_grad_norm: float = Field(
        default=1.0, gt=0.0, description="Gradient clipping max norm."
    )
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Steps to accumulate gradients before an optimiser step."
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description=(
            "Compute device. 'auto' selects CUDA → MPS → CPU in order of availability."
        ),
    )
    fp16: bool = Field(default=False, description="Use float16 mixed-precision training.")
    bf16: bool = Field(default=False, description="Use bfloat16 mixed-precision training.")
    early_stopping_patience: int = Field(
        default=5, ge=1, description="Stop training after this many epochs without improvement."
    )
    save_every_n_epochs: int = Field(
        default=1, ge=1, description="Checkpoint frequency in epochs."
    )
    eval_every_n_steps: Optional[int] = Field(
        default=None, ge=1, description="Evaluate every N steps (None = end of each epoch)."
    )

    @field_validator("device", mode="before")
    @classmethod
    def _auto_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v

    @model_validator(mode="after")
    def _resolve_eval_batch(self) -> "TrainingConfig":
        if self.eval_batch_size is None:
            object.__setattr__(self, "eval_batch_size", self.batch_size)
        return self

    @model_validator(mode="after")
    def _precision_exclusive(self) -> "TrainingConfig":
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be True.")
        return self


class LoggingConfig(BaseSettings):
    """Logging and experiment tracking settings."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Root logging level."
    )
    log_dir: Path = Field(default=Path("./logs"), description="Directory for log files.")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging.")
    wandb_project: str = Field(
        default="my-ai-project", description="W&B project name."
    )
    wandb_entity: Optional[str] = Field(
        default=None, description="W&B team / user entity (None = personal account)."
    )
    use_mlflow: bool = Field(default=False, description="Enable MLflow experiment tracking.")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking server URI."
    )
    mlflow_experiment: str = Field(
        default="default", description="MLflow experiment name."
    )

    @field_validator("log_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return _resolve_path(v)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseSettings):
    """
    Root application configuration.

    Loads settings from (in increasing priority):
      1. Pydantic field defaults
      2. Environment variables (via pydantic-settings)
      3. YAML file (merged in :func:`load_config`)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_config(yaml_path: Optional[str | Path] = None) -> AppConfig:
    """
    Build an :class:`AppConfig` from environment variables and an optional YAML file.

    The YAML file values override environment defaults but can themselves be
    overridden by explicit environment variables set at shell level.

    Parameters
    ----------
    yaml_path:
        Path to a YAML config file (e.g. ``configs/default_config.yaml``).
        If *None*, only environment variables and defaults are used.

    Returns
    -------
    AppConfig
        A fully validated configuration object.
    """
    if yaml_path is None:
        return AppConfig()

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open() as fh:
        yaml_data: dict[str, Any] = yaml.safe_load(fh) or {}

    # Flatten nested YAML into env-var-style kwargs for pydantic-settings.
    # e.g. {"training": {"batch_size": 64}} → passed as nested dict directly.
    return AppConfig.model_validate(yaml_data)

"""
ai_project — Production-ready AI/ML project package.

This package provides a structured foundation for building, training,
and deploying machine learning models with best practices baked in.

Modules:
    config      — Pydantic-based configuration management
    models      — Base model classes and architectures
    data        — Data loading and preprocessing utilities
    training    — Training loop, checkpointing, and early stopping
    inference   — Model inference and prediction utilities
    utils       — Logging, metrics, and general helpers
"""

from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Package version — read from installed metadata (set in pyproject.toml)
# ---------------------------------------------------------------------------
try:
    __version__: str = version("ai-project")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source tree)
    __version__ = "0.1.0.dev0"

__author__ = "Your Name"
__email__ = "you@example.com"

# ---------------------------------------------------------------------------
# Public API — explicit re-exports for IDE auto-complete and star-imports
# ---------------------------------------------------------------------------
from ai_project.config import (  # noqa: E402
    AppConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "__version__",
    # Config
    "AppConfig",
    "ProjectConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "LoggingConfig",
    "load_config",
]

"""ai_project.training — Training loop, checkpointing, and early stopping."""

from ai_project.training.trainer import EarlyStopping, Trainer, TrainingResult

__all__ = ["Trainer", "EarlyStopping", "TrainingResult"]

"""
base_model.py — Abstract base class for all models in this project.

Every model you build should inherit from :class:`BaseModel` and implement
the :meth:`forward` method. The base class provides:

  - save / load helpers (state dict + config JSON)
  - parameter counting utility
  - device-aware convenience property
  - a typed :class:`ModelOutput` dataclass for consistent return shapes
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """
    A generic container for model forward-pass results.

    Attributes
    ----------
    logits:
        Raw (un-normalised) class scores of shape ``(batch, num_labels)``.
    loss:
        Scalar loss tensor when labels are provided, otherwise *None*.
    hidden_states:
        Optional tuple of hidden-state tensors from each layer.
    attentions:
        Optional tuple of attention weight tensors from each layer.
    extras:
        Arbitrary additional values keyed by name.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (tensors converted to Python lists)."""
        return {
            "logits": self.logits.tolist(),
            "loss": self.loss.item() if self.loss is not None else None,
        }


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all project models.

    Subclasses must implement :meth:`forward`.  All other methods
    (``save``, ``load``, ``count_parameters``, …) are provided here
    so you can focus on the architecture.

    Parameters
    ----------
    config:
        Arbitrary keyword arguments that describe the model.  They are
        stored as ``self.model_config`` and persisted alongside weights
        when :meth:`save` is called.

    Example
    -------
    >>> class MyClassifier(BaseModel):
    ...     def __init__(self, hidden_size: int = 256, num_labels: int = 2):
    ...         super().__init__(hidden_size=hidden_size, num_labels=num_labels)
    ...         self.fc = nn.Linear(hidden_size, num_labels)
    ...
    ...     def forward(self, x: torch.Tensor, labels=None) -> ModelOutput:
    ...         logits = self.fc(x)
    ...         loss = None
    ...         if labels is not None:
    ...             loss = nn.functional.cross_entropy(logits, labels)
    ...         return ModelOutput(logits=logits, loss=loss)
    """

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        # Store raw config for serialisation purposes.
        self.model_config: dict[str, Any] = model_config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> ModelOutput:
        """
        Run the forward pass.

        Subclasses must return a :class:`ModelOutput` instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, save_dir: str | Path, *, prefix: str = "model") -> Path:
        """
        Persist model weights and configuration to *save_dir*.

        Two files are written:

        * ``<prefix>.pt``   — PyTorch state dict
        * ``<prefix>.json`` — model_config dict (JSON)

        Parameters
        ----------
        save_dir:
            Directory that will be created if it does not exist.
        prefix:
            File-name prefix (default ``"model"``).

        Returns
        -------
        Path
            The directory where the files were written.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        weights_path = save_dir / f"{prefix}.pt"
        config_path = save_dir / f"{prefix}.json"

        torch.save(self.state_dict(), weights_path)
        config_path.write_text(json.dumps(self.model_config, indent=2))

        return save_dir

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        *,
        prefix: str = "model",
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> "BaseModel":
        """
        Reconstruct a model from files written by :meth:`save`.

        Parameters
        ----------
        save_dir:
            Directory containing ``<prefix>.pt`` and ``<prefix>.json``.
        prefix:
            File-name prefix used when saving.
        map_location:
            Passed to :func:`torch.load` (e.g. ``"cpu"`` when loading a
            GPU checkpoint on a CPU-only machine).
        strict:
            Whether to enforce that the keys in the state dict match
            exactly (passed to :meth:`nn.Module.load_state_dict`).

        Returns
        -------
        BaseModel
            A new model instance with weights restored.
        """
        save_dir = Path(save_dir)
        weights_path = save_dir / f"{prefix}.pt"
        config_path = save_dir / f"{prefix}.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        model_config: dict[str, Any] = json.loads(config_path.read_text())
        model = cls(**model_config)

        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict, strict=strict)

        return model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self, *, only_trainable: bool = True) -> int:
        """
        Return the number of (optionally only trainable) parameters.

        Parameters
        ----------
        only_trainable:
            When *True* (default) count only parameters where
            ``requires_grad=True``.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad if only_trainable else True)
        )

    def parameter_summary(self) -> str:
        """Return a human-readable string with trainable vs total parameter counts."""
        trainable = self.count_parameters(only_trainable=True)
        total = self.count_parameters(only_trainable=False)
        frozen = total - trainable

        def _fmt(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.2f}B"
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}M"
            if n >= 1_000:
                return f"{n / 1_000:.2f}K"
            return str(n)

        return (
            f"Parameters — total: {_fmt(total)}, "
            f"trainable: {_fmt(trainable)}, frozen: {_fmt(frozen)}"
        )

    @property
    def device(self) -> torch.device:
        """The device of the first parameter in the model."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __repr__(self) -> str:  # pragma: no cover
        base = super().__repr__()
        return f"{base}\n{self.parameter_summary()}"

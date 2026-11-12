"""
trainer.py — Training loop with early stopping, checkpointing, and experiment tracking.

The :class:`Trainer` orchestrates the standard training workflow:

  1. Iterate over epochs.
  2. Run :meth:`train_epoch` → forward pass, backward pass, optimiser step.
  3. Run :meth:`evaluate` → compute metrics on validation set.
  4. Checkpoint the model if validation metric improved.
  5. Stop early if :class:`EarlyStopping` threshold is met.
  6. Optionally log to Weights & Biases and / or MLflow.

Example
-------
>>> trainer = Trainer(
...     model=my_model,
...     config=app_config,
...     train_loader=train_dl,
...     val_loader=val_dl,
... )
>>> result = trainer.train()
>>> print(f"Best val loss: {result.best_val_loss:.4f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_project.config import AppConfig
from ai_project.models.base_model import BaseModel
from ai_project.utils.logger import get_logger
from ai_project.utils.metrics import compute_classification_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary statistics returned by :meth:`Trainer.train`."""

    best_val_loss: float = float("inf")
    best_val_metrics: dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    total_epochs_run: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a metric and signal when training should stop.

    Parameters
    ----------
    patience:
        Number of epochs to wait without improvement before stopping.
    min_delta:
        Minimum absolute improvement to count as an improvement.
    mode:
        ``"min"`` for loss-like metrics, ``"max"`` for accuracy-like metrics.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> bool:
        """
        Update state with a new metric value.

        Returns
        -------
        bool
            *True* if the metric improved, *False* otherwise.
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )

        if improved:
            self._best = metric
            self._counter = 0
            return True
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
            return False

    @property
    def best(self) -> float:
        return self._best

    @property
    def epochs_without_improvement(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Generic model trainer with checkpointing and experiment tracking.

    Parameters
    ----------
    model:
        A :class:`~ai_project.models.base_model.BaseModel` to train.
    config:
        The root :class:`~ai_project.config.AppConfig`.
    train_loader:
        DataLoader for the training split.
    val_loader:
        DataLoader for the validation split (used for early stopping and
        checkpoint selection).
    optimizer:
        Pre-built optimiser.  If *None*, AdamW is constructed from
        ``config.training``.
    scheduler:
        Learning-rate scheduler.  If *None*, no scheduling is applied.
    loss_fn:
        Loss function.  If *None*, ``nn.CrossEntropyLoss()`` is used.
    metric_fn:
        Callable ``(logits, labels) -> dict[str, float]`` for extra metrics.
        If *None*, accuracy is computed automatically.
    """

    def __init__(
        self,
        model: BaseModel,
        config: AppConfig,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        metric_fn: Optional[Callable[..., dict[str, float]]] = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Optimiser
        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = scheduler
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metric_fn = metric_fn or self._default_metrics

        # Mixed precision
        self._use_amp = config.training.fp16 or config.training.bf16
        self._amp_dtype = torch.float16 if config.training.fp16 else torch.bfloat16

        # Early stopping
        self._early_stop = EarlyStopping(
            patience=config.training.early_stopping_patience, mode="min"
        )

        # Checkpoint dir
        self._ckpt_dir = config.data.checkpoint_dir
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Experiment tracking
        self._wandb_run: Any = None
        self._mlflow_active = False
        self._setup_tracking()

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """
        Run the full training loop.

        Returns
        -------
        TrainingResult
            Best checkpoint metrics and per-epoch history.
        """
        result = TrainingResult()
        num_epochs = self.cfg.training.num_epochs

        logger.info(
            f"Starting training | epochs={num_epochs} | "
            f"device={self.device} | "
            f"{self.model.parameter_summary()}"
        )

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()

            # --- Train ---
            train_loss, train_metrics = self.train_epoch(epoch)

            # --- Validate ---
            val_loss, val_metrics = self.evaluate(self.val_loader, prefix="val")

            elapsed = time.perf_counter() - epoch_start
            self._log_epoch(epoch, train_loss, train_metrics, val_loss, val_metrics, elapsed)

            # Record history
            result.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            # --- Checkpoint ---
            improved = self._early_stop.step(val_loss)
            if improved:
                result.best_val_loss = val_loss
                result.best_val_metrics = val_metrics
                result.best_epoch = epoch
                self._save_checkpoint("best_model")
                logger.info(f"  ✓ New best checkpoint saved (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            if epoch % self.cfg.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}")

            # --- Scheduler step ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Early stopping ---
            if self._early_stop.should_stop:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"({self._early_stop.patience} epochs without improvement)."
                )
                break

        result.total_epochs_run = epoch
        self._finish_tracking(result)
        return result

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """
        Train for a single epoch.

        Parameters
        ----------
        epoch:
            Current epoch number (1-based, used only for logging).

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(mean_loss, metrics_dict)``
        """
        self.model.train()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        accum_steps = self.cfg.training.gradient_accumulation_steps
        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp and self.device.type == "cuda")

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [train]",
            leave=False,
            unit="batch",
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.get("labels")

            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                output = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss: torch.Tensor = (
                    output.loss
                    if output.loss is not None
                    else self.loss_fn(output.logits, labels)
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.max_grad_norm
                )
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps  # undo the / accum_steps scaling
            if labels is not None:
                all_logits.append(output.logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

            progress.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        mean_loss = total_loss / len(self.train_loader)

        metrics: dict[str, float] = {}
        if all_logits and all_labels:
            cat_logits = torch.cat(all_logits)
            cat_labels = torch.cat(all_labels)
            metrics = self.metric_fn(cat_logits, cat_labels)

        return mean_loss, metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, loader: DataLoader, *, prefix: str = "val"  # type: ignore[type-arg]
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate the model on *loader*.

        Parameters
        ----------
        loader:
            DataLoader containing the evaluation data.
        prefix:
            String prefix added to metric keys in the log output.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(mean_loss, metrics_dict)``
        """
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[{prefix}]", leave=False, unit="batch"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                labels = batch.get("labels")

                output = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss: torch.Tensor = (
                    output.loss
                    if output.loss is not None
                    else self.loss_fn(output.logits, labels)
                )
                total_loss += loss.item()
                if labels is not None:
                    all_logits.append(output.logits.cpu())
                    all_labels.append(labels.cpu())

        mean_loss = total_loss / max(len(loader), 1)

        metrics: dict[str, float] = {}
        if all_logits and all_labels:
            cat_logits = torch.cat(all_logits)
            cat_labels = torch.cat(all_labels)
            metrics = self.metric_fn(cat_logits, cat_labels)

        return mean_loss, metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        save_path = self._ckpt_dir / name
        self.model.save(save_path)
        # Also save optimiser state
        torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")

    def load_checkpoint(self, checkpoint_dir: str | Path, *, strict: bool = True) -> None:
        """Load a saved checkpoint into the current model and optimiser."""
        checkpoint_dir = Path(checkpoint_dir)
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=self.device, weights_only=True),
            strict=strict,
        )
        opt_path = checkpoint_dir / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=True)
            )

    # ------------------------------------------------------------------
    # Experiment tracking
    # ------------------------------------------------------------------

    def _setup_tracking(self) -> None:
        """Initialise W&B and / or MLflow if enabled in config."""
        if self.cfg.logging.use_wandb:
            try:
                import wandb  # type: ignore[import-untyped]

                self._wandb_run = wandb.init(
                    project=self.cfg.logging.wandb_project,
                    entity=self.cfg.logging.wandb_entity,
                    name=self.cfg.project.name,
                    config=self.cfg.model_dump(),
                )
                logger.info("W&B run initialised.")
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging.")

        if self.cfg.logging.use_mlflow:
            try:
                import mlflow  # type: ignore[import-untyped]

                mlflow.set_tracking_uri(self.cfg.logging.mlflow_tracking_uri)
                mlflow.set_experiment(self.cfg.logging.mlflow_experiment)
                mlflow.start_run(run_name=self.cfg.project.name)
                self._mlflow_active = True
                logger.info("MLflow run started.")
            except ImportError:
                logger.warning("mlflow not installed; skipping MLflow logging.")

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: dict[str, float],
        val_loss: float,
        val_metrics: dict[str, float],
        elapsed: float,
    ) -> None:
        log_data = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in log_data.items() if k != "epoch")
        logger.info(f"Epoch {epoch:3d}/{self.cfg.training.num_epochs} | {metrics_str} | {elapsed:.1f}s")

        if self._wandb_run is not None:
            self._wandb_run.log(log_data, step=epoch)

        if self._mlflow_active:
            try:
                import mlflow

                mlflow.log_metrics({k: v for k, v in log_data.items() if isinstance(v, float)}, step=epoch)
            except Exception:
                pass

    def _finish_tracking(self, result: TrainingResult) -> None:
        if self._wandb_run is not None:
            self._wandb_run.summary.update(
                {"best_val_loss": result.best_val_loss, "best_epoch": result.best_epoch}
            )
            self._wandb_run.finish()

        if self._mlflow_active:
            try:
                import mlflow

                mlflow.log_metrics(
                    {"best_val_loss": result.best_val_loss, "best_epoch": float(result.best_epoch)}
                )
                mlflow.end_run()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_metrics(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Compute accuracy from raw logits and integer labels."""
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc}

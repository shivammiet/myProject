"""
predictor.py — High-level inference interface for trained models.

The :class:`Predictor` loads a saved model and exposes:

  - :meth:`predict`        — single-item inference
  - :meth:`batch_predict`  — batched inference with progress bar
  - :meth:`stream_predict` — generator-based streaming for LLMs
  - :meth:`predict_proba`  — class probabilities (classification)

Example
-------
>>> predictor = Predictor.from_checkpoint(
...     "checkpoints/best_model",
...     model_cls=MyClassifier,
...     tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
...     device="cuda",
... )
>>> result = predictor.predict("This film was absolutely fantastic!")
>>> print(result.label, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Type

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ai_project.models.base_model import BaseModel, ModelOutput
from ai_project.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Container for a single prediction.

    Attributes
    ----------
    label:
        Predicted class index (classification) or *None* for regression / generation.
    confidence:
        Softmax probability of the predicted class.
    probabilities:
        Full probability distribution over all classes.
    raw_logits:
        Un-normalised logits returned by the model.
    metadata:
        Any extra information from the model (hidden states, attention, etc.).
    """

    label: Optional[int] = None
    confidence: float = 0.0
    probabilities: list[float] = field(default_factory=list)
    raw_logits: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_class(self) -> Optional[int]:
        """Alias for :attr:`label`."""
        return self.label


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Wraps a trained :class:`~ai_project.models.base_model.BaseModel` for inference.

    Prefer constructing via :meth:`from_checkpoint` rather than
    instantiating directly.

    Parameters
    ----------
    model:
        A loaded model (already on the target device and in eval mode).
    tokenizer:
        A HuggingFace ``PreTrainedTokenizer`` for text inputs.
        Can be *None* for non-text models.
    device:
        Target device string.
    max_length:
        Maximum tokenisation length.
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer: Any = None,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        model_cls: Type[BaseModel],
        *,
        tokenizer: Any = None,
        device: str = "cpu",
        max_length: int = 512,
        prefix: str = "model",
    ) -> "Predictor":
        """
        Load a model from a checkpoint directory created by
        :meth:`~ai_project.models.base_model.BaseModel.save`.

        Parameters
        ----------
        checkpoint_dir:
            Path to the directory containing ``model.pt`` and ``model.json``.
        model_cls:
            The :class:`BaseModel` subclass to instantiate.
        tokenizer:
            Optional HuggingFace tokenizer.
        device:
            Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
        max_length:
            Maximum tokenisation length.
        prefix:
            File-name prefix used when saving.

        Returns
        -------
        Predictor
            A fully initialised predictor ready for inference.
        """
        logger.info(f"Loading model from {checkpoint_dir} …")
        model = model_cls.load(
            checkpoint_dir,
            prefix=prefix,
            map_location=device,
        )
        logger.info(f"Model loaded. {model.parameter_summary()}")
        return cls(model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, text_or_input: str | dict[str, Any]) -> PredictionResult:
        """
        Run inference on a single example.

        Parameters
        ----------
        text_or_input:
            Either a raw string (tokenized automatically) or a dict of
            pre-tokenised tensors matching the model's expected inputs.

        Returns
        -------
        PredictionResult
        """
        batch_input = self._prepare_input(text_or_input)
        output: ModelOutput = self.model(**batch_input)
        return self._output_to_result(output)

    @torch.no_grad()
    def predict_proba(self, text_or_input: str | dict[str, Any]) -> list[float]:
        """
        Return class probabilities for a single example.

        Returns
        -------
        list[float]
            Softmax probabilities over all classes.
        """
        return self.predict(text_or_input).probabilities

    @torch.no_grad()
    def batch_predict(
        self,
        inputs: list[str | dict[str, Any]],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[PredictionResult]:
        """
        Run inference on a list of inputs in mini-batches.

        Parameters
        ----------
        inputs:
            List of raw strings or pre-tokenised input dicts.
        batch_size:
            Number of examples per forward pass.
        show_progress:
            Show a tqdm progress bar.

        Returns
        -------
        list[PredictionResult]
            One result per input, in the same order.
        """
        results: list[PredictionResult] = []

        batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        progress = tqdm(batches, desc="Predicting", disable=not show_progress)

        for batch in progress:
            if isinstance(batch[0], str) and self.tokenizer is not None:
                # Tokenise the whole mini-batch at once for efficiency
                encoding: dict[str, torch.Tensor] = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_input = {k: v.to(self.device) for k, v in encoding.items()}
                output: ModelOutput = self.model(**batch_input)
                probs = F.softmax(output.logits, dim=-1)
                preds = probs.argmax(dim=-1)
                for i in range(len(batch)):
                    results.append(
                        PredictionResult(
                            label=preds[i].item(),
                            confidence=probs[i, preds[i]].item(),
                            probabilities=probs[i].tolist(),
                            raw_logits=output.logits[i].tolist(),
                        )
                    )
            else:
                # Fall back to per-example inference
                for item in batch:
                    results.append(self.predict(item))

        return results

    def stream_predict(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Generator-based token-by-token streaming for language models.

        This method is intended for use with HuggingFace generative models
        exposed via the :attr:`model` attribute. If the underlying model
        does not support ``generate()``, it raises :class:`NotImplementedError`.

        Parameters
        ----------
        prompt:
            Input text to continue / respond to.
        max_new_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature (1.0 = no scaling).
        top_p:
            Nucleus sampling probability threshold.

        Yields
        ------
        str
            Decoded tokens as they are generated.
        """
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for streaming generation.")

        if not hasattr(self.model, "generate"):
            raise NotImplementedError(
                "stream_predict requires the model to expose a 'generate' method "
                "(e.g. a HuggingFace PreTrainedModel)."
            )

        from transformers import TextIteratorStreamer  # type: ignore[import-untyped]
        import threading

        inputs: dict[str, torch.Tensor] = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature != 1.0 or top_p != 1.0,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        try:
            yield from streamer
        finally:
            thread.join()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, text_or_input: str | dict[str, Any]) -> dict[str, torch.Tensor]:
        """Tokenise a string or move a pre-built dict to the target device."""
        if isinstance(text_or_input, str):
            if self.tokenizer is None:
                raise ValueError("A tokenizer must be set to process raw string inputs.")
            encoding: dict[str, torch.Tensor] = self.tokenizer(
                text_or_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            return {k: v.to(self.device) for k, v in encoding.items()}

        # Pre-tokenised dict — just move tensors to device
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in text_or_input.items()
        }

    def _output_to_result(self, output: ModelOutput) -> PredictionResult:
        """Convert a :class:`ModelOutput` into a :class:`PredictionResult`."""
        logits = output.logits.squeeze(0)  # remove batch dim for single example
        probs = F.softmax(logits, dim=-1)
        label = int(probs.argmax().item())
        return PredictionResult(
            label=label,
            confidence=float(probs[label].item()),
            probabilities=probs.tolist(),
            raw_logits=logits.tolist(),
        )

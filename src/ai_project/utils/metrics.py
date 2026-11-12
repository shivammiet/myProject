"""
metrics.py — Evaluation metric functions for classification and text generation.

All functions accept either numpy arrays or PyTorch tensors unless noted.

Classification
--------------
- :func:`compute_accuracy`
- :func:`compute_f1`
- :func:`compute_precision`
- :func:`compute_recall`
- :func:`compute_classification_metrics`  (all-in-one)

Text Generation
---------------
- :func:`compute_bleu`
- :func:`compute_rouge`
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert any array-like to a 1-D numpy int array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _logits_to_preds(logits_or_preds: ArrayLike) -> np.ndarray:
    """
    If *logits_or_preds* is 2-D, take the argmax to get class indices.
    If already 1-D, return as-is.
    """
    arr = _to_numpy(logits_or_preds)
    if arr.ndim == 2:
        return arr.argmax(axis=-1)
    return arr


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_accuracy(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    from_logits: bool = True,
) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions:
        Model logits (shape ``(N, C)``) or predicted class indices (shape ``(N,)``).
    labels:
        Ground-truth class indices (shape ``(N,)``).
    from_logits:
        If *True*, ``predictions`` are raw logits and argmax is applied.

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.
    """
    preds = _logits_to_preds(predictions) if from_logits else _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return float(accuracy_score(lbls, preds))


def compute_f1(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    average: str = "weighted",
    from_logits: bool = True,
) -> float:
    """
    Compute F1 score.

    Parameters
    ----------
    predictions:
        Model logits or predicted class indices.
    labels:
        Ground-truth class indices.
    average:
        Averaging strategy passed to scikit-learn
        (``"weighted"``, ``"macro"``, ``"micro"``, ``"binary"``).
    from_logits:
        Apply argmax to *predictions* if *True*.

    Returns
    -------
    float
    """
    preds = _logits_to_preds(predictions) if from_logits else _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return float(f1_score(lbls, preds, average=average, zero_division=0))


def compute_precision(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    average: str = "weighted",
    from_logits: bool = True,
) -> float:
    """Compute precision score. See :func:`compute_f1` for parameter docs."""
    preds = _logits_to_preds(predictions) if from_logits else _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return float(precision_score(lbls, preds, average=average, zero_division=0))


def compute_recall(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    average: str = "weighted",
    from_logits: bool = True,
) -> float:
    """Compute recall score. See :func:`compute_f1` for parameter docs."""
    preds = _logits_to_preds(predictions) if from_logits else _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return float(recall_score(lbls, preds, average=average, zero_division=0))


def compute_classification_metrics(
    predictions: ArrayLike,
    labels: ArrayLike,
    *,
    average: str = "weighted",
    from_logits: bool = True,
) -> dict[str, float]:
    """
    Compute accuracy, F1, precision, and recall in one call.

    Parameters
    ----------
    predictions:
        Model logits (shape ``(N, C)``) or predicted class indices (shape ``(N,)``).
    labels:
        Ground-truth class indices.
    average:
        Averaging strategy for multi-class metrics.
    from_logits:
        Apply argmax to *predictions* if *True*.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``f1``, ``precision``, ``recall``.
    """
    preds = _logits_to_preds(predictions) if from_logits else _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return {
        "accuracy": float(accuracy_score(lbls, preds)),
        "f1": float(f1_score(lbls, preds, average=average, zero_division=0)),
        "precision": float(precision_score(lbls, preds, average=average, zero_division=0)),
        "recall": float(recall_score(lbls, preds, average=average, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Text generation metrics
# ---------------------------------------------------------------------------

def compute_bleu(
    predictions: list[str],
    references: list[str | list[str]],
    *,
    max_order: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """
    Compute corpus-level BLEU score using the ``nltk`` library.

    Parameters
    ----------
    predictions:
        List of generated / hypothesis strings.
    references:
        List of reference strings or lists of reference strings
        (one per prediction, multiple references supported).
    max_order:
        Maximum n-gram order (default 4 for BLEU-4).
    smooth:
        Apply Chen & Cherry (2014) smoothing to avoid zero scores.

    Returns
    -------
    dict[str, float]
        ``{"bleu": <score>}`` where the score is in ``[0.0, 1.0]``.
    """
    try:
        from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
            corpus_bleu,
            SmoothingFunction,
        )
    except ImportError as exc:
        raise ImportError(
            "nltk is required for BLEU computation. Install with: pip install nltk"
        ) from exc

    tokenised_preds = [pred.lower().split() for pred in predictions]
    tokenised_refs = [
        [ref.lower().split()] if isinstance(ref, str) else [r.lower().split() for r in ref]
        for ref in references
    ]

    smoothing = SmoothingFunction().method1 if smooth else None
    score = corpus_bleu(
        tokenised_refs,
        tokenised_preds,
        weights=tuple(1.0 / max_order for _ in range(max_order)),
        smoothing_function=smoothing,
    )
    return {"bleu": float(score)}


def compute_rouge(
    predictions: list[str],
    references: list[str],
    *,
    rouge_types: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Compute ROUGE scores using the ``rouge_score`` library.

    Parameters
    ----------
    predictions:
        List of generated / hypothesis strings.
    references:
        List of reference strings (one per prediction).
    rouge_types:
        ROUGE variants to compute.  Defaults to
        ``["rouge1", "rouge2", "rougeL"]``.

    Returns
    -------
    dict[str, float]
        One entry per ROUGE type (F-measure), e.g.
        ``{"rouge1": 0.52, "rouge2": 0.31, "rougeL": 0.48}``.
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "rouge_score is required. Install with: pip install rouge-score"
        ) from exc

    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    aggregated: dict[str, list[float]] = {rt: [] for rt in rouge_types}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for rt in rouge_types:
            aggregated[rt].append(scores[rt].fmeasure)

    return {rt: float(np.mean(vals)) for rt, vals in aggregated.items()}

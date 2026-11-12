"""
data_loader.py — Flexible data loading for HuggingFace datasets, CSV, and JSON.

Provides :class:`AIDataLoader` which wraps the HuggingFace ``datasets``
library and adds helpers for:

  - Loading from the Hub or from local CSV / JSON files
  - Tokenisation via a HuggingFace tokenizer
  - Train / validation / test splitting
  - PyTorch :class:`~torch.utils.data.DataLoader` construction

Example
-------
>>> loader = AIDataLoader(
...     tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
...     max_length=128,
...     batch_size=32,
... )
>>> splits = loader.load_hf_dataset("imdb")
>>> train_dl = loader.get_dataloader(splits["train"], shuffle=True)
"""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DataSplit enum — used as dict keys throughout the module
# ---------------------------------------------------------------------------

class DataSplit(str, Enum):
    """Canonical names for dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# ---------------------------------------------------------------------------
# Simple in-memory dataset wrapper
# ---------------------------------------------------------------------------

class InMemoryDataset(Dataset):  # type: ignore[type-arg]
    """
    A minimal :class:`~torch.utils.data.Dataset` backed by a list of dicts.

    Each item in *records* must be a dict with at least the keys expected
    by the collation function.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class AIDataLoader:
    """
    A unified interface for loading and preprocessing datasets.

    Parameters
    ----------
    tokenizer:
        A HuggingFace ``PreTrainedTokenizer`` instance.  Can be *None*
        if you are working with pre-tokenised data.
    max_length:
        Maximum token sequence length for truncation / padding.
    batch_size:
        Default batch size for DataLoader creation.
    num_workers:
        Number of worker processes for :class:`~torch.utils.data.DataLoader`.
    pin_memory:
        Pin memory in DataLoaders (recommended when training on GPU).
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_hf_dataset(
        self,
        dataset_name: str,
        *,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        text_column: str = "text",
        label_column: str = "label",
        max_samples: Optional[int] = None,
    ) -> dict[DataSplit, InMemoryDataset]:
        """
        Load a dataset from the HuggingFace Hub.

        Parameters
        ----------
        dataset_name:
            Hub dataset identifier (e.g. ``"imdb"``, ``"glue"``).
        config_name:
            Dataset configuration name (e.g. ``"sst2"`` for GLUE).
        split:
            A specific split to load (``"train"``, ``"test"``…).
            If *None*, all available splits are loaded.
        text_column:
            Column name that holds the raw text.
        label_column:
            Column name that holds the class label.
        max_samples:
            Truncate each split to at most this many samples.

        Returns
        -------
        dict[DataSplit, InMemoryDataset]
            Mapping from split name to in-memory dataset.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install it with: pip install datasets"
            ) from exc

        raw = load_dataset(dataset_name, config_name, split=split)

        # Normalise to a dict of splits
        if hasattr(raw, "items"):
            split_map: dict[str, Any] = dict(raw.items())  # type: ignore[arg-type]
        else:
            # A single split was requested
            split_map = {split or "train": raw}

        result: dict[DataSplit, InMemoryDataset] = {}
        for split_name, ds in split_map.items():
            records = self._hf_to_records(
                ds,
                text_column=text_column,
                label_column=label_column,
                max_samples=max_samples,
            )
            key = DataSplit(split_name) if split_name in DataSplit._value2member_map_ else DataSplit.TRAIN
            result[key] = InMemoryDataset(records)

        return result

    def load_csv(
        self,
        file_path: str | Path,
        *,
        text_column: str = "text",
        label_column: str = "label",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_samples: Optional[int] = None,
    ) -> dict[DataSplit, InMemoryDataset]:
        """
        Load a CSV file and split into train / validation / test sets.

        Parameters
        ----------
        file_path:
            Path to the CSV file.
        text_column:
            Column name for input text.
        label_column:
            Column name for target labels.
        train_ratio:
            Fraction of data for training.
        val_ratio:
            Fraction of data for validation (remainder goes to test).
        max_samples:
            Limit total samples loaded.

        Returns
        -------
        dict[DataSplit, InMemoryDataset]
        """
        file_path = Path(file_path)
        records: list[dict[str, Any]] = []

        with file_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in tqdm(reader, desc=f"Loading {file_path.name}"):
                if text_column not in row or label_column not in row:
                    continue
                records.append(
                    self._preprocess_record(
                        text=row[text_column], label=row[label_column]
                    )
                )
                if max_samples and len(records) >= max_samples:
                    break

        return self._split_records(records, train_ratio=train_ratio, val_ratio=val_ratio)

    def load_json(
        self,
        file_path: str | Path,
        *,
        text_key: str = "text",
        label_key: str = "label",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_samples: Optional[int] = None,
    ) -> dict[DataSplit, InMemoryDataset]:
        """
        Load a JSON / JSONL file and split into train / validation / test sets.

        Supports both a top-level list ``[{...}, ...]``
        and newline-delimited JSON (one object per line).

        Parameters
        ----------
        file_path:
            Path to the JSON or JSONL file.
        text_key:
            Key for input text in each record.
        label_key:
            Key for target label in each record.
        train_ratio:
            Fraction for training.
        val_ratio:
            Fraction for validation.
        max_samples:
            Maximum total samples to load.
        """
        file_path = Path(file_path)
        raw_records: list[dict[str, Any]] = []

        with file_path.open(encoding="utf-8") as fh:
            first_char = fh.read(1)
            fh.seek(0)
            if first_char == "[":
                # Standard JSON array
                raw_records = json.load(fh)
            else:
                # JSONL
                for line in fh:
                    line = line.strip()
                    if line:
                        raw_records.append(json.loads(line))

        records = [
            self._preprocess_record(text=r[text_key], label=r.get(label_key))
            for r in (raw_records[:max_samples] if max_samples else raw_records)
        ]
        return self._split_records(records, train_ratio=train_ratio, val_ratio=val_ratio)

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        dataset: Dataset,  # type: ignore[type-arg]
        *,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Callable[..., Any]] = None,
    ) -> DataLoader:  # type: ignore[type-arg]
        """
        Wrap a dataset in a :class:`~torch.utils.data.DataLoader`.

        Parameters
        ----------
        dataset:
            Any ``torch.utils.data.Dataset``.
        shuffle:
            Shuffle samples each epoch (use ``True`` for training).
        batch_size:
            Override the instance-level ``self.batch_size``.
        collate_fn:
            Custom collation function.  If *None* and a tokenizer is set,
            a default tokenising collator is used.
        """
        if collate_fn is None and self.tokenizer is not None:
            collate_fn = self._tokenising_collate_fn

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_record(
        self, text: str, label: Optional[Union[int, str]] = None
    ) -> dict[str, Any]:
        """Clean a single record. Override in a subclass for custom logic."""
        return {
            "text": text.strip(),
            "label": int(label) if label is not None else -1,
        }

    def _hf_to_records(
        self,
        ds: Any,
        text_column: str,
        label_column: str,
        max_samples: Optional[int],
    ) -> list[dict[str, Any]]:
        """Convert a HuggingFace Dataset to a list of dicts."""
        records: list[dict[str, Any]] = []
        for item in tqdm(ds, desc="Preprocessing", leave=False):
            records.append(
                self._preprocess_record(
                    text=item.get(text_column, ""),
                    label=item.get(label_column),
                )
            )
            if max_samples and len(records) >= max_samples:
                break
        return records

    def _split_records(
        self,
        records: list[dict[str, Any]],
        train_ratio: float,
        val_ratio: float,
    ) -> dict[DataSplit, InMemoryDataset]:
        """Split a flat list of records into train / val / test."""
        n = len(records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return {
            DataSplit.TRAIN: InMemoryDataset(records[:n_train]),
            DataSplit.VALIDATION: InMemoryDataset(records[n_train : n_train + n_val]),
            DataSplit.TEST: InMemoryDataset(records[n_train + n_val :]),
        }

    def _tokenising_collate_fn(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Default collate function: tokenise texts and stack labels."""
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        encoding: dict[str, torch.Tensor] = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding

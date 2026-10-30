"""Loading and normalization for the pinned NQ-Open dataset."""

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any


class NQOpenSourceError(ValueError):
    """Raised when an NQ-Open row does not match its documented schema."""


@dataclass(frozen=True)
class NQOpenRecord:
    """Minimal NQ-Open source record, identified within a pinned split."""

    source_id: str
    split: str
    question: str
    answers: tuple[str, ...]


def normalize_nq_open_record(
    row: Mapping[str, Any], split: str, row_index: int
) -> NQOpenRecord:
    """Validate a source row and preserve its question and answer list."""
    if not isinstance(split, str) or not split.strip():
        raise NQOpenSourceError("split must be a non-empty string")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise NQOpenSourceError("row_index must be a non-negative integer")
    if not isinstance(row, Mapping):
        raise NQOpenSourceError(f"{split}[{row_index}] must be a mapping")

    question = row.get("question")
    answers = row.get("answer")
    if not isinstance(question, str):
        raise NQOpenSourceError(f"{split}[{row_index}].question must be a string")
    if not isinstance(answers, (list, tuple)) or any(
        not isinstance(answer, str) for answer in answers
    ):
        raise NQOpenSourceError(
            f"{split}[{row_index}].answer must be a list of strings"
        )

    return NQOpenRecord(
        source_id=f"{split}:{row_index}",
        split=split,
        question=question,
        answers=tuple(answers),
    )


def load_nq_open(
    repository: str,
    revision: str,
    splits: tuple[str, ...],
) -> list[NQOpenRecord]:
    """Load selected NQ-Open splits at an immutable Hugging Face revision.

    The optional dependency is imported only when this function is called.
    Repository, revision, and splits must be supplied by the caller.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise NQOpenSourceError(
            "revision must be a 40-character lowercase commit SHA"
        )
    if not isinstance(repository, str) or not repository.strip():
        raise NQOpenSourceError("A dataset repository is required")
    if not splits or any(
        not isinstance(split, str) or not split.strip() for split in splits
    ):
        raise NQOpenSourceError("At least one non-empty dataset split is required")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading NQ-Open requires the 'datasets' package. "
            "Install project requirements before loading source data."
        ) from exc

    records: list[NQOpenRecord] = []
    for split in splits:
        dataset = load_dataset(
            repository,
            "nq_open",
            split=split,
            revision=revision,
        )
        records.extend(
            normalize_nq_open_record(row, split, row_index)
            for row_index, row in enumerate(dataset)
        )
    return records

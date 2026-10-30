"""Deterministic near-duplicate removal using token-set Jaccard."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
from collections.abc import Iterable

from .nq_open_source import NQOpenRecord
from .text_processing import tokenize_question


@dataclass(frozen=True)
class DeduplicationReport:
    """Counts from greedy source-order lexical deduplication."""

    input_count: int
    retained_count: int
    duplicates_dropped: int
    jaccard_comparisons: int
    threshold: float


def deduplicate_records(
    records: Iterable[NQOpenRecord], threshold: float
) -> tuple[list[NQOpenRecord], DeduplicationReport]:
    """Keep the first record unless it matches a previously retained record.

    Token sets are sorted by global document frequency and indexed by a
    Jaccard prefix filter, then every candidate pair is checked exactly.
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be a finite number between 0 and 1")

    input_records = list(records)
    token_sets = [set(tokenize_question(record.question)) for record in input_records]
    if any(not tokens for tokens in token_sets):
        raise ValueError(
            "deduplication requires records with at least one token; "
            "run candidate filtering first"
        )

    document_frequency = Counter(
        token for tokens in token_sets for token in tokens
    )
    token_order = {
        token: rank
        for rank, token in enumerate(sorted(
            document_frequency,
            key=lambda item: (document_frequency[item], item),
        ))
    }
    threshold_fraction = Fraction(str(threshold))

    retained_records: list[NQOpenRecord] = []
    retained_token_sets: list[set[str]] = []
    prefix_index: dict[str, list[int]] = defaultdict(list)
    comparisons = 0

    for record, tokens in zip(input_records, token_sets):
        ordered_tokens = sorted(tokens, key=token_order.__getitem__)
        prefix_length = _prefix_length(len(tokens), threshold_fraction)
        if threshold == 0:
            candidate_indices = set(range(len(retained_records)))
        else:
            candidate_indices = set()
            for token in ordered_tokens[:prefix_length]:
                candidate_indices.update(prefix_index.get(token, ()))

        is_duplicate = False
        for candidate_index in sorted(candidate_indices):
            previous_tokens = retained_token_sets[candidate_index]
            if (
                min(len(tokens), len(previous_tokens))
                / max(len(tokens), len(previous_tokens))
                < threshold
            ):
                continue
            comparisons += 1
            intersection_size = len(tokens & previous_tokens)
            union_size = len(tokens | previous_tokens)
            if intersection_size / union_size >= threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        retained_index = len(retained_records)
        retained_records.append(record)
        retained_token_sets.append(tokens)
        for token in ordered_tokens[:prefix_length]:
            prefix_index[token].append(retained_index)

    report = DeduplicationReport(
        input_count=len(input_records),
        retained_count=len(retained_records),
        duplicates_dropped=len(input_records) - len(retained_records),
        jaccard_comparisons=comparisons,
        threshold=float(threshold),
    )
    return retained_records, report


def _prefix_length(set_size: int, threshold: Fraction) -> int:
    numerator = threshold.numerator * set_size
    ceil_product = (numerator + threshold.denominator - 1) // threshold.denominator
    return min(set_size, set_size - ceil_product + 1)

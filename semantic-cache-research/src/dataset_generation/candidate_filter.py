"""Mechanical candidate eligibility checks for NQ-Open records."""

from collections import Counter
from dataclasses import dataclass
import re
import sys
import unicodedata
from collections.abc import Iterable
from typing import TextIO

from .nq_open_source import NQOpenRecord
from .text_processing import tokenize_question


REJECTION_RULES = (
    "empty_question",
    "control_characters",
    "no_alphabetic_character",
    "url_present",
    "markup_present",
    "token_count_out_of_range",
    "dominant_token",
    "empty_answer_list",
)

_URL_PATTERN = re.compile(
    r"(?i)(?:\bhttps?://|\bwww\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b"
)
_MARKUP_PATTERN = re.compile(
    r"<\s*/?\s*[A-Za-z][^>]*>"
    r"|`{1,3}"
    r"|\[[^\]]+\]\([^)]+\)"
    r"|\*\*[^*]+\*\*"
    r"|__[^_]+__"
    r"|~~[^~]+~~"
    r"|^\s{0,3}(?:#{1,6}\s|[-+*]\s|\d+\.\s)",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class CandidateFilterReport:
    """Counts for the candidate filter; rule counts may overlap."""

    input_count: int
    retained_count: int
    rejected_record_count: int
    rejection_counts_by_rule: dict[str, int]


def filter_candidates(
    records: Iterable[NQOpenRecord],
    minimum_question_tokens: int,
    maximum_question_tokens: int,
    dominant_token_max_fraction: float = 0.5,
) -> tuple[list[NQOpenRecord], CandidateFilterReport]:
    """Apply mechanical question and answer eligibility checks.

    Records violating multiple rules increment each corresponding rule count.
    `rejected_record_count` counts unique rejected records.
    """
    if (
        isinstance(minimum_question_tokens, bool)
        or not isinstance(minimum_question_tokens, int)
        or minimum_question_tokens <= 0
    ):
        raise ValueError("minimum_question_tokens must be positive")
    if (
        isinstance(maximum_question_tokens, bool)
        or not isinstance(maximum_question_tokens, int)
        or maximum_question_tokens < minimum_question_tokens
    ):
        raise ValueError(
            "maximum_question_tokens must be at least minimum_question_tokens"
        )
    if (
        isinstance(dominant_token_max_fraction, bool)
        or not isinstance(dominant_token_max_fraction, (int, float))
        or not 0 <= dominant_token_max_fraction <= 1
    ):
        raise ValueError("dominant_token_max_fraction must be between 0 and 1")

    retained: list[NQOpenRecord] = []
    rejection_counts = Counter({rule: 0 for rule in REJECTION_RULES})
    rejected_record_count = 0
    input_count = 0

    for record in records:
        input_count += 1
        violations = _question_violations(
            record.question,
            minimum_question_tokens,
            maximum_question_tokens,
            dominant_token_max_fraction,
        )
        if not any(answer.strip() for answer in record.answers):
            violations.add("empty_answer_list")

        if violations:
            rejected_record_count += 1
            rejection_counts.update(violations)
        else:
            retained.append(record)

    report = CandidateFilterReport(
        input_count=input_count,
        retained_count=len(retained),
        rejected_record_count=rejected_record_count,
        rejection_counts_by_rule=dict(rejection_counts),
    )
    return retained, report


def print_filter_report(
    report: CandidateFilterReport, file: TextIO | None = None
) -> None:
    """Print unique totals and per-rule rejection counts."""
    if file is None:
        file = sys.stdout
    print(f"Input records: {report.input_count}", file=file)
    print(f"Retained records: {report.retained_count}", file=file)
    print(f"Rejected records: {report.rejected_record_count}", file=file)
    print("Rejections by rule (counts may overlap):", file=file)
    for rule in REJECTION_RULES:
        print(f"  {rule}: {report.rejection_counts_by_rule[rule]}", file=file)


def _question_violations(
    question: str,
    minimum_question_tokens: int,
    maximum_question_tokens: int,
    dominant_token_max_fraction: float,
) -> set[str]:
    violations: set[str] = set()
    if not question.strip():
        violations.add("empty_question")
    if any(unicodedata.category(character) == "Cc" for character in question):
        violations.add("control_characters")
    if not any(character.isalpha() for character in question):
        violations.add("no_alphabetic_character")
    if _URL_PATTERN.search(question):
        violations.add("url_present")
    if _MARKUP_PATTERN.search(question):
        violations.add("markup_present")

    tokens = tokenize_question(question)
    if not minimum_question_tokens <= len(tokens) <= maximum_question_tokens:
        violations.add("token_count_out_of_range")
    if tokens and Counter(tokens).most_common(1)[0][1] / len(tokens) > (
        dominant_token_max_fraction
    ):
        violations.add("dominant_token")
    return violations

"""Generate validated P1–P4 paraphrases from NQ-Open anchors."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Any

from .ollama_generation import generate_json, require_answer_list, require_text_list
from .text_processing import tokenize_question


LEVEL_INSTRUCTIONS = {
    "P1": "Make a light edit with a synonym swap or small word-order change.",
    "P2": "Restructure the syntax while preserving the same information need.",
    "P3": "Change the register (casual or keyword-style) while preserving meaning.",
    "P4": (
        "Heavily reword the question with a different sentence structure and "
        "low lexical overlap. Do not simply add a question word to the anchor; "
        "retain only the names and numbers needed to identify the topic. "
        "For season and episode numbers, use natural ordinal wording where "
        "possible (for example, 'the sixth episode of season two')."
    ),
}
_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_NUMBER_WORD_FORMS = {
    1: {"one", "first"},
    2: {"two", "second", "ii", "wwii", "ww2"},
    3: {"three", "third", "iii"},
    4: {"four", "fourth", "iv"},
    5: {"five", "fifth", "v"},
    6: {"six", "sixth", "vi"},
    7: {"seven", "seventh", "vii"},
    8: {"eight", "eighth", "viii"},
    9: {"nine", "ninth", "ix"},
    10: {"ten", "tenth"},
    11: {"eleven", "eleventh"},
    12: {"twelve", "twelfth"},
    13: {"thirteen", "thirteenth"},
    14: {"fourteen", "fourteenth"},
    15: {"fifteen", "fifteenth"},
    16: {"sixteen", "sixteenth"},
    17: {"seventeen", "seventeenth"},
    18: {"eighteen", "eighteenth"},
    19: {"nineteen", "nineteenth"},
    20: {"twenty", "twentieth"},
}
_DECIMAL_WORD_FORMS = {
    "12.5": {
        ("twelve", "point", "five"),
        ("twelve", "and", "a", "half"),
        ("twelve", "and", "half"),
    },
}
_PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"paraphrase": {"type": "string"}},
    "required": ["paraphrase"],
    "additionalProperties": False,
}


def _preserves_numbers(anchor: str, candidate: str) -> bool:
    """Accept exact numeric strings and common equivalent English/ordinal forms.

    Decimals and values outside the small-number mapping must remain literal;
    this prevents years and quantities from being silently changed.
    """
    candidate_numbers = set(_NUMBER_PATTERN.findall(candidate))
    candidate_tokens = tokenize_question(candidate)
    candidate_token_set = set(candidate_tokens)
    anchor_tokens = tokenize_question(anchor)

    def has_sequence(sequence: tuple[str, ...]) -> bool:
        width = len(sequence)
        return any(
            candidate_tokens[index:index + width] == sequence
            for index in range(len(candidate_tokens) - width + 1)
        )

    for number in _NUMBER_PATTERN.findall(anchor):
        if number in candidate_numbers:
            continue
        if any(mark in number for mark in ".,"):
            alternatives = _DECIMAL_WORD_FORMS.get(number, set())
            if any(has_sequence(sequence) for sequence in alternatives):
                continue
            return False
        try:
            value = int(number)
        except ValueError:
            return False
        forms = _NUMBER_WORD_FORMS.get(value)
        if not forms or not candidate_token_set.intersection(forms | {
            f"{value}st", f"{value}nd", f"{value}rd", f"{value}th"
        }):
            # A decade is an equivalent form for a ten-year range; a billion
            # may be phrased with the article "a" rather than "one".
            if value == 10 and "years" in anchor_tokens and candidate_token_set.intersection({"decade", "decades"}):
                continue
            if value == 1 and has_sequence(("a", "billion")) and "billion" in anchor_tokens:
                continue
            return False
    return True


def _contains_gold_answer(candidate: str, answers: list[str]) -> bool:
    """Reject paraphrases that reveal a source answer in the query itself."""
    candidate_tokens = tokenize_question(candidate)
    for answer in answers:
        answer_tokens = tokenize_question(answer)
        if not answer_tokens or len(answer_tokens) > len(candidate_tokens):
            continue
        width = len(answer_tokens)
        if any(
            candidate_tokens[index:index + width] == answer_tokens
            for index in range(len(candidate_tokens) - width + 1)
        ):
            return True
    return False


def _is_multiple_result_span(anchor: str, answers: list[str]) -> bool:
    """Detect questions asking for winners/results across a time or event set."""
    if len(answers) < 2:
        return False
    return bool(re.search(
        r"\b(?:last|past|previous|most recent)\b.{0,50}\b(?:years?|seasons?|series|"
        r"championships?|tournaments?|competitions?)\b",
        anchor,
        flags=re.IGNORECASE,
    ))


def _collapses_multiple_result_scope(
    candidate: str, anchor: str, answers: list[str]
) -> bool:
    if not _is_multiple_result_span(anchor, answers):
        return False
    return bool(re.search(
        r"\bwhich\s+(?:team|club|franchise|side)\b|\bmost\s+(?:titles?|"
        r"championships?|wins?)\b",
        candidate,
        flags=re.IGNORECASE,
    ))


def generate_paraphrase_records(
    seed_question: str,
    answers: list[str] | tuple[str, ...],
    style_examples: list[str] | tuple[str, ...],
    *,
    model: str = "gemma3:4b",
    temperature: float = 0.0,
    minimum_length_ratio: float = 0.5,
    maximum_length_ratio: float = 2.0,
    preserve_numbers_and_years: bool = True,
    maximum_retries: int = 3,
    levels: tuple[str, ...] | list[str] = ("P1", "P2", "P3", "P4"),
) -> "ParaphraseGenerationResult":
    """Generate one valid paraphrase for every requested level.

    Each record contains the paraphrase, inherited source answers, model,
    level, prompt hash, and retry count. Invalid candidates are retried up to
    ``maximum_retries`` times after the initial attempt; exhausted levels are
    omitted and reported in ``generation_failures`` on the returned list.
    """
    if not isinstance(seed_question, str) or not seed_question.strip():
        raise ValueError("seed_question must be a non-empty string")
    answers_list = require_answer_list(answers)
    examples = require_text_list(style_examples, "style_examples")
    if not 5 <= len(examples) <= 8:
        raise ValueError("style_examples must contain 5 to 8 real questions")
    if any(level not in LEVEL_INSTRUCTIONS for level in levels):
        raise ValueError("levels must be selected from P1, P2, P3, and P4")
    if minimum_length_ratio <= 0 or maximum_length_ratio < minimum_length_ratio:
        raise ValueError("invalid paraphrase length-ratio bounds")
    if (
        isinstance(maximum_retries, bool)
        or not isinstance(maximum_retries, int)
        or maximum_retries < 0
    ):
        raise ValueError("maximum_retries must be a non-negative integer")

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    failure_reasons: dict[str, str] = {}
    for level in levels:
        accepted = None
        last_failure = "candidate failed validation"
        for attempt in range(maximum_retries + 1):
            retry_hint = ""
            if attempt:
                retry_hint = f"Previous candidate failed validation: {last_failure}. Produce a fresh candidate that fixes this issue."
            prompt = _build_prompt(
                seed_question,
                answers_list,
                examples,
                level,
                retry_hint=retry_hint,
                minimum_length_ratio=minimum_length_ratio,
                maximum_length_ratio=maximum_length_ratio,
            )
            payload, prompt_hash = generate_json(
                model, prompt, temperature, response_schema=_PARAPHRASE_SCHEMA
            )
            candidate = payload.get("paraphrase")
            if not isinstance(candidate, str):
                last_failure = "model response omitted the paraphrase string"
                continue
            candidate = candidate.strip().strip('"')
            ratio = len(tokenize_question(candidate)) / max(
                1, len(tokenize_question(seed_question))
            )
            if not tokenize_question(candidate):
                last_failure = "candidate is empty or has no lexical tokens"
                continue
            if candidate.casefold() == seed_question.casefold():
                last_failure = "candidate is identical to the anchor"
                continue
            if not minimum_length_ratio <= ratio <= maximum_length_ratio:
                last_failure = (
                    f"candidate token-length ratio is {ratio:.3f}; required range is "
                    f"[{minimum_length_ratio:.3f}, {maximum_length_ratio:.3f}]"
                )
                continue
            if preserve_numbers_and_years:
                if not _preserves_numbers(seed_question, candidate):
                    last_failure = "candidate omits or changes a number/year in the anchor"
                    continue
            if _contains_gold_answer(candidate, answers_list):
                last_failure = "candidate reveals a gold answer in the query"
                continue
            if _collapses_multiple_result_scope(candidate, seed_question, answers_list):
                last_failure = (
                    "candidate narrows a multiple-winner time span to one team/club "
                    "or the most frequent winner"
                )
                continue
            accepted = {
                "level": level,
                "text": candidate,
                "gold_answers": list(answers_list),
                "generator_model": model,
                "prompt_hash": prompt_hash,
                "retry_count": attempt,
            }
            break
        if accepted is None:
            failures.append(level)
            failure_reasons[level] = last_failure
        else:
            records.append(accepted)

    return ParaphraseGenerationResult(
        records=records,
        generation_failures=tuple(failures),
        failure_reasons=failure_reasons,
    )


@dataclass(frozen=True)
class ParaphraseGenerationResult:
    records: list[dict[str, Any]]
    generation_failures: tuple[str, ...]
    failure_reasons: dict[str, str]


def generate_paraphrases(seed_question: str, n: int = 5) -> list[str]:
    """Legacy wrapper retained for old callers; use the record API for corpus work."""
    if n < 1:
        return []
    raise ValueError(
        "generate_paraphrases now requires source answers and 5–8 style examples; "
        "use generate_paraphrase_records"
    )


def _build_prompt(
    anchor: str,
    answers: list[str],
    examples: list[str],
    level: str,
    retry_hint: str = "",
    minimum_length_ratio: float = 0.5,
    maximum_length_ratio: float = 2.0,
) -> str:
    anchor_token_count = max(1, len(tokenize_question(anchor)))
    minimum_tokens = math.ceil(minimum_length_ratio * anchor_token_count)
    maximum_tokens = math.floor(maximum_length_ratio * anchor_token_count)
    length_requirement = (
        f"Use between {minimum_tokens} and {maximum_tokens} word tokens; "
        "count tokens as words separated from punctuation.\n"
    )
    numeric_values = _NUMBER_PATTERN.findall(anchor)
    numeric_requirement = (
        "The anchor contains these numeric values: "
        f"{', '.join(numeric_values)}. Preserve every one of them; including "
        "only some of them is not sufficient.\n"
        if numeric_values else ""
    )
    answer_scope_hint = (
        "The answer list has multiple entries and this anchor asks about results "
        "across a time span. Ask for the full set of winners/results using a "
        "plural formulation (for example, which teams or clubs); do not ask "
        "which single team/club won the most or won every event.\n"
        if _is_multiple_result_span(anchor, answers) else ""
    )
    return (
        "Generate one natural question paraphrase as a JSON object with the "
        'single key "paraphrase". Return no other keys or prose.\n'
        f"Target level {level}: {LEVEL_INSTRUCTIONS[level]}\n"
        f"{retry_hint}\n"
        "Use natural, grammatical, idiomatic English. Avoid personifying dates "
        "or times. Preserve the information need and answerability. Match the style, "
        "casing, punctuation, and register of the examples; P3 is the only "
        "level that intentionally changes register. Do not answer the question. "
        "Preserve every numeric value and year accurately. Equivalent number "
        "forms are allowed only when they state the same value (for example, "
        "7 and seventh); do not change or omit a number. Do not include any "
        "gold answer from the answer list in the rewritten question. Preserve "
        "the original question's information need and answer type; do not "
        "change a who/what/when/where/which question into a yes/no question "
        "or a statement.\n"
        f"{numeric_requirement}"
        f"{answer_scope_hint}"
        f"{length_requirement}"
        f"Style examples: {examples!r}\n"
        f"Anchor question: {anchor!r}\n"
        f"Source answer list (reference only; do not rewrite): {answers!r}\n"
    )

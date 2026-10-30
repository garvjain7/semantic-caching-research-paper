"""Generate answer-changing E/N/T/S queries for corpus auditing."""

from __future__ import annotations

from typing import Any

from .ollama_generation import generate_json, require_answer_list, require_text_list


TYPE_INSTRUCTIONS = {
    "E": "Substitute the target entity and change no other element.",
    "N": "Flip the question's negation or polarity and change no other element.",
    "T": "Change the target number, date, or time and change no other element.",
    "S": "Narrow or broaden the scope and change no other element.",
}


def generate_hard_negative(
    seed_question: str,
    seed_answer: str | list[str] | tuple[str, ...],
    *,
    negative_type: str = "E",
    style_examples: list[str] | tuple[str, ...],
    model: str = "gemma3:4b",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Generate one typed negative or record that the edit is inapplicable.

    The returned answer-difference statement is an unverified audit hint. It
    is deliberately not written as ``gold_answers``.
    """
    if not isinstance(seed_question, str) or not seed_question.strip():
        raise ValueError("seed_question must be a non-empty string")
    answers = [seed_answer] if isinstance(seed_answer, str) else list(seed_answer)
    answers = require_answer_list(answers)
    examples = require_text_list(style_examples, "style_examples")
    if not 5 <= len(examples) <= 8:
        raise ValueError("style_examples must contain 5 to 8 real questions")
    if negative_type not in TYPE_INSTRUCTIONS:
        raise ValueError("negative_type must be one of E, N, T, or S")

    prompt = _build_prompt(seed_question, answers, examples, negative_type)
    payload, prompt_hash = generate_json(model, prompt, temperature)
    status = payload.get("status")
    if status == "NOT_APPLICABLE":
        return {
            "subtype": negative_type,
            "status": "NOT_APPLICABLE",
            "text": None,
            "answer_difference_statement": None,
            "gold_answers": None,
            "generator_model": model,
            "prompt_hash": prompt_hash,
        }
    if status != "OK":
        raise ValueError("hard-negative response status must be OK or NOT_APPLICABLE")
    question = payload.get("question")
    difference = payload.get("answer_difference_statement")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("hard-negative response must include a non-empty question")
    question = question.strip().strip('"')
    if question.casefold() == seed_question.casefold():
        raise ValueError("hard-negative question must differ from its anchor")
    if not isinstance(difference, str) or not difference.strip():
        raise ValueError("hard-negative response must include an answer-difference statement")
    return {
        "subtype": negative_type,
        "status": "OK",
        "text": question,
        "answer_difference_statement": difference.strip(),
        "gold_answers": None,
        "generator_model": model,
        "prompt_hash": prompt_hash,
    }


def _build_prompt(
    anchor: str, answers: list[str], examples: list[str], negative_type: str
) -> str:
    return (
        "Generate one natural hard-negative question as a JSON object with "
        'keys "status", "question", and "answer_difference_statement". '
        'Use status "NOT_APPLICABLE" with null question and statement when '
        "the requested edit cannot be made responsibly. Otherwise use status "
        '"OK". Return no other keys or prose.\n'
        f"Edit type {negative_type}: {TYPE_INSTRUCTIONS[negative_type]}\n"
        "Keep the sentence frame and style of the examples. The new question "
        "must read naturally and require a different correct answer. The "
        "answer-difference statement is an unverified audit hint, not a gold answer.\n"
        f"Style examples: {examples!r}\n"
        f"Anchor question: {anchor!r}\n"
        f"Source answer list (reference only): {answers!r}\n"
    )

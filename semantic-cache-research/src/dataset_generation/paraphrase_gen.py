"""
paraphrase_gen.py

Generates same-answer paraphrases of seed questions from the base QA
dataset (data/raw/). These form the "paraphrase group" — queries that
SHOULD trigger a true cache hit.

IMPORTANT: after generation, actual embedding similarity must be
measured and bucketed empirically — do not trust the generation
prompt's stated intent as the real similarity (see PROJECT_STATE.md,
Assumptions section, and data/processed/verification_report.md).

Contains:
    - generate_paraphrases(seed_question, n) -> list[str]
"""


def generate_paraphrases(seed_question: str, n: int = 5) -> list[str]:
    """Call an LLM with a paraphrase prompt, return n rewordings."""
    # TODO: implement
    raise NotImplementedError

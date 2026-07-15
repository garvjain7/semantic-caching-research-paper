"""
hard_negative_gen.py

Generates hard-negative pairs: queries that are lexically/topically
close to a seed question but have a DIFFERENT correct answer.

Built via template/entity-swap methods, NOT free-form LLM generation,
so the ground-truth answer change is guaranteed rather than assumed
(see PROJECT_STATE.md Decisions Log, 2026-07-15).

Contains:
    - generate_hard_negative(seed_question, seed_answer) ->
      (new_question, new_answer)
"""


def generate_hard_negative(seed_question: str, seed_answer: str):
    """
    Identify a swappable entity in seed_question (e.g. a country, a
    date, a number) and substitute it with a different valid entity,
    producing a question with a known, different correct answer.
    """
    # TODO: implement
    raise NotImplementedError

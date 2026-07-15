"""
unrelated_sampler.py

Samples unrelated questions from the base dataset as the control
group — expected to have low similarity to any given seed question.

Contains:
    - sample_unrelated(seed_question, dataset, n) -> list[str]
"""


def sample_unrelated(seed_question: str, dataset, n: int = 5) -> list:
    """Randomly sample n questions, excluding anything topically close
    to seed_question."""
    # TODO: implement
    raise NotImplementedError

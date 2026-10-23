from itertools import combinations
import unittest

from src.dataset_generation.lexical_dedup import deduplicate_records
from src.dataset_generation.nq_open_source import NQOpenRecord
from src.dataset_generation.text_processing import tokenize_question


class LexicalDedupTests(unittest.TestCase):
    def test_keeps_first_source_occurrence_at_or_above_threshold(self):
        first = record("alpha beta gamma delta epsilon", "train:0")
        duplicate = record(
            "alpha beta gamma delta epsilon zeta", "validation:0", "validation"
        )

        retained, report = deduplicate_records([first, duplicate], 0.8)

        self.assertEqual(retained, [first])
        self.assertEqual(report.duplicates_dropped, 1)

    def test_keeps_records_below_threshold(self):
        first = record("alpha beta gamma delta epsilon", "train:0")
        second = record("alpha beta gamma theta iota", "train:1")

        retained, report = deduplicate_records([first, second], 0.8)

        self.assertEqual(retained, [first, second])
        self.assertEqual(report.duplicates_dropped, 0)

    def test_compares_only_with_previously_retained_records(self):
        first = record("alpha beta gamma delta epsilon", "train:0")
        duplicate_of_first = record(
            "alpha beta gamma delta epsilon zeta", "train:1"
        )
        similar_only_to_dropped = record(
            "alpha beta gamma delta zeta", "train:2"
        )

        retained, _ = deduplicate_records(
            [first, duplicate_of_first, similar_only_to_dropped], 0.8
        )

        self.assertEqual(retained, [first, similar_only_to_dropped])

    def test_zero_threshold_treats_disjoint_questions_as_duplicates(self):
        records = [
            record("alpha beta", "train:0"),
            record("gamma delta", "train:1"),
        ]

        retained, report = deduplicate_records(records, 0)

        self.assertEqual(retained, records[:1])
        self.assertEqual(report.duplicates_dropped, 1)

    def test_requires_nonempty_tokenized_questions(self):
        with self.assertRaisesRegex(ValueError, "run candidate filtering first"):
            deduplicate_records([record("!!!", "train:0")], 0.8)

    def test_prefix_index_matches_brute_force_greedy_results(self):
        vocabulary = ("a", "b", "c", "d", "e", "f")
        token_sets = [
            set(combo)
            for size in range(1, len(vocabulary) + 1)
            for combo in combinations(vocabulary, size)
        ]
        records = [
            record(" ".join(sorted(tokens)), f"train:{index}")
            for index, tokens in enumerate(token_sets)
        ]

        for threshold in (0.0, 0.2, 0.5, 0.8, 1.0):
            with self.subTest(threshold=threshold):
                actual, _ = deduplicate_records(records, threshold)
                expected = brute_force_greedy(records, threshold)
                self.assertEqual(
                    [item.source_id for item in actual],
                    [item.source_id for item in expected],
                )


def brute_force_greedy(
    records: list[NQOpenRecord], threshold: float
) -> list[NQOpenRecord]:
    retained = []
    retained_tokens = []
    for item in records:
        tokens = set(tokenize_question(item.question))
        duplicate = any(
            len(tokens & previous) / len(tokens | previous) >= threshold
            for previous in retained_tokens
        )
        if not duplicate:
            retained.append(item)
            retained_tokens.append(tokens)
    return retained


def record(
    question: str, source_id: str, split: str = "train"
) -> NQOpenRecord:
    return NQOpenRecord(
        source_id=source_id,
        split=split,
        question=question,
        answers=("answer",),
    )


if __name__ == "__main__":
    unittest.main()

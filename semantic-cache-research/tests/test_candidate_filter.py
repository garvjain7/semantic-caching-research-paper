from contextlib import redirect_stdout
from io import StringIO
import unittest

from src.dataset_generation.candidate_filter import (
    filter_candidates,
    print_filter_report,
)
from src.dataset_generation.nq_open_source import NQOpenRecord
from src.dataset_generation.text_processing import tokenize_question


class CandidateFilterTests(unittest.TestCase):
    def test_tokenizer_lowercases_unicode_words_and_excludes_punctuation(self):
        self.assertEqual(
            tokenize_question("¿Qué es la capital de Perú, en 2024?"),
            ("qué", "es", "la", "capital", "de", "perú", "en", "2024"),
        )

    def test_accepts_valid_question_without_capitalization_or_question_mark(self):
        candidate = record("capital city of france is")

        retained, report = filter_candidates([candidate], 5, 20)

        self.assertEqual(retained, [candidate])
        self.assertEqual(report.rejected_record_count, 0)

    def test_counts_each_rejection_rule(self):
        records = [
            record("   "),
            record("12345"),
            record("Who wrote Hamlet?"),
            record("What is the capital https://example.org"),
            record("What is **bold** text here today?"),
            record("Why why why is this"),
            record("What is the capital of France?", answers=()),
        ]

        retained, report = filter_candidates(records, 5, 20)

        self.assertEqual(retained, [])
        self.assertEqual(report.input_count, 7)
        self.assertEqual(report.rejected_record_count, 7)
        self.assertEqual(
            report.rejection_counts_by_rule,
            {
                "empty_question": 1,
                "control_characters": 0,
                "no_alphabetic_character": 2,
                "url_present": 1,
                "markup_present": 1,
                "token_count_out_of_range": 3,
                "dominant_token": 2,
                "empty_answer_list": 1,
            },
        )

    def test_counts_control_characters_even_when_question_is_otherwise_valid(self):
        retained, report = filter_candidates(
            [record("What is the capital\x00 of France?")], 5, 20
        )

        self.assertEqual(retained, [])
        self.assertEqual(report.rejection_counts_by_rule["control_characters"], 1)

    def test_prints_unique_and_per_rule_rejection_counts(self):
        _, report = filter_candidates([record("Who wrote Hamlet?")], 5, 20)
        output = StringIO()

        with redirect_stdout(output):
            print_filter_report(report)

        self.assertIn("Rejected records: 1", output.getvalue())
        self.assertIn("token_count_out_of_range: 1", output.getvalue())


def record(question: str, answers: tuple[str, ...] = ("answer",)) -> NQOpenRecord:
    return NQOpenRecord(
        source_id="train:0",
        split="train",
        question=question,
        answers=answers,
    )


if __name__ == "__main__":
    unittest.main()

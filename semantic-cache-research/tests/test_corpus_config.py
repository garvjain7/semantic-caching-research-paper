import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.corpus_config import (
    CorpusConfigError,
    StageReadinessError,
    load_corpus_config,
    missing_stage_requirements,
    require_stage_ready,
)
from src.dataset_generation.nq_open_source import (
    NQOpenSourceError,
    load_nq_open,
    normalize_nq_open_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus.yaml"


class CorpusConfigTests(unittest.TestCase):
    def test_loads_repository_config_with_explicit_unresolved_values(self):
        config = load_corpus_config(CONFIG_PATH)

        self.assertEqual(config["corpus"]["sampling"]["anchors"], 600)
        self.assertEqual(config["corpus"]["source"]["dataset"], "nq_open")
        self.assertEqual(config["corpus"]["source"]["answers_field"], "answer")
        self.assertIsNone(config["corpus"]["seed"])
        self.assertIsNone(config["corpus"]["models"]["generator"]["id"])

    def test_rejects_invalid_probability(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        config["corpus"]["filtering"][
            "deduplication_token_jaccard_threshold"
        ] = 1.1

        with self.assertRaisesRegex(
            CorpusConfigError, "deduplication_token_jaccard_threshold"
        ):
            self._load_mapping(config)

    def test_rejects_unrecognized_keys(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        config["corpus"]["sampling"]["anchor"] = 600

        with self.assertRaisesRegex(CorpusConfigError, "unknown keys"):
            self._load_mapping(config)

    def test_rejects_non_commit_source_revision(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        config["corpus"]["source"]["revision"] = "main"

        with self.assertRaisesRegex(CorpusConfigError, "40-character lowercase"):
            self._load_mapping(config)

    def test_requires_model_id_and_version_together(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        config["corpus"]["models"]["generator"]["id"] = "some-model"

        with self.assertRaisesRegex(
            CorpusConfigError, "both be set or both be null"
        ):
            self._load_mapping(config)

    def test_reports_missing_requirements_for_sampling(self):
        config = load_corpus_config(CONFIG_PATH)

        self.assertEqual(
            missing_stage_requirements(config, "sampling"),
            ["corpus.seed"],
        )

    def test_blocks_generation_until_model_and_seed_are_set(self):
        config = load_corpus_config(CONFIG_PATH)

        with self.assertRaisesRegex(
            StageReadinessError, "corpus.models.generator.id"
        ):
            require_stage_ready(config, "paraphrase_generation")

    def test_allows_sampling_when_prerequisites_are_explicit(self):
        config = copy.deepcopy(load_corpus_config(CONFIG_PATH))
        config["corpus"]["seed"] = 1234

        require_stage_ready(config, "sampling")

    def test_rejects_unknown_stage(self):
        config = load_corpus_config(CONFIG_PATH)

        with self.assertRaisesRegex(CorpusConfigError, "Unknown corpus stage"):
            missing_stage_requirements(config, "all")

    def test_normalizes_nq_open_row_without_collapsing_answer_list(self):
        record = normalize_nq_open_record(
            {
                "question": "Who wrote Hamlet?",
                "answer": ["William Shakespeare", "Shakespeare"],
                "unused_field": "not copied",
            },
            "train",
            7,
        )

        self.assertEqual(record.source_id, "train:7")
        self.assertEqual(
            record.answers, ("William Shakespeare", "Shakespeare")
        )
        self.assertFalse(hasattr(record, "unused_field"))

    def test_rejects_malformed_answer_field(self):
        with self.assertRaisesRegex(NQOpenSourceError, "list of strings"):
            normalize_nq_open_record(
                {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
                "train",
                0,
            )

    def test_loads_each_requested_split_at_pinned_revision(self):
        revision = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
        calls = []

        def fake_load_dataset(repository, config, split, revision):
            calls.append((repository, config, split, revision))
            return [{"question": f"Question from {split}", "answer": ["answer"]}]

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            records = load_nq_open(
                "google-research-datasets/nq_open",
                revision,
                ("train", "validation"),
            )

        self.assertEqual(
            calls,
            [
                (
                    "google-research-datasets/nq_open",
                    "nq_open",
                    "train",
                    revision,
                ),
                (
                    "google-research-datasets/nq_open",
                    "nq_open",
                    "validation",
                    revision,
                ),
            ],
        )
        self.assertEqual(
            [record.source_id for record in records],
            ["train:0", "validation:0"],
        )

    @staticmethod
    def _load_mapping(config):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "corpus.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            return load_corpus_config(config_path)


if __name__ == "__main__":
    unittest.main()

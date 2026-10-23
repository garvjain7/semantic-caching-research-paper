"""Load and validate corpus-construction configuration."""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml


SUPPORTED_STAGES = (
    "sampling",
    "cache_responses",
    "paraphrase_generation",
    "hard_negative_generation",
    "judge_evaluation",
)


class CorpusConfigError(ValueError):
    """Raised when a corpus configuration is malformed or inconsistent."""


class StageReadinessError(CorpusConfigError):
    """Raised when a stage is requested before its config is ready."""


def load_corpus_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML corpus config and validate its declared settings.

    Explicitly unresolved values may remain null. Operational readiness checks
    belong to the pipeline stage that needs those values.
    """
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise CorpusConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    _expect_mapping(raw_config, "root")
    if set(raw_config) != {"corpus"}:
        raise CorpusConfigError("root must contain only the 'corpus' key")

    corpus = raw_config["corpus"]
    _expect_mapping(corpus, "corpus")
    _require_keys(
        corpus,
        {
            "source",
            "seed",
            "sampling",
            "filtering",
            "paraphrases",
            "hard_negatives",
            "cache_responses",
            "audit",
            "models",
        },
        "corpus",
    )
    _validate_source(_required_mapping(corpus, "source", "corpus"))
    _validate_seed(corpus)
    _validate_sampling(_required_mapping(corpus, "sampling", "corpus"))
    _validate_filtering(_required_mapping(corpus, "filtering", "corpus"))
    _validate_paraphrases(_required_mapping(corpus, "paraphrases", "corpus"))
    _validate_hard_negatives(
        _required_mapping(corpus, "hard_negatives", "corpus")
    )
    _validate_cache_responses(
        _required_mapping(corpus, "cache_responses", "corpus")
    )
    _validate_audit(_required_mapping(corpus, "audit", "corpus"))
    _validate_models(_required_mapping(corpus, "models", "corpus"))
    return raw_config


def missing_stage_requirements(
    config: Mapping[str, Any], stage: str
) -> list[str]:
    """Return configured prerequisites that are missing for a pipeline stage.

    This checks configuration only. It does not verify that source files,
    model installations, or external services are available.
    """
    if stage not in SUPPORTED_STAGES:
        supported = ", ".join(SUPPORTED_STAGES)
        raise CorpusConfigError(
            f"Unknown corpus stage {stage!r}; supported stages: {supported}"
        )

    corpus = config["corpus"]
    requirements: dict[str, tuple[tuple[str, Any], ...]] = {
        "sampling": (
            ("corpus.seed", corpus["seed"]),
            ("corpus.source.revision", corpus["source"]["revision"]),
        ),
        "cache_responses": (
            ("corpus.models.fallback.id", corpus["models"]["fallback"]["id"]),
            (
                "corpus.models.fallback.version",
                corpus["models"]["fallback"]["version"],
            ),
        ),
        "paraphrase_generation": (
            ("corpus.seed", corpus["seed"]),
            ("corpus.models.generator.id", corpus["models"]["generator"]["id"]),
            (
                "corpus.models.generator.version",
                corpus["models"]["generator"]["version"],
            ),
        ),
        "hard_negative_generation": (
            ("corpus.seed", corpus["seed"]),
            ("corpus.models.generator.id", corpus["models"]["generator"]["id"]),
            (
                "corpus.models.generator.version",
                corpus["models"]["generator"]["version"],
            ),
        ),
        "judge_evaluation": (
            ("corpus.models.judge.id", corpus["models"]["judge"]["id"]),
            (
                "corpus.models.judge.version",
                corpus["models"]["judge"]["version"],
            ),
        ),
    }
    missing = [
        path
        for path, value in requirements[stage]
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if (
        stage == "judge_evaluation"
        and not corpus["models"]["judge"]["must_differ_from_generator_family"]
    ):
        missing.append(
            "corpus.models.judge.must_differ_from_generator_family must be true"
        )
    return missing


def require_stage_ready(config: Mapping[str, Any], stage: str) -> None:
    """Raise with actionable missing config paths if a stage is not ready."""
    missing = missing_stage_requirements(config, stage)
    if missing:
        requirements = "\n".join(f"- {item}" for item in missing)
        raise StageReadinessError(
            f"Corpus stage {stage!r} is not ready. Set:\n{requirements}"
        )


def _expect_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise CorpusConfigError(f"{name} must be a mapping")


def _required_mapping(
    parent: Mapping[str, Any], key: str, parent_name: str
) -> Mapping[str, Any]:
    if key not in parent:
        raise CorpusConfigError(f"{parent_name}.{key} is required")
    value = parent[key]
    _expect_mapping(value, f"{parent_name}.{key}")
    return value


def _require_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise CorpusConfigError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise CorpusConfigError(f"{name} has unknown keys: {sorted(unknown)}")


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusConfigError(f"{name} must be a positive integer")


def _nonnegative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusConfigError(f"{name} must be a non-negative integer")


def _probability(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise CorpusConfigError(f"{name} must be between 0 and 1 inclusive")


def _positive_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise CorpusConfigError(f"{name} must be a positive number")


def _nullable_nonempty_string(value: Any, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise CorpusConfigError(f"{name} must be a non-empty string or null")


def _validate_source(source: Mapping[str, Any]) -> None:
    _require_keys(
        source,
        {
            "dataset",
            "repository",
            "revision",
            "splits",
            "question_field",
            "answers_field",
            "raw_path",
        },
        "corpus.source",
    )
    if source["dataset"] != "nq_open":
        raise CorpusConfigError("corpus.source.dataset must be 'nq_open'")
    if source["repository"] != "google-research-datasets/nq_open":
        raise CorpusConfigError(
            "corpus.source.repository must be 'google-research-datasets/nq_open'"
        )
    revision = source["revision"]
    _nullable_nonempty_string(revision, "corpus.source.revision")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CorpusConfigError(
            "corpus.source.revision must be a 40-character lowercase commit SHA or null"
        )
    if source["splits"] != ["train", "validation"]:
        raise CorpusConfigError(
            "corpus.source.splits must be [train, validation]"
        )
    if source["question_field"] != "question":
        raise CorpusConfigError("corpus.source.question_field must be 'question'")
    if source["answers_field"] != "answer":
        raise CorpusConfigError("corpus.source.answers_field must be 'answer'")
    if not isinstance(source["raw_path"], str) or not source["raw_path"].strip():
        raise CorpusConfigError("corpus.source.raw_path must be a non-empty string")


def _validate_seed(corpus: Mapping[str, Any]) -> None:
    if "seed" not in corpus:
        raise CorpusConfigError("corpus.seed is required")
    seed = corpus["seed"]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise CorpusConfigError("corpus.seed must be an integer or null")


def _validate_sampling(sampling: Mapping[str, Any]) -> None:
    _require_keys(
        sampling,
        {
            "anchors",
            "dev_anchors",
            "held_out_questions_minimum",
            "unrelated_queries",
        },
        "corpus.sampling",
    )
    for key, value in sampling.items():
        _positive_integer(value, f"corpus.sampling.{key}")
    if sampling["dev_anchors"] >= sampling["anchors"]:
        raise CorpusConfigError(
            "corpus.sampling.dev_anchors must be less than anchors"
        )


def _validate_filtering(filtering: Mapping[str, Any]) -> None:
    _require_keys(
        filtering,
        {
            "minimum_question_tokens",
            "maximum_question_tokens",
            "tokenizer",
            "deduplication_token_jaccard_threshold",
            "unrelated_maximum_token_jaccard_to_any_anchor",
            "dominant_token_max_fraction",
        },
        "corpus.filtering",
    )
    _positive_integer(
        filtering["minimum_question_tokens"],
        "corpus.filtering.minimum_question_tokens",
    )
    _positive_integer(
        filtering["maximum_question_tokens"],
        "corpus.filtering.maximum_question_tokens",
    )
    if filtering["tokenizer"] != "unicode_word_tokens_lowercase_v1":
        raise CorpusConfigError(
            "corpus.filtering.tokenizer must be "
            "'unicode_word_tokens_lowercase_v1'"
        )
    if filtering["minimum_question_tokens"] > filtering["maximum_question_tokens"]:
        raise CorpusConfigError(
            "minimum_question_tokens must not exceed maximum_question_tokens"
        )
    _probability(
        filtering["deduplication_token_jaccard_threshold"],
        "corpus.filtering.deduplication_token_jaccard_threshold",
    )
    _probability(
        filtering["unrelated_maximum_token_jaccard_to_any_anchor"],
        "corpus.filtering.unrelated_maximum_token_jaccard_to_any_anchor",
    )
    _probability(
        filtering["dominant_token_max_fraction"],
        "corpus.filtering.dominant_token_max_fraction",
    )


def _validate_paraphrases(paraphrases: Mapping[str, Any]) -> None:
    _require_keys(
        paraphrases,
        {
            "per_anchor",
            "levels",
            "style_examples_minimum",
            "style_examples_maximum",
            "minimum_length_ratio",
            "maximum_length_ratio",
            "preserve_anchor_numbers_and_years",
            "maximum_retries",
        },
        "corpus.paraphrases",
    )
    _positive_integer(paraphrases["per_anchor"], "corpus.paraphrases.per_anchor")
    if paraphrases["levels"] != ["P1", "P2", "P3", "P4"]:
        raise CorpusConfigError(
            "corpus.paraphrases.levels must be [P1, P2, P3, P4]"
        )
    for key in ("style_examples_minimum", "style_examples_maximum"):
        _positive_integer(paraphrases[key], f"corpus.paraphrases.{key}")
    if paraphrases["style_examples_minimum"] > paraphrases["style_examples_maximum"]:
        raise CorpusConfigError(
            "style_examples_minimum must not exceed style_examples_maximum"
        )
    _positive_number(
        paraphrases["minimum_length_ratio"],
        "corpus.paraphrases.minimum_length_ratio",
    )
    _positive_number(
        paraphrases["maximum_length_ratio"],
        "corpus.paraphrases.maximum_length_ratio",
    )
    if paraphrases["minimum_length_ratio"] > paraphrases["maximum_length_ratio"]:
        raise CorpusConfigError(
            "minimum_length_ratio must not exceed maximum_length_ratio"
        )
    if not isinstance(paraphrases["preserve_anchor_numbers_and_years"], bool):
        raise CorpusConfigError(
            "corpus.paraphrases.preserve_anchor_numbers_and_years must be boolean"
        )
    _nonnegative_integer(
        paraphrases["maximum_retries"], "corpus.paraphrases.maximum_retries"
    )


def _validate_hard_negatives(hard_negatives: Mapping[str, Any]) -> None:
    _require_keys(
        hard_negatives,
        {"types", "quota_per_type"},
        "corpus.hard_negatives",
    )
    if hard_negatives["types"] != ["E", "N", "T", "S"]:
        raise CorpusConfigError(
            "corpus.hard_negatives.types must be [E, N, T, S]"
        )
    _positive_integer(
        hard_negatives["quota_per_type"],
        "corpus.hard_negatives.quota_per_type",
    )


def _validate_cache_responses(cache_responses: Mapping[str, Any]) -> None:
    _require_keys(
        cache_responses,
        {"provider", "temperature"},
        "corpus.cache_responses",
    )
    if cache_responses["provider"] != "ollama":
        raise CorpusConfigError("corpus.cache_responses.provider must be 'ollama'")
    if (
        isinstance(cache_responses["temperature"], bool)
        or not isinstance(cache_responses["temperature"], (int, float))
        or cache_responses["temperature"] < 0
    ):
        raise CorpusConfigError(
            "corpus.cache_responses.temperature must be a non-negative number"
        )


def _validate_audit(audit: Mapping[str, Any]) -> None:
    _require_keys(
        audit,
        {
            "samples_per_stratum",
            "strata",
            "regeneration_accuracy_threshold",
            "confidence_interval",
        },
        "corpus.audit",
    )
    _positive_integer(
        audit["samples_per_stratum"], "corpus.audit.samples_per_stratum"
    )
    expected_strata = ["P1", "P2", "P3", "P4", "E", "N", "T", "S", "unrelated"]
    if audit["strata"] != expected_strata:
        raise CorpusConfigError(
            "corpus.audit.strata must be "
            "[P1, P2, P3, P4, E, N, T, S, unrelated]"
        )
    _probability(
        audit["regeneration_accuracy_threshold"],
        "corpus.audit.regeneration_accuracy_threshold",
    )
    if audit["confidence_interval"] != "wilson_95":
        raise CorpusConfigError(
            "corpus.audit.confidence_interval must be 'wilson_95'"
        )


def _validate_models(models: Mapping[str, Any]) -> None:
    expected_roles = {"generator", "fallback", "judge"}
    _require_keys(models, expected_roles, "corpus.models")
    for role in expected_roles:
        model = _required_mapping(models, role, "corpus.models")
        expected_model_keys = {"id", "version"}
        if role == "judge":
            expected_model_keys.add("must_differ_from_generator_family")
        _require_keys(model, expected_model_keys, f"corpus.models.{role}")
        _nullable_nonempty_string(model["id"], f"corpus.models.{role}.id")
        _nullable_nonempty_string(
            model["version"], f"corpus.models.{role}.version"
        )
        if (model["id"] is None) != (model["version"] is None):
            raise CorpusConfigError(
                f"corpus.models.{role}.id and version must both be set or both be null"
            )
        if role == "judge":
            if not isinstance(
                model["must_differ_from_generator_family"], bool
            ):
                raise CorpusConfigError(
                    "corpus.models.judge.must_differ_from_generator_family "
                    "must be boolean"
                )

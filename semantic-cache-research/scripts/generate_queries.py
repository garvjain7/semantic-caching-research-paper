"""Generate paraphrases and hard negatives from the sampled anchor files.

Run from the project root with the local Ollama service available:
    python scripts/generate_queries.py

The script writes test_queries.jsonl and generation_manifest.json under
data/processed. It does not alter the source anchor, dev, or unrelated files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.corpus_config import load_corpus_config, require_stage_ready  # noqa: E402
from src.dataset_generation.hard_negative_gen import generate_hard_negative  # noqa: E402
from src.dataset_generation.paraphrase_gen import generate_paraphrase_records  # noqa: E402
from src.dataset_generation.text_processing import (  # noqa: E402
    normalized_edit_distance,
    token_jaccard,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _covariates(query: str, anchor: str) -> dict[str, float]:
    return {
        "jaccard_to_anchor": token_jaccard(query, anchor),
        "edit_dist_to_anchor": normalized_edit_distance(query, anchor),
    }


def build_queries(project_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_corpus_config(config_path)
    require_stage_ready(config, "paraphrase_generation")
    require_stage_ready(config, "hard_negative_generation")
    corpus = config["corpus"]
    processed = project_root / "data" / "processed"
    anchors = _read_jsonl(processed / "anchors.jsonl")
    dev = _read_jsonl(processed / "dev_anchors.jsonl")
    unrelated = _read_jsonl(processed / "unrelated.jsonl")
    if not anchors or not dev:
        raise ValueError("Build the sampled anchors and dev anchors before generation")

    seed = corpus["seed"]
    rng = random.Random(seed)
    generator = corpus["models"]["generator"]
    paraphrase_cfg = corpus["paraphrases"]
    negative_cfg = corpus["hard_negatives"]
    dev_questions = [row["question"] for row in dev]
    output: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {level: [] for level in paraphrase_cfg["levels"]}

    for anchor_index, anchor in enumerate(anchors, start=1):
        if anchor_index == 1 or anchor_index % 10 == 0 or anchor_index == len(anchors):
            print(
                f"Generating paraphrases for anchor {anchor_index}/{len(anchors)}",
                flush=True,
            )
        sample_count = rng.randint(
            paraphrase_cfg["style_examples_minimum"],
            paraphrase_cfg["style_examples_maximum"],
        )
        examples = rng.sample(dev_questions, sample_count)
        generated = generate_paraphrase_records(
            seed_question=anchor["question"],
            answers=anchor["gold_answers"],
            style_examples=examples,
            model=generator["id"],
            temperature=generator["temperature"],
            minimum_length_ratio=paraphrase_cfg["minimum_length_ratio"],
            maximum_length_ratio=paraphrase_cfg["maximum_length_ratio"],
            preserve_numbers_and_years=paraphrase_cfg["preserve_anchor_numbers_and_years"],
            maximum_retries=paraphrase_cfg["maximum_retries"],
            levels=paraphrase_cfg["levels"],
        )
        for level in generated.generation_failures:
            failures[level].append(anchor["source_id"])
        for item in generated.records:
            output.append({
                "query_id": f"{anchor['source_id']}:{item['level']}",
                "anchor_id": anchor["source_id"],
                "question": item["text"],
                "role": "paraphrase",
                "subtype": item["level"],
                "should_hit": True,
                "gold_answers": item["gold_answers"],
                "generator_model": item["generator_model"],
                "prompt_hash": item["prompt_hash"],
                "seed": seed,
                "audit_status": "pending",
                **_covariates(item["text"], anchor["question"]),
            })

    negative_counts = {kind: 0 for kind in negative_cfg["types"]}
    not_applicable = {kind: 0 for kind in negative_cfg["types"]}
    invalid_negative = {kind: 0 for kind in negative_cfg["types"]}
    for kind in negative_cfg["types"]:
        print(f"Generating hard negatives of type {kind}", flush=True)
        randomized_anchors = list(anchors)
        rng.shuffle(randomized_anchors)
        for anchor in randomized_anchors:
            if negative_counts[kind] >= negative_cfg["quota_per_type"]:
                break
            sample_count = rng.randint(
                paraphrase_cfg["style_examples_minimum"],
                paraphrase_cfg["style_examples_maximum"],
            )
            examples = rng.sample(dev_questions, sample_count)
            try:
                item = generate_hard_negative(
                    seed_question=anchor["question"],
                    seed_answer=anchor["gold_answers"],
                    negative_type=kind,
                    style_examples=examples,
                    model=generator["id"],
                    temperature=generator["temperature"],
                )
            except ValueError:
                invalid_negative[kind] += 1
                continue
            if item["status"] == "NOT_APPLICABLE":
                if not negative_cfg["not_applicable_allowed"]:
                    raise ValueError("generator returned NOT_APPLICABLE but config disallows it")
                not_applicable[kind] += 1
                continue
            output.append({
                "query_id": f"{anchor['source_id']}:{kind}",
                "anchor_id": anchor["source_id"],
                "question": item["text"],
                "role": "hard_negative",
                "subtype": kind,
                "should_hit": False,
                "gold_answers": None,
                "answer_difference_statement": item["answer_difference_statement"],
                "generator_model": item["generator_model"],
                "prompt_hash": item["prompt_hash"],
                "seed": seed,
                "audit_status": "pending",
                **_covariates(item["text"], anchor["question"]),
            })
            negative_counts[kind] += 1

    for row in unrelated:
        best_anchor = max(
            anchors,
            key=lambda anchor: token_jaccard(row["question"], anchor["question"]),
        )
        output.append({
            "query_id": row["source_id"],
            "anchor_id": best_anchor["source_id"],
            "covariate_reference": "maximum_token_jaccard_anchor",
            "question": row["question"],
            "role": "unrelated",
            "subtype": "unrelated",
            "should_hit": False,
            "gold_answers": row["gold_answers"],
            "generator_model": None,
            "prompt_hash": None,
            "seed": seed,
            "audit_status": "pending",
            **_covariates(row["question"], best_anchor["question"]),
        })

    output_path = processed / "test_queries.jsonl"
    _write_jsonl(output_path, output)
    manifest = {
        "seed": seed,
        "generator_model": generator["id"],
        "query_count": len(output),
        "paraphrase_failures_by_level": {
            key: len(value) for key, value in failures.items()
        },
        "hard_negative_counts": negative_counts,
        "hard_negative_not_applicable_counts": not_applicable,
        "hard_negative_invalid_response_counts": invalid_negative,
        "hard_negative_quota_per_type": negative_cfg["quota_per_type"],
        "source_files": ["anchors.jsonl", "dev_anchors.jsonl", "unrelated.jsonl"],
    }
    manifest_path = processed / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs" / "corpus.yaml"
    )
    args = parser.parse_args()
    manifest = build_queries(_PROJECT_ROOT, args.config.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

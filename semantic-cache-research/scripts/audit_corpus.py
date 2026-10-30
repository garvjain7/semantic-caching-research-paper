"""Prepare the blinded audit form or summarize completed human ratings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.compute_metrics import wilson_ci  # noqa: E402
from src.corpus_config import load_corpus_config  # noqa: E402


FORM_FIELDS = (
    "audit_id",
    "query_text",
    "matched_anchor_text",
    "cached_response",
    "same_information_need",
    "correct_answer_differs",
    "cached_response_acceptable",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_audit(project_root: Path, result_dir: Path, config_path: Path) -> Path:
    config = load_corpus_config(config_path)["corpus"]
    processed = project_root / "data" / "processed"
    queries = _read_jsonl(processed / "test_queries.jsonl")
    retrievals = _read_jsonl(result_dir / "retrieval_records.jsonl")
    query_by_id = {row["query_id"]: row for row in queries}
    chosen_config = config["audit"]["match_configuration"]
    retrieval_by_id = {
        row["query_id"]: row
        for row in retrievals
        if row["strategy"] == chosen_config
    }
    strata = config["audit"]["strata"]
    sample_size = config["audit"]["samples_per_stratum"]
    rng = random.Random(config["seed"])
    selected: list[tuple[str, dict[str, Any]]] = []
    for stratum in strata:
        candidates = [row for row in queries if row["subtype"] == stratum]
        if len(candidates) < sample_size:
            raise ValueError(
                f"stratum {stratum!r} has {len(candidates)} items; needs {sample_size}"
            )
        selected.extend((stratum, row) for row in rng.sample(candidates, sample_size))
    rng.shuffle(selected)

    audit_dir = project_root / "data" / "processed"
    form_path = audit_dir / "audit_sample.csv"
    key_path = audit_dir / "audit_key.jsonl"
    key_rows = []
    with form_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FORM_FIELDS)
        writer.writeheader()
        for index, (stratum, query) in enumerate(selected, start=1):
            query_id = query["query_id"]
            retrieval = retrieval_by_id.get(query_id)
            if retrieval is None:
                raise ValueError(f"missing {chosen_config} retrieval for {query_id!r}")
            audit_id = f"A{index:03d}"
            writer.writerow({
                "audit_id": audit_id,
                "query_text": query["question"],
                "matched_anchor_text": retrieval["anchor_question"],
                "cached_response": retrieval["anchor_answer"],
                "same_information_need": "",
                "correct_answer_differs": "",
                "cached_response_acceptable": "",
            })
            key_rows.append({
                "audit_id": audit_id,
                "query_id": query_id,
                "stratum": stratum,
                "configuration": chosen_config,
            })
    _write_jsonl(key_path, key_rows)
    (audit_dir / "audit_instructions.md").write_text(
        "# Human audit instructions\n\n"
        "For each row, answer each field yes or no. Judge the query and cached "
        "response on their merits. Do not infer the intended subtype. Leave no "
        "rating blank. `same_information_need` asks whether the new question "
        "seeks the same information as the matched anchor. "
        "`correct_answer_differs` asks whether the query requires a different "
        "correct answer. `cached_response_acceptable` asks whether the shown "
        "cached response correctly answers the query. The `audit_key.jsonl` "
        "file is the separate scoring key; keep it from the rater.\n",
        encoding="utf-8",
    )
    return form_path


def _binary(value: str, field: str, audit_id: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"{audit_id}: {field} must be yes or no")
    return normalized == "yes"


def _cohen_kappa(human: list[bool], judge: list[bool]) -> float | None:
    if not human:
        return None
    observed = sum(left == right for left, right in zip(human, judge)) / len(human)
    human_yes = sum(human) / len(human)
    judge_yes = sum(judge) / len(judge)
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def analyze_audit(
    project_root: Path,
    result_dir: Path,
    config_path: Path,
    audit_csv: Path | None = None,
    output_path: Path | None = None,
    ratings_source: str = "human",
) -> dict[str, Any]:
    config = load_corpus_config(config_path)["corpus"]
    processed = project_root / "data" / "processed"
    audit_csv = audit_csv or processed / "audit_sample.csv"
    with audit_csv.open(encoding="utf-8-sig", newline="") as source:
        rating_rows = list(csv.DictReader(source))
    key_by_id = {row["audit_id"]: row for row in _read_jsonl(processed / "audit_key.jsonl")}
    verdict_by_query = {
        row["query_id"]: row["verdict"]
        for row in _read_jsonl(result_dir / "judge_verdicts.jsonl")
        if row["configuration"] == config["audit"]["match_configuration"]
    }
    ratings_by_stratum: dict[str, list[bool]] = {
        stratum: [] for stratum in config["audit"]["strata"]
    }
    rating_acceptability: list[bool] = []
    judge_acceptability: list[bool] = []
    for row in rating_rows:
        audit_id = row["audit_id"]
        key = key_by_id.get(audit_id)
        if key is None:
            raise ValueError(f"audit key missing for {audit_id!r}")
        same_need = _binary(row["same_information_need"], "same_information_need", audit_id)
        differs = _binary(row["correct_answer_differs"], "correct_answer_differs", audit_id)
        acceptable = _binary(
            row["cached_response_acceptable"], "cached_response_acceptable", audit_id
        )
        stratum = key["stratum"]
        if stratum in {"P1", "P2", "P3", "P4"}:
            correct = same_need
        elif stratum in {"E", "N", "T", "S"}:
            correct = differs
        else:
            correct = not acceptable
        ratings_by_stratum[stratum].append(correct)

        verdict = verdict_by_query.get(key["query_id"])
        if verdict is None:
            raise ValueError(f"judge verdict missing for audited query {key['query_id']!r}")
        rating_acceptability.append(acceptable)
        judge_acceptability.append(verdict == "correct_hit")

    per_stratum = {}
    for stratum, correct_ratings in ratings_by_stratum.items():
        successes = sum(correct_ratings)
        low, point, high = wilson_ci(successes, len(correct_ratings))
        per_stratum[stratum] = {
            "correct": successes,
            "total": len(correct_ratings),
            "accuracy": point,
            "wilson_95": [low, high],
            "regeneration_required": point < config["audit"]["regeneration_accuracy_threshold"],
        }
    kappa = _cohen_kappa(rating_acceptability, judge_acceptability)
    summary = {
        "configuration": config["audit"]["match_configuration"],
        "audit_items": len(rating_rows),
        "ratings_source": ratings_source,
        "ratings_file": str(audit_csv),
        "ratings_judge_raw_agreement": (
            sum(r == j for r, j in zip(rating_acceptability, judge_acceptability))
            / len(rating_acceptability)
            if rating_acceptability else None
        ),
        "ratings_judge_cohen_kappa": kappa,
        "minimum_cohen_kappa": config["audit"]["minimum_cohen_kappa"],
        "meets_kappa_target": kappa is not None and kappa >= config["audit"]["minimum_cohen_kappa"],
        "regeneration_threshold": config["audit"]["regeneration_accuracy_threshold"],
        "regeneration_required_strata": [
            stratum
            for stratum, values in per_stratum.items()
            if values["regeneration_required"]
        ],
        "per_stratum_intended_label_accuracy": per_stratum,
    }
    output_path = output_path or processed / (
        "audit_summary.json" if ratings_source == "human" else "audit_summary_ai_consensus.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "analyze"])
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs" / "corpus.yaml"
    )
    parser.add_argument(
        "--results", type=Path, default=_PROJECT_ROOT / "results" / "threshold_sweep"
    )
    parser.add_argument("--ratings", type=Path, help="Completed audit ratings CSV")
    parser.add_argument(
        "--ratings-source", choices=("human", "ai_consensus"), default="human"
    )
    parser.add_argument("--output", type=Path, help="Summary JSON output path")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare_audit(_PROJECT_ROOT, args.results, args.config.resolve()))
    else:
        summary = analyze_audit(
            _PROJECT_ROOT,
            args.results,
            args.config.resolve(),
            audit_csv=args.ratings,
            output_path=args.output,
            ratings_source=args.ratings_source,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

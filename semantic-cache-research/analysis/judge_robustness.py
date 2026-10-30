"""Analyze agreement and label sensitivity on the saved AI audit sample.

This is a supplemental analysis of agreement among the configured AI systems.
It reads the three rating files that reproduce the saved three-model
consensus, the audit key, and persisted retrieval/judge records. It writes
only to results/judge_robustness/ and never changes the main experiment.

Run from the repository root with::

    python -m analysis.judge_robustness
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .compute_metrics import wilson_ci
except ImportError:  # supports direct execution as a script too
    from compute_metrics import wilson_ci


RATING_FIELDS = (
    "same_information_need",
    "correct_answer_differs",
    "cached_response_acceptable",
)
THRESHOLDS = tuple(round(0.50 + step * 0.05, 2) for step in range(10))
AUDIT_CONFIGURATION = "minilm_cosine"


@dataclass(frozen=True)
class RaterSource:
    key: str
    filename: str
    display_name: str


RATERS = (
    RaterSource("chatgpt_gpt6_luna", "audit_sample_rated.csv", "ChatGPT GPT-6 Luna"),
    RaterSource("sonnet_5_5", "audit_sample_rated (1).csv", "Sonnet 5.5"),
    RaterSource(
        "claude_sonnet_4_6",
        "audit_sample_antigravity_v2.csv",
        "Claude Sonnet 4.6 (Thinking; reviewer_model field)",
    ),
)


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    if "audit_id" not in rows[0]:
        raise ValueError(f"Missing audit_id in {path}")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = row["audit_id"].strip()
        if not audit_id or audit_id in indexed:
            raise ValueError(f"Blank or duplicate audit_id {audit_id!r} in {path}")
        indexed[audit_id] = row
    return indexed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _kappa(left: list[str], right: list[str]) -> float | None:
    """Cohen's kappa for two equally sized yes/no rating vectors."""
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    labels = ("yes", "no")
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return round((observed - expected) / (1 - expected), 6)


def _validate_ratings(
    rows_by_rater: dict[str, dict[str, dict[str, str]]],
    audit_ids: list[str],
) -> None:
    expected = set(audit_ids)
    for source in RATERS:
        rows = rows_by_rater[source.key]
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(
                f"Audit IDs differ in {source.filename}; missing={missing[:5]}, extra={extra[:5]}"
            )
        for audit_id, row in rows.items():
            for field in RATING_FIELDS:
                if row.get(field, "").strip().lower() not in {"yes", "no"}:
                    raise ValueError(
                        f"Expected yes/no in {source.filename}:{audit_id}:{field}"
                    )


def _agreement_outputs(
    root: Path,
    output_dir: Path,
    rows_by_rater: dict[str, dict[str, dict[str, str]]],
    audit_ids: list[str],
    cache_response_matches_original: dict[str, set[str]],
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    rater_summaries: dict[str, dict[str, Any]] = {}
    for source in RATERS:
        row_counts: Counter[str] = Counter()
        for field in RATING_FIELDS:
            row_counts[field] = sum(
                rows_by_rater[source.key][audit_id][field].strip().lower() == "yes"
                for audit_id in audit_ids
            )
        rater_summaries[source.key] = {
            "display_name": source.display_name,
            "source_file": source.filename,
            "ratings_per_field": len(audit_ids),
            "yes_counts": dict(row_counts),
        }

    for left_index, left_source in enumerate(RATERS):
        for right_source in RATERS[left_index + 1 :]:
            for field in RATING_FIELDS:
                comparable_ids = audit_ids
                excluded = 0
                if field == "cached_response_acceptable":
                    comparable_ids = [
                        audit_id for audit_id in audit_ids
                        if audit_id in cache_response_matches_original[left_source.key]
                        and audit_id in cache_response_matches_original[right_source.key]
                    ]
                    excluded = len(audit_ids) - len(comparable_ids)
                left = [rows_by_rater[left_source.key][i][field].strip().lower() for i in comparable_ids]
                right = [rows_by_rater[right_source.key][i][field].strip().lower() for i in comparable_ids]
                pair_rows.append({
                    "analysis_type": "ai_rater_pairwise_agreement",
                    "rater_a": left_source.display_name,
                    "rater_a_file": left_source.filename,
                    "rater_b": right_source.display_name,
                    "rater_b_file": right_source.filename,
                    "rating_field": field,
                    "n": len(comparable_ids),
                    "excluded_response_content_mismatch_rows": excluded,
                    "raw_agreement": round(sum(a == b for a, b in zip(left, right)) / len(left), 6) if left else None,
                    "cohen_kappa": _kappa(left, right),
                })
    _write_csv(output_dir / "pairwise_ai_rater_agreement.csv", pair_rows)

    per_row: list[dict[str, Any]] = []
    field_unanimity: Counter[str] = Counter()
    all_fields_unanimous = 0
    field_splits: Counter[str] = Counter()
    for audit_id in audit_ids:
        result: dict[str, Any] = {"audit_id": audit_id}
        every_field_unanimous = True
        for field in RATING_FIELDS:
            votes = [rows_by_rater[r.key][audit_id][field].strip().lower() for r in RATERS]
            distinct = set(votes)
            result[f"{field}_votes"] = "/".join(votes)
            result[f"{field}_unanimous"] = len(distinct) == 1
            result[f"{field}_majority"] = "yes" if votes.count("yes") >= 2 else "no"
            if field == "cached_response_acceptable":
                result["source_response_content_matches_original"] = all(
                    audit_id in cache_response_matches_original[rater.key] for rater in RATERS
                )
            if len(distinct) == 1:
                field_unanimity[field] += 1
            else:
                field_splits[field] += 1
                every_field_unanimous = False
        result["all_fields_unanimous"] = every_field_unanimous
        all_fields_unanimous += every_field_unanimous
        per_row.append(result)
    _write_csv(output_dir / "row_level_ai_disagreement.csv", per_row)

    consensus_file = root / "data" / "processed" / "audit_sample_consensus_3models.csv"
    consensus_rows = _read_csv(consensus_file)
    consensus_mismatches = 0
    for audit_id in audit_ids:
        for field in RATING_FIELDS:
            votes = [rows_by_rater[r.key][audit_id][field].strip().lower() for r in RATERS]
            majority = "yes" if votes.count("yes") >= 2 else "no"
            consensus_mismatches += (
                majority != consensus_rows[audit_id][field].strip().lower()
            )

    return {
        "rating_sources": rater_summaries,
        "pairwise_comparisons": len(pair_rows),
        "row_count": len(audit_ids),
        "field_unanimity_counts": dict(field_unanimity),
        "field_disagreement_counts": dict(field_splits),
        "all_three_fields_unanimous_rows": all_fields_unanimous,
        "majority_vs_saved_consensus_mismatched_cells": consensus_mismatches,
    }


def _judge_and_threshold_outputs(
    root: Path,
    output_dir: Path,
    rows_by_rater: dict[str, dict[str, dict[str, str]]],
    audit_ids: list[str],
    query_by_audit_id: dict[str, str],
    cache_response_matches_original: dict[str, set[str]],
) -> dict[str, Any]:
    results = root / "results" / "threshold_sweep"
    judge_rows = _read_jsonl(results / "judge_verdicts.jsonl")
    retrieval_rows = _read_jsonl(results / "retrieval_records.jsonl")
    audit_query_ids = set(query_by_audit_id.values())

    judges: dict[tuple[str, str], str] = {}
    for row in judge_rows:
        if row.get("query_id") in audit_query_ids:
            key = (row["query_id"], row["configuration"])
            if key in judges:
                raise ValueError(f"Duplicate saved judge verdict for {key}")
            judges[key] = row["verdict"]
    similarities: dict[tuple[str, str], float] = {}
    for row in retrieval_rows:
        if row.get("query_id") in audit_query_ids and row.get("strategy") == AUDIT_CONFIGURATION:
            key = (row["query_id"], row["strategy"])
            if key in similarities:
                raise ValueError(f"Duplicate saved retrieval record for {key}")
            similarities[key] = float(row["similarity"])

    per_rater_judge: list[dict[str, Any]] = []
    for source in RATERS:
        for configuration in sorted({cfg for _, cfg in judges}):
            humanlike: list[str] = []
            judged: list[str] = []
            for audit_id in audit_ids:
                if audit_id not in cache_response_matches_original[source.key]:
                    continue
                query_id = query_by_audit_id[audit_id]
                verdict = judges.get((query_id, configuration))
                if verdict not in {"correct_hit", "false_hit"}:
                    continue
                humanlike.append(rows_by_rater[source.key][audit_id]["cached_response_acceptable"].strip().lower())
                judged.append("yes" if verdict == "correct_hit" else "no")
            per_rater_judge.append({
                "analysis_type": "ai_rating_vs_phi4_mini",
                "ai_rater": source.display_name,
                "ai_rater_file": source.filename,
                "configuration": configuration,
                "rating_field": "cached_response_acceptable",
                "n": len(humanlike),
                "raw_agreement": round(sum(a == b for a, b in zip(humanlike, judged)) / len(humanlike), 6) if humanlike else None,
                "cohen_kappa": _kappa(humanlike, judged),
            })
    _write_csv(output_dir / "ai_rater_vs_phi4_agreement.csv", per_rater_judge)

    sensitivity_rows: list[dict[str, Any]] = []
    for source in RATERS:
        for threshold in THRESHOLDS:
            hits = false_hits = acceptable_hits = 0
            eligible_ids = [
                audit_id for audit_id in audit_ids
                if audit_id in cache_response_matches_original[source.key]
            ]
            for audit_id in eligible_ids:
                query_id = query_by_audit_id[audit_id]
                similarity = similarities.get((query_id, AUDIT_CONFIGURATION))
                if similarity is None:
                    continue
                acceptable = rows_by_rater[source.key][audit_id]["cached_response_acceptable"].strip().lower() == "yes"
                if similarity >= threshold:
                    hits += 1
                    false_hits += not acceptable
                else:
                    acceptable_hits += acceptable
            low, point, high = wilson_ci(false_hits, hits)
            sensitivity_rows.append({
                "analysis_type": "stratified_90_row_audit_diagnostic_only",
                "ai_rater": source.display_name,
                "ai_rater_file": source.filename,
                "configuration": AUDIT_CONFIGURATION,
                "threshold": threshold,
                "audit_rows": len(eligible_ids),
                "cache_hits": hits,
                "unacceptable_cached_responses_among_hits": false_hits,
                "false_hit_rate_among_hits": point,
                "false_hit_wilson_95_low": low,
                "false_hit_wilson_95_high": high,
                "acceptable_responses_missed": acceptable_hits,
                "acceptable_response_miss_rate": round(acceptable_hits / sum(
                    rows_by_rater[source.key][audit_id]["cached_response_acceptable"].strip().lower() == "yes"
                    for audit_id in eligible_ids
                ), 6) if eligible_ids and sum(
                    rows_by_rater[source.key][audit_id]["cached_response_acceptable"].strip().lower() == "yes"
                    for audit_id in eligible_ids
                ) else None,
            })
    _write_csv(output_dir / "audit_threshold_sensitivity.csv", sensitivity_rows)

    return {
        "judge_records_matched": len(judges),
        "retrieval_records_matched_for_audit_configuration": len(similarities),
        "judge_comparison_note": "Phi4 verdicts are compared only on the same audit query/configuration; all reported agreement values are inter-model comparisons.",
        "threshold_sensitivity_note": "The 90 audit rows are stratified and were sampled from the 663-query pre-recovery corpus. Rates are descriptive diagnostics, not representative estimates for the final 700-query corpus.",
        "threshold_sensitivity_configuration": AUDIT_CONFIGURATION,
    }


def analyze(project_root: Path, output_dir: Path) -> Path:
    processed = project_root / "data" / "processed"
    key_rows = _read_jsonl(processed / "audit_key.jsonl")
    audit_ids = [row["audit_id"] for row in key_rows]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("Duplicate audit_id in audit_key.jsonl")
    query_by_audit_id = {row["audit_id"]: row["query_id"] for row in key_rows}
    if len(query_by_audit_id) != len(key_rows):
        raise ValueError("Duplicate audit_id in audit_key.jsonl")

    rows_by_rater = {
        source.key: _read_csv(processed / source.filename)
        for source in RATERS
    }
    _validate_ratings(rows_by_rater, audit_ids)

    audit_sample = _read_csv(processed / "audit_sample.csv")
    if set(audit_sample) != set(audit_ids):
        raise ValueError("audit_sample.csv IDs do not match audit_key.jsonl")
    cache_response_matches_original: dict[str, set[str]] = {}
    response_mismatches: dict[str, list[str]] = {}
    for source in RATERS:
        response_matches: set[str] = set()
        mismatches: list[str] = []
        for audit_id in audit_ids:
            for column in ("query_text", "matched_anchor_text"):
                if rows_by_rater[source.key][audit_id].get(column) != audit_sample[audit_id].get(column):
                    raise ValueError(f"Audit content mismatch in {source.filename}:{audit_id}:{column}")
            if rows_by_rater[source.key][audit_id].get("cached_response") == audit_sample[audit_id].get("cached_response"):
                response_matches.add(audit_id)
            else:
                mismatches.append(audit_id)
        cache_response_matches_original[source.key] = response_matches
        response_mismatches[source.key] = mismatches

    output_dir.mkdir(parents=True, exist_ok=True)
    agreement = _agreement_outputs(
        project_root, output_dir, rows_by_rater, audit_ids,
        cache_response_matches_original,
    )
    result_metrics = _judge_and_threshold_outputs(
        project_root, output_dir, rows_by_rater, audit_ids, query_by_audit_id,
        cache_response_matches_original,
    )
    audit_summary = json.loads((processed / "audit_summary_ai_consensus.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed / "final_corpus_manifest.json").read_text(encoding="utf-8"))
    summary = {
        "analysis": "AI judge robustness and audit label sensitivity",
        "interpretation": "Supplemental AI-vs-AI analysis of rating-source consistency and threshold sensitivity.",
        "audit_sample_rows": len(audit_ids),
        "audit_sampled_from_query_count": audit_summary.get("corpus_query_count_at_audit_sampling"),
        "final_corpus_query_count": manifest.get("counts", {}).get("test_queries.jsonl"),
        "saved_ai_consensus_judge_kappa": audit_summary.get("ratings_judge_cohen_kappa"),
        "rating_sources": agreement["rating_sources"],
        "model_family_note": "Claude Sonnet 4.6 and Sonnet 5.5 are from the same model family, so these three outputs should not be treated as fully independent evidence.",
        "response_content_mismatch_audit_ids_by_rater": response_mismatches,
        "rows_with_exact_response_match_across_all_sources": sum(
            all(audit_id in cache_response_matches_original[source.key] for source in RATERS)
            for audit_id in audit_ids
        ),
        "cached_response_unanimous_on_exactly_aligned_rows": sum(
            len({rows_by_rater[source.key][audit_id]["cached_response_acceptable"].strip().lower() for source in RATERS}) == 1
            for audit_id in audit_ids
            if all(audit_id in cache_response_matches_original[source.key] for source in RATERS)
        ),
        "all_three_fields_unanimous_on_exactly_aligned_rows": sum(
            all(len({rows_by_rater[source.key][audit_id][field].strip().lower() for source in RATERS}) == 1 for field in RATING_FIELDS)
            for audit_id in audit_ids
            if all(audit_id in cache_response_matches_original[source.key] for source in RATERS)
        ),
        "pairwise_rater_comparisons": agreement["pairwise_comparisons"],
        "field_unanimity_counts_out_of_90": agreement["field_unanimity_counts"],
        "field_disagreement_counts_out_of_90": agreement["field_disagreement_counts"],
        "rows_unanimous_on_all_three_fields": agreement["all_three_fields_unanimous_rows"],
        "majority_vs_saved_consensus_mismatched_rating_cells": agreement["majority_vs_saved_consensus_mismatched_cells"],
        **result_metrics,
        "input_sha256": {
            source.filename: hashlib.sha256((processed / source.filename).read_bytes()).hexdigest()
            for source in RATERS
        },
    }
    _write_json(output_dir / "summary.json", summary)
    readme = """# Supplemental AI judge robustness analysis\n\nThese outputs are an isolated sensitivity analysis. They do not alter the main threshold sweep or its files. They quantify consistency among the configured AI rating sources and Phi-4-mini. The Claude Sonnet 4.6 and Sonnet 5.5 sources share a model family, so the three ratings are not fully independent evidence.\n\n## Outputs\n\n- `pairwise_ai_rater_agreement.csv`: raw agreement and Cohen's kappa by AI-rater pair and rating field. For cached-response ratings, rows where the supplied response text differed from the original audit sample are excluded and counted.\n- `row_level_ai_disagreement.csv`: per-row votes, majority labels, and unanimity flags; includes a flag for response text matching the original.\n- `ai_rater_vs_phi4_agreement.csv`: each AI rater versus saved phi4-mini verdicts, by retrieval configuration. Response-text mismatches are excluded from this comparison.\n- `audit_threshold_sensitivity.csv`: threshold replay using each model's cached-response-acceptability labels on the same stratified audit sample. Treat these as descriptive diagnostics, not population estimates.\n- `summary.json`: provenance, counts, validation, and interpretation notes.\n\nThe audit sample was drawn from the 663-query corpus before 37 paraphrases were recovered; the final corpus has 700 queries. The original consensus was checked against the majority of the three configured source files. The raw source filenames and SHA-256 hashes are retained in `summary.json`.\n\nRegenerate with `python -m analysis.judge_robustness` from the repository root. Outputs are written only in this directory.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.project_root / "results" / "judge_robustness"
    print(analyze(args.project_root, output_dir))


if __name__ == "__main__":
    main()

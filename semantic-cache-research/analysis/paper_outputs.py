"""Create paper figures and CSV tables from experiment and audit outputs.

This module produces the requested artifacts only; it does not compose a
paper or narrative report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .plots import (
        plot_embedding_comparison,
        plot_subtype_disagreement,
        plot_tradeoff_curve,
    )
except ImportError:
    from plots import (
        plot_embedding_comparison,
        plot_subtype_disagreement,
        plot_tradeoff_curve,
    )


CONFIGURATIONS = (
    "minilm_cosine",
    "bge_cosine",
    "minilm_crossencoder_top5",
    "lexical_difflib",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows available for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_outputs(project_root: Path, result_dir: Path) -> Path:
    metric_rows = {
        name: _read_jsonl(result_dir / f"metrics_{name}.jsonl")
        for name in CONFIGURATIONS
    }
    if any(not rows for rows in metric_rows.values()):
        raise ValueError("Run all four configurations and metric sweeps first")

    processed = project_root / "data" / "processed"
    output_dir = project_root / "paper" / "generated"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    plot_tradeoff_curve(metric_rows["minilm_cosine"], figure_dir / "figure_1.png")
    plot_embedding_comparison(metric_rows, figure_dir / "figure_2.png")
    plot_subtype_disagreement(metric_rows["minilm_cosine"], figure_dir / "figure_3.png")

    corpus_counts = {}
    for name in ("anchors", "dev_anchors", "held_out_pool", "unrelated", "test_queries", "cache_entries"):
        path = processed / f"{name}.jsonl"
        corpus_counts[name] = len(_read_jsonl(path)) if path.exists() else 0
    subtype_counts = Counter(
        row["subtype"] for row in _read_jsonl(processed / "test_queries.jsonl")
    ) if (processed / "test_queries.jsonl").exists() else Counter()
    audit_path = processed / "audit_sample.csv"
    audit_items = sum(1 for _ in audit_path.open(encoding="utf-8-sig")) - 1 if audit_path.exists() else 0
    table1 = [{
        **corpus_counts,
        "query_counts_by_subtype": json.dumps(dict(subtype_counts), ensure_ascii=False),
        "audit_items": max(0, audit_items),
    }]
    _write_csv(table_dir / "table_1_corpus_audit.csv", table1)

    table2 = []
    for name, rows in metric_rows.items():
        for row in rows:
            table2.append({
                "configuration": name,
                "threshold": row["threshold"],
                "total_queries": row["total_queries"],
                "judged_false_hits": row["judged_false_hit_rate"]["numerator"],
                "judged_false_hit_rate": row["judged_false_hit_rate"]["point"],
                "judged_false_hit_ci_low": row["judged_false_hit_rate"]["ci_low"],
                "judged_false_hit_ci_high": row["judged_false_hit_rate"]["ci_high"],
                "judge_label_disagreement_rate": row["judge_label_disagreement_rate"]["point"],
                "judge_label_disagreement_ci_low": row["judge_label_disagreement_rate"]["ci_low"],
                "judge_label_disagreement_ci_high": row["judge_label_disagreement_rate"]["ci_high"],
                "query_false_hit_rate": row["query_false_hit_rate"]["point"],
                "query_false_hit_ci_low": row["query_false_hit_rate"]["ci_low"],
                "query_false_hit_ci_high": row["query_false_hit_rate"]["ci_high"],
                "false_miss_rate": row["false_miss_rate"]["point"],
                "false_miss_ci_low": row["false_miss_rate"]["ci_low"],
                "false_miss_ci_high": row["false_miss_rate"]["ci_high"],
                "cost_savings_rate": row["cost_savings_rate"]["point"],
                "cost_savings_ci_low": row["cost_savings_rate"]["ci_low"],
                "cost_savings_ci_high": row["cost_savings_rate"]["ci_high"],
            })
    _write_csv(table_dir / "table_2_threshold_metrics.csv", table2)

    table3 = []
    for name, rows in metric_rows.items():
        for row in rows:
            for subtype, values in row["query_false_hit_rate_by_subtype"].items():
                table3.append({
                    "configuration": name,
                    "threshold": row["threshold"],
                    "subtype": subtype,
                    "query_false_hit_n": values["numerator"],
                    "query_n": values["denominator"],
                    "query_false_hit_rate": values["point"],
                    "query_false_hit_ci_low": values["ci_low"],
                    "query_false_hit_ci_high": values["ci_high"],
                    "false_miss_n": row["false_miss_rate_by_paraphrase_subtype"].get(subtype, {}).get("numerator"),
                    "false_miss_rate": row["false_miss_rate_by_paraphrase_subtype"].get(subtype, {}).get("point"),
                    "judge_disagreement_rate": row["judge_disagreement_by_subtype"][subtype]["point"],
                    "judge_disagreement_ci_low": row["judge_disagreement_by_subtype"][subtype]["ci_low"],
                    "judge_disagreement_ci_high": row["judge_disagreement_by_subtype"][subtype]["ci_high"],
                })
    _write_csv(table_dir / "table_3_subtype_metrics.csv", table3)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    result_dir = args.results or args.project_root / "results" / "threshold_sweep"
    print(generate_outputs(args.project_root, result_dir))


if __name__ == "__main__":
    main()

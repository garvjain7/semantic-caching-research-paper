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
        plot_jfh_threshold,
        plot_savings_threshold,
        plot_tradeoff_frontier,
        plot_hard_negative_heatmap,
        plot_disagreement_subtypes,
        plot_distance_error,
    )
except ImportError:
    from plots import (
        plot_jfh_threshold,
        plot_savings_threshold,
        plot_tradeoff_frontier,
        plot_hard_negative_heatmap,
        plot_disagreement_subtypes,
        plot_distance_error,
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

    plot_jfh_threshold(metric_rows, figure_dir / "figure_5_jfh_threshold.png")
    plot_savings_threshold(metric_rows, figure_dir / "figure_6_savings_threshold.png")
    plot_tradeoff_frontier(metric_rows, figure_dir / "figure_7_tradeoff_frontier.png")
    plot_hard_negative_heatmap(metric_rows, figure_dir / "figure_8_hard_negative_heatmap.png")
    plot_disagreement_subtypes(metric_rows, figure_dir / "figure_10_judge_label_disagreement.png")

    metadata_rows = _read_jsonl(processed / "test_queries.jsonl")
    metadata = {row["query_id"]: row for row in metadata_rows}
    retrieval_by_config = {}
    for name in CONFIGURATIONS:
        retrieval_by_config[name] = [r for r in _read_jsonl(result_dir / "retrieval_records.jsonl") if r["strategy"] == name]
    distance_metrics = plot_distance_error(retrieval_by_config, metadata, figure_dir / "figure_9_distance_error.png")

    corpus_counts = {}
    for name in ("anchors", "dev_anchors", "held_out_pool", "unrelated", "test_queries", "cache_entries"):
        path = processed / f"{name}.jsonl"
        corpus_counts[name] = len(_read_jsonl(path)) if path.exists() else 0
    subtype_counts = Counter(
        row["subtype"] for row in _read_jsonl(processed / "test_queries.jsonl")
    ) if (processed / "test_queries.jsonl").exists() else Counter()
    audit_path = processed / "audit_sample.csv"
    audit_items = sum(1 for _ in audit_path.open(encoding="utf-8-sig")) - 1 if audit_path.exists() else 0
    strata = {}
    for subtype in ("P1","P2","P3","P4","E","N","T","S","unrelated"):
        items = [row for row in metadata_rows if row.get("subtype") == subtype]
        strata[subtype] = {
            "n": len(items),
            "mean_jaccard": round(sum(float(x.get("jaccard_to_anchor",0)) for x in items)/len(items),4) if items else None,
            "mean_normalized_edit_distance": round(sum(float(x.get("edit_dist_to_anchor",0)) for x in items)/len(items),4) if items else None,
        }
    table1 = [{
        **corpus_counts,
        "query_counts_by_subtype": json.dumps(dict(subtype_counts), ensure_ascii=False),
        "stratum_statistics": json.dumps(strata, ensure_ascii=False),
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
                "label_false_hit_rate_among_hits": row["label_false_hit_rate"]["point"],
                "label_false_hit_ci_low": row["label_false_hit_rate"]["ci_low"],
                "label_false_hit_ci_high": row["label_false_hit_rate"]["ci_high"],
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
                disagreement = row["judge_disagreement_by_subtype"][subtype]
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
                    "judge_disagreement_rate": disagreement["point"] if disagreement["denominator"] else "NA",
                    "judge_disagreement_ci_low": disagreement["ci_low"] if disagreement["denominator"] else "NA",
                    "judge_disagreement_ci_high": disagreement["ci_high"] if disagreement["denominator"] else "NA",
                    "judge_disagreement_n": disagreement["numerator"],
                    "judge_disagreement_hits": disagreement["denominator"],
                })
    _write_csv(table_dir / "table_3_subtype_metrics.csv", table3)

    # One row per configuration at its minimum observed judged false-hit rate.
    operating = []
    for name, rows in metric_rows.items():
        row = min(rows, key=lambda r: r["judged_false_hit_rate"]["point"])
        operating.append({"configuration": name, "threshold": row["threshold"],
            "judged_false_hit_rate": row["judged_false_hit_rate"]["point"],
            "jfh_ci_low": row["judged_false_hit_rate"]["ci_low"], "jfh_ci_high": row["judged_false_hit_rate"]["ci_high"],
            "judged_false_hits": row["judged_false_hit_rate"]["numerator"], "judged_hits": row["judged_false_hit_rate"]["denominator"],
            "label_false_hit_rate_among_hits": row["label_false_hit_rate"]["point"],
            "label_fhr_ci_low": row["label_false_hit_rate"]["ci_low"], "label_fhr_ci_high": row["label_false_hit_rate"]["ci_high"],
            "query_false_hit_incidence": row["query_false_hit_rate"]["point"],
            "query_false_hit_ci_low": row["query_false_hit_rate"]["ci_low"], "query_false_hit_ci_high": row["query_false_hit_rate"]["ci_high"],
            "false_miss_rate": row["false_miss_rate"]["point"], "false_miss_ci_low": row["false_miss_rate"]["ci_low"], "false_miss_ci_high": row["false_miss_rate"]["ci_high"],
            "call_savings": row["cost_savings_rate"]["point"], "savings_ci_low": row["cost_savings_rate"]["ci_low"], "savings_ci_high": row["cost_savings_rate"]["ci_high"]})
    _write_csv(table_dir / "table_4_best_observed_operating_points.csv", operating)

    shared = []
    for name, rows in metric_rows.items():
        row = min(rows, key=lambda r: abs(r["threshold"] - .75))
        shared.append({"configuration": name, "threshold": row["threshold"],
            "judged_false_hit_rate": row["judged_false_hit_rate"]["point"],
            "jfh_ci_low": row["judged_false_hit_rate"]["ci_low"], "jfh_ci_high": row["judged_false_hit_rate"]["ci_high"],
            "label_false_hit_rate_among_hits": row["label_false_hit_rate"]["point"],
            "label_fhr_ci_low": row["label_false_hit_rate"]["ci_low"], "label_fhr_ci_high": row["label_false_hit_rate"]["ci_high"],
            "query_false_hit_incidence": row["query_false_hit_rate"]["point"],
            "query_false_hit_ci_low": row["query_false_hit_rate"]["ci_low"], "query_false_hit_ci_high": row["query_false_hit_rate"]["ci_high"],
            "false_miss_rate": row["false_miss_rate"]["point"], "false_miss_ci_low": row["false_miss_rate"]["ci_low"], "false_miss_ci_high": row["false_miss_rate"]["ci_high"],
            "call_savings": row["cost_savings_rate"]["point"], "savings_ci_low": row["cost_savings_rate"]["ci_low"], "savings_ci_high": row["cost_savings_rate"]["ci_high"],
            "mean_saved_latency_ms": row["mean_saved_latency_ms"]})
    _write_csv(table_dir / "table_5_shared_threshold_comparison.csv", shared)

    hn_rows = []
    subtype_rows = []
    for name, rows in metric_rows.items():
        row = min(rows, key=lambda r: abs(r["threshold"] - .75))
        for subtype, values in row["false_hit_rate_by_hard_negative_type"].items():
            hn_rows.append({"configuration": name, "threshold": row["threshold"], "type": subtype,
                "false_hits": values["numerator"], "queries": values["denominator"], "false_hit_rate": values["point"],
                "ci_low": values["ci_low"], "ci_high": values["ci_high"]})
        for subtype, values in row["judge_disagreement_by_subtype"].items():
            subtype_rows.append({"configuration": name, "threshold": row["threshold"], "subtype": subtype,
                "judge_label_disagreements": values["numerator"], "hits": values["denominator"],
                "disagreement_rate": values["point"] if values["denominator"] else "NA",
                "ci_low": values["ci_low"] if values["denominator"] else "NA",
                "ci_high": values["ci_high"] if values["denominator"] else "NA"})
    _write_csv(table_dir / "table_6_hard_negative_breakdown.csv", hn_rows)
    _write_csv(table_dir / "table_7_subtype_judge_disagreement.csv", subtype_rows)
    distance_rows = []
    for (config, measure), bins in distance_metrics.items():
        for label, (false_hits, hits) in zip(("0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1.0"), bins):
            if hits:
                p = false_hits / hits; z = 1.96; den = 1 + z*z/hits
                center = (p + z*z/(2*hits))/den
                half = z*((p*(1-p)/hits + z*z/(4*hits*hits))**.5)/den
                low, high = max(0, center-half), min(1, center+half)
            else:
                low = high = "NA"
            distance_rows.append({"configuration": config, "distance_measure": measure, "bin": label,
                "false_hits": false_hits, "cache_hits": hits,
                "label_false_hit_rate": round(false_hits/hits, 6) if hits else "NA",
                "wilson_95_ci_low": round(low, 6) if isinstance(low, float) else low,
                "wilson_95_ci_high": round(high, 6) if isinstance(high, float) else high})
    _write_csv(table_dir / "table_8_distance_bin_metrics.csv", distance_rows)
    judge_agreement = []
    for name, rows in metric_rows.items():
        row = min(rows, key=lambda r: abs(r["threshold"] - .75))
        value = row["judge_label_disagreement_rate"]
        judge_agreement.append({"configuration": name, "threshold": row["threshold"],
            "disagreements": value["numerator"], "hits": value["denominator"],
            "disagreement_rate": value["point"], "wilson_95_ci_low": value["ci_low"],
            "wilson_95_ci_high": value["ci_high"]})
    _write_csv(table_dir / "table_9_overall_judge_label_disagreement.csv", judge_agreement)
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

"""Run the four retrieval configurations and replay the configured sweep."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.compute_metrics import (  # noqa: E402
    compute_metrics_sweep,
    load_experiment_records,
    write_metrics_jsonl,
)
from src.cache import CacheIndex  # noqa: E402
from src.corpus_config import load_corpus_config, require_stage_ready  # noqa: E402
from src.embedding import Embedder  # noqa: E402
from src.experiment_runner import (  # noqa: E402
    run_fallback_pass,
    run_retrieval_pass,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _thresholds(sweep: dict[str, Any]) -> list[float]:
    start = Decimal(str(sweep["start"]))
    end = Decimal(str(sweep["end"]))
    step = Decimal(str(sweep["step"]))
    if step <= 0 or end < start:
        raise ValueError("threshold sweep must have a positive step and end >= start")
    values = []
    current = start
    while current <= end:
        values.append(float(current))
        current += step
    return values


def _build_index(cache_rows: list[dict], embedder: Embedder) -> CacheIndex:
    questions = [row["question"] for row in cache_rows]
    vectors = embedder.embed_batch(questions)
    index = CacheIndex(vectors.shape[1], embedding_backend=embedder.model_name)
    for row, vector in zip(cache_rows, vectors):
        index.add(vector, row)
    return index


def run(project_root: Path, corpus_config_path: Path, sweep_config_path: Path) -> Path:
    config = load_corpus_config(corpus_config_path)
    require_stage_ready(config, "judge_evaluation")
    require_stage_ready(config, "cache_responses")
    corpus = config["corpus"]
    processed = project_root / "data" / "processed"
    anchors = _read_jsonl(processed / "anchors.jsonl")
    cache_rows = _read_jsonl(processed / "cache_entries.jsonl")
    queries = _read_jsonl(processed / "test_queries.jsonl")
    if not anchors or not cache_rows or not queries:
        raise ValueError("Build anchors, cache entries, and test queries first")
    cache_by_id = {row["source_id"]: row for row in cache_rows}
    missing = [row["source_id"] for row in anchors if row["source_id"] not in cache_by_id]
    if missing:
        raise ValueError(f"cache responses missing for {len(missing)} anchors")
    cache_rows = [cache_by_id[row["source_id"]] for row in anchors]

    with sweep_config_path.open(encoding="utf-8") as source:
        sweep_config = yaml.safe_load(source)
    if sweep_config["llm_model"] != corpus["models"]["fallback"]["id"]:
        raise ValueError("threshold-sweep llm_model must match corpus fallback model")
    if sweep_config["embedding_model"] != corpus["embeddings"]["primary"]["id"]:
        raise ValueError("threshold-sweep embedding_model must match primary corpus embedding")
    thresholds = _thresholds(sweep_config["threshold_range"])
    models = corpus["embeddings"]
    minilm = Embedder(models["primary"]["id"])
    bge = Embedder(models["secondary"]["id"])
    indices = {
        "minilm": _build_index(cache_rows, minilm),
        "bge": _build_index(cache_rows, bge),
    }

    output_dir = project_root / "results" / sweep_config["experiment_name"]
    query_metadata = [
        {
            "query_id": row["query_id"],
            "role": row["role"],
            "subtype": row["subtype"],
            "should_hit": row["should_hit"],
        }
        for row in queries
    ]
    _write_jsonl(output_dir / "test_queries.jsonl", query_metadata)

    configurations = (
        ("minilm_cosine", minilm, indices["minilm"]),
        ("bge_cosine", bge, indices["bge"]),
        ("minilm_crossencoder_top5", minilm, indices["minilm"]),
        ("lexical_difflib", None, indices["minilm"]),
    )
    for name, embedder, index in configurations:
        run_retrieval_pass(
            queries=queries,
            embedder=embedder,
            strategy_name=name,
            cache_index=index,
            judge_model=corpus["models"]["judge"]["id"],
            output_dir=output_dir,
        )

    run_fallback_pass(
        queries=queries,
        fallback_model=corpus["models"]["fallback"]["id"],
        output_dir=output_dir,
    )
    for name, _, _ in configurations:
        records = load_experiment_records(
            output_dir,
            query_metadata_path=output_dir / "test_queries.jsonl",
            configuration=name,
        )
        metrics = compute_metrics_sweep(
            records,
            thresholds,
            judge_model=corpus["models"]["judge"]["id"],
        )
        write_metrics_jsonl(metrics, output_dir / f"metrics_{name}.jsonl")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "corpus.yaml",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "threshold_sweep.yaml",
    )
    args = parser.parse_args()
    print(run(_PROJECT_ROOT, args.corpus_config.resolve(), args.sweep_config.resolve()))


if __name__ == "__main__":
    main()

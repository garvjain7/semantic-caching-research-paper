"""Generate one deterministic Ollama response per sampled anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.corpus_config import load_corpus_config, require_stage_ready  # noqa: E402
from src.dataset_generation.ollama_generation import generate_text  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def build_cache(project_root: Path, config_path: Path) -> int:
    config = load_corpus_config(config_path)
    require_stage_ready(config, "cache_responses")
    corpus = config["corpus"]
    processed = project_root / "data" / "processed"
    anchors = _read_jsonl(processed / "anchors.jsonl")
    output_path = processed / "cache_entries.jsonl"
    completed = {
        row["source_id"] for row in _read_jsonl(output_path)
    } if output_path.exists() else set()
    model = corpus["cache_responses"]["model"]
    temperature = corpus["cache_responses"]["temperature"]

    with output_path.open("a", encoding="utf-8", newline="\n") as destination:
        for anchor in anchors:
            if anchor["source_id"] in completed:
                continue
            prompt = (
                "Answer the following question accurately and concisely. "
                "Return only the answer.\nQuestion: "
                f"{anchor['question']}"
            )
            answer, latency, tokens, prompt_hash = generate_text(
                model=model,
                prompt=prompt,
                temperature=temperature,
            )
            row = {
                "source_id": anchor["source_id"],
                "question": anchor["question"],
                "response": answer,
                "latency_seconds": latency,
                "tokens": tokens,
                "model": model,
                "temperature": temperature,
                "prompt_hash": prompt_hash,
            }
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            completed.add(anchor["source_id"])
    corpus_files = (
        "anchors.jsonl",
        "dev_anchors.jsonl",
        "held_out_pool.jsonl",
        "unrelated.jsonl",
        "test_queries.jsonl",
        "cache_entries.jsonl",
    )
    corpus_manifest = {
        "dataset": corpus["source"]["repository"],
        "revision": corpus["source"]["revision"],
        "seed": corpus["seed"],
        "counts": {
            path: len(_read_jsonl(processed / path)) for path in corpus_files
        },
        "sha256": {
            path: hashlib.sha256((processed / path).read_bytes()).hexdigest()
            for path in corpus_files
        },
    }
    (processed / "final_corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs" / "corpus.yaml"
    )
    args = parser.parse_args()
    count = build_cache(_PROJECT_ROOT, args.config.resolve())
    print(f"Cache responses stored for {count} anchors.")


if __name__ == "__main__":
    main()

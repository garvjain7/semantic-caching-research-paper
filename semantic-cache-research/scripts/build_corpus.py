"""scripts/build_corpus.py — corpus sampling entrypoint.

Run from the semantic-cache-research/ root:

    python scripts/build_corpus.py [--config configs/corpus.yaml] [--dry-run]

Pipeline
--------
1. Load  : NQ-Open at the pinned revision from configs/corpus.yaml.
2. Filter : filter_candidates (mechanical eligibility checks).
3. Dedupe : deduplicate_records (greedy Jaccard, threshold from config).
4. Sample : seeded split into four disjoint sets:
            anchors (100), dev_anchors (50), held_out_pool (at least 1000), unrelated (100).
5. Write  : each set as JSONL under data/processed/.

Each output row carries: source_id, split, question, gold_answers (list), role.
The unrelated set additionally carries max_anchor_jaccard (float, ≤ 0.2).

Paths are resolved relative to the directory that contains this script's
parent (i.e. semantic-cache-research/), so the script can be invoked from
any working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve paths before any local imports so that running the script from
# any cwd works correctly.
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent  # semantic-cache-research/
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.corpus_config import load_corpus_config, require_stage_ready  # noqa: E402
from src.dataset_generation.nq_open_source import load_nq_open, NQOpenRecord  # noqa: E402
from src.dataset_generation.candidate_filter import (  # noqa: E402
    filter_candidates,
    print_filter_report,
)
from src.dataset_generation.lexical_dedup import deduplicate_records  # noqa: E402
from src.dataset_generation.text_processing import tokenize_question  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_jaccard(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    """Return the Jaccard similarity of two frozen token sets.

    Returns 0.0 when both sets are empty (conservative: avoids treating two
    empty questions as identical, though empty questions are already rejected
    by filter_candidates).
    """
    union_size = len(tokens_a | tokens_b)
    if union_size == 0:
        return 0.0
    return len(tokens_a & tokens_b) / union_size


def _max_jaccard_to_anchors(
    candidate_tokens: frozenset[str],
    anchor_token_sets: list[frozenset[str]],
) -> float:
    """Return the maximum token-Jaccard of a candidate against all anchors.

    Short-circuits to 1.0 as soon as a perfect match is found.
    """
    max_j = 0.0
    for anchor_tokens in anchor_token_sets:
        j = _token_jaccard(candidate_tokens, anchor_tokens)
        if j >= 1.0:
            return 1.0
        if j > max_j:
            max_j = j
    return max_j


def _record_to_dict(record: NQOpenRecord, role: str, **extra) -> dict:
    """Serialize an NQOpenRecord to a plain dict ready for JSON output."""
    return {
        "source_id": record.source_id,
        "split": record.split,
        "question": record.question,
        "gold_answers": list(record.answers),
        "role": role,
        **extra,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write *rows* as newline-delimited JSON to *path* (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _sample_unrelated(
    pool: list[NQOpenRecord],
    anchor_token_sets: list[frozenset[str]],
    n: int,
    rng: random.Random,
    max_jaccard: float,
) -> list[tuple[NQOpenRecord, float]]:
    """Return *n* records whose max token-Jaccard to any anchor is ≤ *max_jaccard*.

    Shuffles *pool* once with *rng* then iterates until *n* qualifying
    records are found or the pool is exhausted.

    Returns a list of (record, max_anchor_jaccard) tuples.
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)
    results: list[tuple[NQOpenRecord, float]] = []
    for record in shuffled:
        candidate_tokens = frozenset(tokenize_question(record.question))
        max_j = _max_jaccard_to_anchors(candidate_tokens, anchor_token_sets)
        if max_j <= max_jaccard:
            results.append((record, max_j))
            if len(results) == n:
                break
    return results


def run(config_path: Path, dry_run: bool) -> None:
    """Execute the full corpus-sampling pipeline."""

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    _log(f"[1/5] Loading config: {config_path}")
    cfg = load_corpus_config(config_path)
    require_stage_ready(cfg, "sampling")

    corpus = cfg["corpus"]
    source = corpus["source"]
    sampling = corpus["sampling"]
    filtering = corpus["filtering"]

    repository: str = source["repository"]
    revision: str = source["revision"]
    splits: tuple[str, ...] = tuple(source["splits"])

    n_anchors: int = sampling["anchors"]
    n_dev: int = sampling["dev_anchors"]
    n_held_out_min: int = sampling["held_out_questions_minimum"]
    n_unrelated: int = sampling["unrelated_queries"]
    seed: int = corpus["seed"]

    min_tokens: int = filtering["minimum_question_tokens"]
    max_tokens: int = filtering["maximum_question_tokens"]
    dominant_frac: float = filtering["dominant_token_max_fraction"]
    dedup_threshold: float = filtering["deduplication_token_jaccard_threshold"]
    unrelated_max_jaccard: float = filtering["unrelated_maximum_token_jaccard_to_any_anchor"]

    _log(
        f"    anchors={n_anchors}, dev={n_dev}, "
        f"held_out_min={n_held_out_min}, unrelated={n_unrelated}, seed={seed}"
    )

    # ------------------------------------------------------------------
    # 2. Load
    # ------------------------------------------------------------------
    _log(f"[2/5] Loading NQ-Open ({repository!r}, revision={revision[:8]}...)")
    all_records = load_nq_open(repository=repository, revision=revision, splits=splits)
    _log(f"    Loaded {len(all_records):,} total records across splits: {list(splits)}")

    # ------------------------------------------------------------------
    # 3. Filter
    # ------------------------------------------------------------------
    _log("[3/5] Filtering candidates...")
    filtered, filter_report = filter_candidates(
        all_records,
        minimum_question_tokens=min_tokens,
        maximum_question_tokens=max_tokens,
        dominant_token_max_fraction=dominant_frac,
    )
    print_filter_report(filter_report)

    # ------------------------------------------------------------------
    # 4. Deduplicate
    # ------------------------------------------------------------------
    _log(f"[4/5] Deduplicating (Jaccard threshold={dedup_threshold})...")
    deduped, dedup_report = deduplicate_records(filtered, threshold=dedup_threshold)
    _log(
        f"    Input: {dedup_report.input_count:,}  "
        f"Retained: {dedup_report.retained_count:,}  "
        f"Dropped: {dedup_report.duplicates_dropped:,}  "
        f"Comparisons: {dedup_report.jaccard_comparisons:,}"
    )

    # ------------------------------------------------------------------
    # 5. Seeded sampling
    # ------------------------------------------------------------------
    _log("[5/5] Seeded sampling...")

    total_needed = n_anchors + n_dev + n_held_out_min + n_unrelated
    if len(deduped) < total_needed:
        raise RuntimeError(
            f"Not enough records after deduplication: need at least {total_needed:,}, "
            f"got {len(deduped):,}. "
            "Reduce corpus size requirements or relax filtering parameters."
        )

    rng = random.Random(seed)

    # Shuffle the deduplicated pool once; all subsequent draws come from
    # this shuffled list to guarantee reproducibility.
    pool = list(deduped)
    rng.shuffle(pool)

    # Draw in order: anchors → dev → held-out (take exactly the minimum) → remainder for unrelated
    anchors = pool[:n_anchors]
    dev_anchors = pool[n_anchors : n_anchors + n_dev]
    held_out_pool = pool[n_anchors + n_dev : n_anchors + n_dev + n_held_out_min]
    unrelated_candidates = pool[n_anchors + n_dev + n_held_out_min :]

    _log(
        f"    Anchors: {len(anchors)}, Dev: {len(dev_anchors)}, "
        f"Held-out pool: {len(held_out_pool)}, Unrelated candidates: {len(unrelated_candidates)}"
    )

    # Build anchor token sets (frozensets) once for Jaccard reuse.
    anchor_token_sets: list[frozenset[str]] = [
        frozenset(tokenize_question(r.question)) for r in anchors
    ]

    # Sample unrelated: shuffle candidates, keep only those with
    # max-anchor-Jaccard ≤ unrelated_max_jaccard.
    _log(
        f"    Sampling {n_unrelated} unrelated queries "
        f"(max anchor Jaccard <= {unrelated_max_jaccard})..."
    )
    unrelated_with_jaccard = _sample_unrelated(
        pool=unrelated_candidates,
        anchor_token_sets=anchor_token_sets,
        n=n_unrelated,
        rng=rng,
        max_jaccard=unrelated_max_jaccard,
    )
    if len(unrelated_with_jaccard) < n_unrelated:
        raise RuntimeError(
            f"Could only find {len(unrelated_with_jaccard)} unrelated queries "
            f"(needed {n_unrelated}) with max anchor Jaccard <= {unrelated_max_jaccard}. "
            "The held-out candidate pool is exhausted. Consider increasing held_out_questions_minimum "
            "or relaxing unrelated_maximum_token_jaccard_to_any_anchor."
        )

    source_ids = {
        "anchors": {record.source_id for record in anchors},
        "dev_anchors": {record.source_id for record in dev_anchors},
        "held_out_pool": {record.source_id for record in held_out_pool},
        "unrelated": {record.source_id for record, _ in unrelated_with_jaccard},
    }
    split_names = tuple(source_ids)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = source_ids[left_name] & source_ids[right_name]
            if overlap:
                raise RuntimeError(
                    f"sampling splits {left_name} and {right_name} overlap: "
                    f"{len(overlap)} source IDs"
                )

    # ------------------------------------------------------------------
    # 6. Write JSONL
    # ------------------------------------------------------------------
    output_dir = _PROJECT_ROOT / "data" / "processed"
    _log(f"Writing JSONL outputs to {output_dir}/")

    files: dict[str, list[dict]] = {
        "anchors": [_record_to_dict(r, role="anchor") for r in anchors],
        "dev_anchors": [_record_to_dict(r, role="dev_anchor") for r in dev_anchors],
        "held_out_pool": [_record_to_dict(r, role="held_out") for r in held_out_pool],
        "unrelated": [
            _record_to_dict(r, role="unrelated", max_anchor_jaccard=round(j, 6))
            for r, j in unrelated_with_jaccard
        ],
    }

    if dry_run:
        _log("[DRY RUN] Skipping writes. Row counts that would be written:")
        for name, rows in files.items():
            _log(f"  {name}.jsonl: {len(rows):,} rows")
        return

    for name, rows in files.items():
        dest = output_dir / f"{name}.jsonl"
        _write_jsonl(dest, rows)
        _log(f"  Wrote {len(rows):,} rows -> {dest.relative_to(_PROJECT_ROOT)}")

    output_hashes = {
        f"{name}.jsonl": hashlib.sha256(
            (output_dir / f"{name}.jsonl").read_bytes()
        ).hexdigest()
        for name in files
    }
    manifest = {
        "dataset": repository,
        "revision": revision,
        "splits": list(splits),
        "seed": seed,
        "filter_report": asdict(filter_report),
        "deduplication_report": asdict(dedup_report),
        "achieved_counts": {name: len(rows) for name, rows in files.items()},
        "unrelated_maximum_token_jaccard": unrelated_max_jaccard,
        "sha256": output_hashes,
    }
    (output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _log("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the NQ-Open corpus: filter, deduplicate, and sample into "
            "anchors / dev_anchors / held_out_pool / unrelated JSONL files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "corpus.yaml",
        help="Path to corpus.yaml (default: configs/corpus.yaml relative to project root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip writing output files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(config_path=args.config.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

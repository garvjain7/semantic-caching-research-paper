"""experiment_runner.py

Three-stage experiment pipeline for the static semantic cache.

Architecture
------------
The cache is STATIC: 100 anchors, each with a pre-computed embedding and an
Ollama response generated once at temperature 0.  No entries are inserted or
evicted during an experiment run.

Stage 1 — Retrieval pass  (run_retrieval_pass)
    For each (embedding backend, match strategy) pair:
        for each query in the query set:
            embed the query once
            find its nearest neighbour in the static cache (no threshold)
            write a StrategyResult to retrieval_records.jsonl

Stage 2 — Fallback pass  (run_fallback_pass)
    For each query:
        call the Ollama fallback LLM once (regardless of strategy or threshold)
        store response text and latency to fallback_records.jsonl
    This is called once per query, not once per threshold or strategy.

Stage 3 — Threshold sweep  (sweep_thresholds)
    Read retrieval_records.jsonl and fallback_records.jsonl.
    For each threshold τ in the configured sweep:
        apply_threshold(result, τ) → hit / miss
        accumulate confusion matrix
        replay stored fallback latency for misses (no Ollama calls)
    Write one row per (strategy, embedding_backend, τ) to
    threshold_sweep_results.jsonl.

The judge verdict for each (configuration, query_id, judge_model) is
persisted to judge_verdicts.jsonl and replayed during the threshold sweep
without re-invoking the judge.

Run all configured retrievals with scripts/run_experiments.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Lazy local imports — keep top-level importable without heavy dependencies
# ---------------------------------------------------------------------------

def _import_match_strategies():
    try:
        from .match_strategies import STRATEGIES, StrategyResult, apply_threshold
    except ImportError:
        # Fallback when src/ is on sys.path directly (e.g. during testing).
        from match_strategies import STRATEGIES, StrategyResult, apply_threshold  # type: ignore[no-redef]
    return STRATEGIES, StrategyResult, apply_threshold


# ---------------------------------------------------------------------------
# Data types for persisted records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FallbackRecord:
    """Ollama fallback call result, stored once per query.

    Persisted to fallback_records.jsonl.  During the threshold sweep, a miss
    at threshold τ replays this stored latency instead of re-calling Ollama.
    """

    query_id: str
    query_text: str
    response: str
    latency_seconds: float
    tokens: int
    model: str            # fallback model id from corpus.yaml

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "response": self.response,
            "latency_seconds": self.latency_seconds,
            "tokens": self.tokens,
            "model": self.model,
        }

    @staticmethod
    def from_dict(d: dict) -> "FallbackRecord":
        return FallbackRecord(
            query_id=d["query_id"],
            query_text=d["query_text"],
            response=d["response"],
            latency_seconds=float(d["latency_seconds"]),
            tokens=int(d["tokens"]),
            model=d["model"],
        )


@dataclass(frozen=True)
class JudgeVerdict:
    """LLM-judge verdict for one (query, nearest-neighbour) pair.

    Persisted to judge_verdicts.jsonl keyed by
    (configuration, query_id, judge_model). Replayed during the threshold
    sweep without re-invoking the judge.
    """

    query_id: str
    neighbour_id: str
    configuration: str
    judge_model: str
    verdict: str          # e.g. "correct_hit" | "false_hit" | "correct_miss"
    raw_output: str       # full judge response text, for audit

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "neighbour_id": self.neighbour_id,
            "configuration": self.configuration,
            "judge_model": self.judge_model,
            "verdict": self.verdict,
            "raw_output": self.raw_output,
        }

    @staticmethod
    def from_dict(d: dict) -> "JudgeVerdict":
        return JudgeVerdict(
            query_id=d["query_id"],
            neighbour_id=d["neighbour_id"],
            configuration=d["configuration"],
            judge_model=d["judge_model"],
            verdict=d["verdict"],
            raw_output=d["raw_output"],
        )


@dataclass(frozen=True)
class ThresholdSweepRow:
    """One row in the threshold sweep output: metrics at one (strategy, backend, τ)."""

    strategy: str
    embedding_backend: str
    threshold: float
    total_queries: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    hit_rate: float
    false_hit_rate: float
    cost_savings_rate: float    # fraction of queries that were cache hits

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "embedding_backend": self.embedding_backend,
            "threshold": self.threshold,
            "total_queries": self.total_queries,
            "confusion_matrix": {
                "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "tn": self.tn,
            },
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hit_rate": self.hit_rate,
            "false_hit_rate": self.false_hit_rate,
            "cost_savings_rate": self.cost_savings_rate,
        }


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Stage 1 — Retrieval pass
# ---------------------------------------------------------------------------

def run_retrieval_pass(
    queries: list[dict],
    embedder,
    strategy_name: str,
    cache_index,
    judge_model: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Find the nearest neighbour for every query; judge each pair once.

    For each query in *queries*:
        1. Embed the query using *embedder* (one embedding call per query).
        2. Call strategy_fn(query_text, embedding, cache_index) → StrategyResult.
           No threshold is applied here.
        3. Call the judge once per unique (query_id, neighbour_id, judge_model)
           pair.  If a verdict for this triple already exists in
           judge_verdicts.jsonl, it is reused without calling the judge again.
        4. Append StrategyResult to retrieval_records.jsonl.
        5. Append JudgeVerdict to judge_verdicts.jsonl (if newly computed).

    Parameters
    ----------
    queries : list[dict]
        Each entry must have keys: "query_id" (str), "question" (str),
        "role" (str), "gold_answers" (list[str]).
    embedder : Embedder
        Exposes embed(text: str) -> np.ndarray (L2-normalised).
    strategy_name : str
        Key into STRATEGIES registry.
    cache_index : CacheIndex
        Static cache pre-loaded with the 100 anchors.
    judge_model : str
        Model id for the LLM judge (from corpus.yaml corpus.models.judge.id).
    output_dir : Path
        Directory to write retrieval_records.jsonl and judge_verdicts.jsonl.
    progress_callback : callable, optional
        Called as callback(done, total) after each query.
    """
    STRATEGIES, StrategyResult, _ = _import_match_strategies()

    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy_name!r}. "
            f"Available: {sorted(STRATEGIES)}"
        )
    strategy_fn = STRATEGIES[strategy_name]

    retrieval_path = output_dir / "retrieval_records.jsonl"
    verdict_path   = output_dir / "judge_verdicts.jsonl"

    # Load already-computed verdicts to avoid re-calling the judge.
    existing_verdicts: set[tuple[str, str, str]] = {
        (row["query_id"], row["configuration"], row["judge_model"])
        for row in _read_jsonl(verdict_path)
    }
    stored_results = {
        (row["query_id"], row["strategy"], row["embedding_backend"]):
            StrategyResult.from_dict(row)
        for row in _read_jsonl(retrieval_path)
    }

    total = len(queries)
    for i, query_record in enumerate(queries):
        query_id   = query_record["query_id"]
        query_text = query_record["question"]
        retrieval_key = (
            query_id,
            strategy_name,
            "none" if strategy_name == "lexical_difflib" else cache_index.embedding_backend,
        )
        result = stored_results.get(retrieval_key)
        if result is None:
            query_embedding = embedder.embed(query_text) if embedder is not None else None
            result = strategy_fn(query_text, query_embedding, cache_index)
            result = StrategyResult(
                query_id=query_id,
                neighbour_id=result.neighbour_id,
                similarity=result.similarity,
                strategy=result.strategy,
                embedding_backend=result.embedding_backend,
                anchor_question=result.anchor_question,
                anchor_answer=result.anchor_answer,
            )
            _append_jsonl(retrieval_path, result.as_dict())
            stored_results[retrieval_key] = result

        # --- judge (once per configuration and query) ---
        verdict_key = (query_id, strategy_name, judge_model)
        if verdict_key not in existing_verdicts:
            verdict = _call_judge(
                query_id=query_id,
                neighbour_id=result.neighbour_id,
                configuration=strategy_name,
                query_text=query_text,
                anchor_question=result.anchor_question,
                anchor_answer=result.anchor_answer,
                gold_answers=query_record.get("gold_answers", []),
                judge_model=judge_model,
            )
            _append_jsonl(verdict_path, verdict.as_dict())
            existing_verdicts.add(verdict_key)

        if progress_callback:
            progress_callback(i + 1, total)


# ---------------------------------------------------------------------------
# Stage 2 — Fallback pass (one Ollama call per query)
# ---------------------------------------------------------------------------

def run_fallback_pass(
    queries: list[dict],
    fallback_model: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Call the Ollama fallback LLM once per query; store response and latency.

    The fallback is called unconditionally for every query regardless of
    strategy or threshold, because the latency is what is replayed during
    the threshold sweep for queries that turn out to be misses.

    If fallback_records.jsonl already contains a record for a query_id, that
    query is skipped (idempotent — safe to re-run after a partial failure).

    Parameters
    ----------
    queries : list[dict]
        Same format as run_retrieval_pass.
    fallback_model : str
        Ollama model id (from corpus.yaml corpus.models.fallback.id).
    output_dir : Path
        Directory to write fallback_records.jsonl.
    progress_callback : callable, optional
        Called as callback(done, total) after each query.
    """
    fallback_path = output_dir / "fallback_records.jsonl"

    already_done: set[str] = {
        row["query_id"] for row in _read_jsonl(fallback_path)
    }

    total = len(queries)
    for i, query_record in enumerate(queries):
        query_id   = query_record["query_id"]
        query_text = query_record["question"]

        if query_id in already_done:
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        record = _call_fallback(
            query_id=query_id,
            query_text=query_text,
            fallback_model=fallback_model,
        )
        _append_jsonl(fallback_path, record.as_dict())
        already_done.add(query_id)

        if progress_callback:
            progress_callback(i + 1, total)


# ---------------------------------------------------------------------------
# Stage 3 — Threshold sweep (pure replay, no model calls)
# ---------------------------------------------------------------------------

def sweep_thresholds(
    thresholds: Iterable[float],
    query_ground_truth: dict[str, bool],
    output_dir: Path,
) -> list[ThresholdSweepRow]:
    """Apply every threshold to stored retrieval results; write metrics.

    Reads retrieval_records.jsonl from *output_dir* and applies each τ in
    *thresholds* using apply_threshold().  Reads fallback_records.jsonl for
    cost/latency accounting.  No model is called.

    Parameters
    ----------
    thresholds : Iterable[float]
        Threshold values to evaluate (0.50 through 0.95 in increments of 0.05).
    query_ground_truth : dict[str, bool]
        Maps query_id → should_hit (True for paraphrases, False for
        hard-negatives and unrelated queries).
    output_dir : Path
        Directory that contains retrieval_records.jsonl (written by Stage 1).
        Results are written to threshold_sweep_results.jsonl in the same dir.

    Returns
    -------
    list[ThresholdSweepRow] — one row per (strategy, embedding_backend, τ).
    """
    _, StrategyResult, apply_threshold = _import_match_strategies()

    retrieval_rows = _read_jsonl(output_dir / "retrieval_records.jsonl")
    if not retrieval_rows:
        raise FileNotFoundError(
            f"No retrieval_records.jsonl found in {output_dir}. "
            "Run run_retrieval_pass first."
        )

    # Group results by (strategy, embedding_backend).
    groups: dict[tuple[str, str], list] = {}
    for row in retrieval_rows:
        key = (row["strategy"], row["embedding_backend"])
        groups.setdefault(key, []).append(StrategyResult.from_dict(row))

    sweep_path = output_dir / "threshold_sweep_results.jsonl"
    all_rows: list[ThresholdSweepRow] = []

    for (strategy, backend), results in sorted(groups.items()):
        for tau in sorted(set(thresholds)):
            tp = fp = fn = tn = 0
            for result in results:
                hit        = apply_threshold(result, tau)
                should_hit = query_ground_truth.get(result.query_id, False)
                if hit and should_hit:      tp += 1
                elif hit and not should_hit: fp += 1
                elif not hit and should_hit: fn += 1
                else:                        tn += 1

            total   = tp + fp + fn + tn
            prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1      = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
            hit_rate      = (tp + fp) / total if total > 0 else 0.0
            false_hit_rate = fp / (fp + tn)  if (fp + tn) > 0 else 0.0
            cost_savings   = (tp + fp) / total if total > 0 else 0.0

            sweep_row = ThresholdSweepRow(
                strategy=strategy,
                embedding_backend=backend,
                threshold=round(float(tau), 6),
                total_queries=total,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(prec, 6),
                recall=round(rec, 6),
                f1=round(f1, 6),
                hit_rate=round(hit_rate, 6),
                false_hit_rate=round(false_hit_rate, 6),
                cost_savings_rate=round(cost_savings, 6),
            )
            all_rows.append(sweep_row)

    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in all_rows:
            destination.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")

    return all_rows


# ---------------------------------------------------------------------------
# Internal model calls
# ---------------------------------------------------------------------------

def _call_judge(
    query_id: str,
    neighbour_id: str,
    configuration: str,
    query_text: str,
    anchor_question: str,
    anchor_answer: str,
    gold_answers: list[str],
    judge_model: str,
) -> JudgeVerdict:
    """Call the Ollama judge once for a (query, nearest-neighbour) pair.

    The judge decides whether the anchor's cached answer is acceptable for
    this query.  The verdict is persisted and replayed across all thresholds
    so the judge is called at most once per unique (query_id, neighbour_id,
    judge_model) triple during an experiment run.

    Returns
    -------
    JudgeVerdict — to be appended to judge_verdicts.jsonl by the caller.
    """
    try:
        from .dataset_generation.ollama_generation import generate_json
    except ImportError:
        from dataset_generation.ollama_generation import generate_json

    prompt = (
        "Evaluate whether the cached response answers the incoming question "
        "correctly. Judge meaning and factual adequacy, not wording. Use the "
        "reference answers when present, but assess independently if they are "
        "absent. Return JSON with one key verdict whose value is exactly "
        '"correct_hit" or "false_hit".\n'
        f"Incoming question: {query_text!r}\n"
        f"Matched anchor: {anchor_question!r}\n"
        f"Cached response: {anchor_answer!r}\n"
        f"Reference answers, if available: {gold_answers!r}\n"
    )
    payload, _ = generate_json(judge_model, prompt, temperature=0)
    verdict = payload.get("verdict")
    if verdict not in {"correct_hit", "false_hit"}:
        raise ValueError("judge returned an invalid verdict")
    return JudgeVerdict(
        query_id=query_id,
        neighbour_id=neighbour_id,
        configuration=configuration,
        judge_model=judge_model,
        verdict=verdict,
        raw_output=json.dumps(payload, ensure_ascii=False),
    )


def _call_fallback(
    query_id: str,
    query_text: str,
    fallback_model: str,
) -> FallbackRecord:
    """Call the Ollama fallback LLM once for a query.

    Temperature is fixed at 0 (from corpus.yaml corpus.models.fallback.temperature).
    Latency and token count are stored so they can be replayed during the
    threshold sweep for any τ that produces a miss, without re-calling Ollama.

    Returns
    -------
    FallbackRecord — to be appended to fallback_records.jsonl by the caller.
    """
    try:
        from .dataset_generation.ollama_generation import generate_text
    except ImportError:
        from dataset_generation.ollama_generation import generate_text

    response, latency, tokens, _ = generate_text(
        model=fallback_model,
        prompt=query_text,
        temperature=0,
    )
    return FallbackRecord(
        query_id=query_id,
        query_text=query_text,
        response=response,
        latency_seconds=latency,
        tokens=tokens,
        model=fallback_model,
    )

"""analysis/compute_metrics.py

Compute all reported metrics from stored experiment output files.

Input files (all under an experiment output directory)
------------------------------------------------------
retrieval_records.jsonl   — one row per (query_id, strategy, embedding_backend);
                            written by experiment_runner.run_retrieval_pass.
                            Required fields: query_id, neighbour_id, similarity,
                            strategy, embedding_backend.

judge_verdicts.jsonl      — one row per (query_id, neighbour_id, judge_model);
                            written by experiment_runner.run_retrieval_pass.
                            Required fields: query_id, neighbour_id, judge_model,
                            verdict ("correct_hit" | "false_hit" | "correct_miss").

fallback_records.jsonl    — one row per query_id;
                            written by experiment_runner.run_fallback_pass.
                            Required fields: query_id, latency_seconds, tokens.

query_metadata.jsonl      — one row per query_id produced by build_corpus.py;
                            required fields: query_id, role ("anchor" | "paraphrase" |
                            "hard_negative" | "unrelated"), subtype
                            (P1/P2/P3/P4 | E/N/T/S | "none"), should_hit (bool).

Metrics produced
----------------
All rates carry a Wilson 95% confidence interval (low, point, high).

Primary
  judged_false_hit_rate     judge-incorrect hits / all hits at threshold τ.
                            Numerator: hits where judge verdict == "false_hit".
                            Denominator: all hits (TP + FP by label).
                            This is the primary result because it uses the
                            judge's semantic assessment rather than the corpus
                            label, which is the point of running the judge.

Secondary (label-based)
  query_false_hit_rate      FP / (TP + FP) = 1 − precision.
                            Denominator is all hits, not all queries.
                            Complements judged_false_hit_rate: when the two
                            diverge, the judge disagrees with the corpus label.

  false_miss_rate           FN / (TP + FN), restricted to queries whose role
                            is "paraphrase" (i.e. only queries that *should*
                            be cache hits are in the denominator).  Queries
                            with role "hard_negative" or "unrelated" are
                            excluded, as a miss on them is correct behaviour.

  cost_savings_rate         1 − (LLM calls made / total queries).
                            LLM calls made = misses (FN + TN); cache hits avoid
                            one LLM call each.  Replays stored fallback latency
                            for misses — no Ollama calls.

  mean_saved_latency_ms     Mean fallback latency (ms) per query, weighted by
                            the fraction of queries that are hits at each τ.
                            Gives the expected per-query latency saving.

Per-subtype
  judge_disagreement_by_subtype
                            Rate at which the judge verdict disagrees with the
                            corpus should_hit label, broken down by subtype
                            (P1–P4, E, N, T, S, unrelated).  Reported only for
                            queries that are hits at threshold τ (the only
                            queries for which the judge is called).

  false_hit_rate_by_hard_negative_type
                            query_false_hit_rate restricted to each hard-negative
                            type (E, N, T, S) separately.  Reported with Wilson CI.

Public API
----------
    load_experiment_records(output_dir) -> ExperimentRecords
    compute_metrics(records, threshold, judge_model) -> MetricSet
    compute_metrics_sweep(records, thresholds, judge_model) -> list[MetricSet]
    MetricSet.as_dict() -> dict
    wilson_ci(successes, trials, z=1.96) -> tuple[float, float, float]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Wilson 95 % confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Return (low, point_estimate, high) Wilson score interval.

    When *trials* == 0, returns (0.0, 0.0, 0.0) — the caller must
    check for zero-denominator cases before interpreting the interval.

    Parameters
    ----------
    successes : int   — number of events (e.g. false hits)
    trials    : int   — total observations (e.g. total hits)
    z         : float — standard normal quantile; 1.96 gives 95 % CI

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical Association.
    """
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p_hat = successes / trials
    centre = (p_hat + z * z / (2 * trials)) / (1 + z * z / trials)
    margin = (
        z
        * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
        / (1 + z * z / trials)
    )
    return (
        round(max(0.0, centre - margin), 6),
        round(p_hat, 6),
        round(min(1.0, centre + margin), 6),
    )


# ---------------------------------------------------------------------------
# Confidence interval carrier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateCI:
    """A proportion with its Wilson 95 % confidence interval.

    Attributes
    ----------
    numerator   : int
    denominator : int
    low         : float — lower bound of Wilson 95 % CI
    point       : float — simple proportion (numerator / denominator)
    high        : float — upper bound of Wilson 95 % CI
    """

    numerator: int
    denominator: int
    low: float
    point: float
    high: float

    @staticmethod
    def compute(numerator: int, denominator: int, z: float = 1.96) -> "RateCI":
        low, point, high = wilson_ci(numerator, denominator, z)
        return RateCI(
            numerator=numerator,
            denominator=denominator,
            low=low,
            point=point,
            high=high,
        )

    def as_dict(self) -> dict:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ci_low": self.low,
            "point": self.point,
            "ci_high": self.high,
        }


# ---------------------------------------------------------------------------
# Input record types (thin wrappers around JSONL rows)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryMeta:
    """Corpus-level metadata for one query."""

    query_id: str
    role: str       # "paraphrase" | "hard_negative" | "unrelated" | "anchor"
    subtype: str    # "P1"–"P4" | "E"|"N"|"T"|"S" | "none"
    should_hit: bool

    @staticmethod
    def from_dict(d: dict) -> "QueryMeta":
        return QueryMeta(
            query_id=d["query_id"],
            role=d["role"],
            subtype=d.get("subtype", "none"),
            should_hit=bool(d["should_hit"]),
        )


@dataclass(frozen=True)
class RetrievalRecord:
    """One nearest-neighbour result row from retrieval_records.jsonl."""

    query_id: str
    neighbour_id: str
    similarity: float
    strategy: str
    embedding_backend: str

    @staticmethod
    def from_dict(d: dict) -> "RetrievalRecord":
        return RetrievalRecord(
            query_id=d["query_id"],
            neighbour_id=d["neighbour_id"],
            similarity=float(d["similarity"]),
            strategy=d["strategy"],
            embedding_backend=d["embedding_backend"],
        )


@dataclass(frozen=True)
class JudgeVerdictRecord:
    """One judge verdict row from judge_verdicts.jsonl."""

    query_id: str
    neighbour_id: str
    configuration: str
    judge_model: str
    verdict: str    # "correct_hit" | "false_hit" | "correct_miss"

    @staticmethod
    def from_dict(d: dict) -> "JudgeVerdictRecord":
        return JudgeVerdictRecord(
            query_id=d["query_id"],
            neighbour_id=d["neighbour_id"],
            configuration=d["configuration"],
            judge_model=d["judge_model"],
            verdict=d["verdict"],
        )


@dataclass(frozen=True)
class FallbackRecord:
    """One fallback latency row from fallback_records.jsonl."""

    query_id: str
    latency_seconds: float
    tokens: int

    @staticmethod
    def from_dict(d: dict) -> "FallbackRecord":
        return FallbackRecord(
            query_id=d["query_id"],
            latency_seconds=float(d["latency_seconds"]),
            tokens=int(d["tokens"]),
        )


# ---------------------------------------------------------------------------
# Loaded experiment snapshot
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecords:
    """All persisted records for one experiment run, ready for metric computation.

    Attributes
    ----------
    retrieval   : list[RetrievalRecord]   — all (query, neighbour) pairs
    verdicts    : dict[(query_id, configuration, judge_model), JudgeVerdictRecord]
    fallback    : dict[query_id, FallbackRecord]
    query_meta  : dict[query_id, QueryMeta]
    strategy    : str   — strategy name (from retrieval records)
    embedding_backend : str
    """

    retrieval: list[RetrievalRecord]
    verdicts: dict[tuple[str, str, str], JudgeVerdictRecord]
    fallback: dict[str, FallbackRecord]
    query_meta: dict[str, QueryMeta]
    strategy: str
    embedding_backend: str


def load_experiment_records(
    output_dir: Path | str,
    query_metadata_path: Path | str | None = None,
    configuration: str | None = None,
) -> ExperimentRecords:
    """Load all persisted JSONL files for one experiment run.

    Parameters
    ----------
    output_dir : Path
        Directory written by experiment_runner (contains retrieval_records.jsonl,
        judge_verdicts.jsonl, fallback_records.jsonl).
    query_metadata_path : Path, optional
        Path to query_metadata.jsonl produced by build_corpus.py.  If None,
        defaults to <output_dir>/test_queries.jsonl.
    configuration : str, optional
        Configuration identifier to select from a multi-configuration run.

    Raises
    ------
    FileNotFoundError  if retrieval_records.jsonl is missing.
    ValueError         if retrieval records span more than one (strategy,
                       embedding_backend) pair — callers must load one
                       experiment at a time.
    """
    output_dir = Path(output_dir)
    retrieval_path = output_dir / "retrieval_records.jsonl"
    verdict_path   = output_dir / "judge_verdicts.jsonl"
    fallback_path  = output_dir / "fallback_records.jsonl"
    if query_metadata_path is None:
        query_metadata_path = output_dir / "test_queries.jsonl"
    query_metadata_path = Path(query_metadata_path)

    if not retrieval_path.exists():
        raise FileNotFoundError(
            f"retrieval_records.jsonl not found in {output_dir}. "
            "Run experiment_runner.run_retrieval_pass first."
        )

    retrieval = [
        RetrievalRecord.from_dict(row)
        for row in _read_jsonl(retrieval_path)
    ]
    if not retrieval:
        raise ValueError(f"retrieval_records.jsonl in {output_dir} is empty.")

    configurations = {r.strategy for r in retrieval}
    if configuration is None and len(configurations) > 1:
        raise ValueError(
            f"retrieval records contain multiple configurations: {configurations}; "
            "select one with configuration=..."
        )
    if configuration is not None:
        retrieval = [r for r in retrieval if r.strategy == configuration]
        if not retrieval:
            raise ValueError(f"no retrieval records for configuration {configuration!r}")
    backends = {r.embedding_backend for r in retrieval}
    if len(backends) > 1:
        raise ValueError(
            f"configuration {configuration!r} spans multiple embedding backends: {backends}"
        )

    verdicts: dict[tuple[str, str, str], JudgeVerdictRecord] = {}
    for row in _read_jsonl(verdict_path):
        rec = JudgeVerdictRecord.from_dict(row)
        verdicts[(rec.query_id, rec.configuration, rec.judge_model)] = rec

    fallback: dict[str, FallbackRecord] = {
        rec.query_id: rec
        for rec in (FallbackRecord.from_dict(row) for row in _read_jsonl(fallback_path))
    }

    query_meta: dict[str, QueryMeta] = {}
    if query_metadata_path.exists():
        for row in _read_jsonl(query_metadata_path):
            meta = QueryMeta.from_dict(row)
            query_meta[meta.query_id] = meta

    return ExperimentRecords(
        retrieval=retrieval,
        verdicts=verdicts,
        fallback=fallback,
        query_meta=query_meta,
        strategy=next(iter({r.strategy for r in retrieval})),
        embedding_backend=next(iter(backends)),
    )


# ---------------------------------------------------------------------------
# MetricSet — all metrics for one (strategy, backend, threshold, judge_model)
# ---------------------------------------------------------------------------

PARAPHRASE_SUBTYPES    = ("P1", "P2", "P3", "P4")
HARD_NEGATIVE_SUBTYPES = ("E", "N", "T", "S")
ALL_SUBTYPES           = PARAPHRASE_SUBTYPES + HARD_NEGATIVE_SUBTYPES + ("unrelated",)


@dataclass
class MetricSet:
    """All metrics for one (strategy, embedding_backend, threshold, judge_model).

    Primary metric
    --------------
    judged_false_hit_rate
        judge-incorrect hits / all hits.
        Numerator: hits where judge verdict == "false_hit".
        Denominator: all hits at this threshold.
        This is the primary result because it uses the judge's *semantic*
        assessment — not the corpus label.  A hit is "judge-incorrect" when
        the judge says the cached answer is unacceptable for the incoming query.

    Secondary label-based metrics
    ------------------------------
    query_false_hit_rate
        False-hit queries / all queries. This is the query-level risk under
        the intended corpus labels; judged_false_hit_rate is conditional on a
        retrieval hit and uses the judge's semantic assessment.

    false_miss_rate
        FN / (TP + FN), denominator restricted to paraphrase-role queries only.
        Queries with role "hard_negative" or "unrelated" are excluded because
        a miss on them is correct behaviour, not a false miss.

    cost_savings_rate
        1 − (LLM calls made / total queries).
        Misses (FN + TN) each trigger one LLM fallback call.
        Hits avoid it.

    mean_saved_latency_ms
        Mean per-query latency saved (ms).  For each query, the saving is
        latency_seconds × 1000 if it is a hit, else 0.  Mean over all queries.

    Per-subtype disaggregation
    --------------------------
    judge_disagreement_by_subtype
        dict[subtype, RateCI] — rate of judge-vs-label disagreement at this τ,
        for queries that are hits (the judge only acts on hits).

    false_hit_rate_by_hard_negative_type
        dict[subtype, RateCI] — false-hit queries divided by all queries of
        each hard-negative type.
    """

    strategy: str
    embedding_backend: str
    threshold: float
    judge_model: str
    total_queries: int

    # Counts used to derive secondary metrics
    tp: int          # hit, should_hit=True   (correct hit)
    fp: int          # hit, should_hit=False  (false hit by label)
    fn: int          # miss, should_hit=True  (false miss)
    tn: int          # miss, should_hit=False (correct miss)

    # Primary metric
    judged_false_hit_rate: RateCI   # judge-incorrect / all hits
    judge_label_disagreement_rate: RateCI  # judge-label disagreements / all hits

    # Secondary label-based metrics
    query_false_hit_rate: RateCI    # false-hit queries / all queries
    false_miss_rate: RateCI         # FN / (TP + FN) — paraphrase queries only
    cost_savings_rate: RateCI       # (TP + FP) / total_queries

    # Latency savings (point estimate only; no CI — continuous variable)
    mean_saved_latency_ms: float

    # Per-subtype metrics
    judge_disagreement_by_subtype: dict[str, RateCI]
    query_false_hit_rate_by_subtype: dict[str, RateCI]
    false_miss_rate_by_paraphrase_subtype: dict[str, RateCI]
    false_hit_rate_by_hard_negative_type: dict[str, RateCI]

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "embedding_backend": self.embedding_backend,
            "threshold": self.threshold,
            "judge_model": self.judge_model,
            "total_queries": self.total_queries,
            "confusion_matrix": {
                "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "tn": self.tn,
            },
            "judged_false_hit_rate": self.judged_false_hit_rate.as_dict(),
            "judge_label_disagreement_rate": self.judge_label_disagreement_rate.as_dict(),
            "query_false_hit_rate": self.query_false_hit_rate.as_dict(),
            "false_miss_rate": self.false_miss_rate.as_dict(),
            "cost_savings_rate": self.cost_savings_rate.as_dict(),
            "mean_saved_latency_ms": round(self.mean_saved_latency_ms, 4),
            "judge_disagreement_by_subtype": {
                st: ci.as_dict()
                for st, ci in self.judge_disagreement_by_subtype.items()
            },
            "query_false_hit_rate_by_subtype": {
                st: ci.as_dict()
                for st, ci in self.query_false_hit_rate_by_subtype.items()
            },
            "false_miss_rate_by_paraphrase_subtype": {
                st: ci.as_dict()
                for st, ci in self.false_miss_rate_by_paraphrase_subtype.items()
            },
            "false_hit_rate_by_hard_negative_type": {
                st: ci.as_dict()
                for st, ci in self.false_hit_rate_by_hard_negative_type.items()
            },
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_metrics(
    records: ExperimentRecords,
    threshold: float,
    judge_model: str,
) -> MetricSet:
    """Compute all metrics for one threshold value.

    The judge verdict lookup uses (query_id, configuration, judge_model),
    matching the persisted judge record. Missing verdicts on hit queries fail
    explicitly instead of being treated as correct.

    Parameters
    ----------
    records      : ExperimentRecords  — loaded by load_experiment_records()
    threshold    : float              — similarity threshold to evaluate
    judge_model  : str                — judge model id (from corpus.yaml)

    Returns
    -------
    MetricSet with all metrics and Wilson 95 % CIs.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")

    # --- per-query decisions ---
    tp = fp = fn = tn = 0

    # Paraphrase-restricted counts for false_miss_rate
    paraphrase_tp = 0
    paraphrase_fn = 0

    # Judge-level counts for judged_false_hit_rate
    judge_incorrect_hits = 0
    judge_label_disagreements = 0
    total_hits           = 0

    # Per-subtype counts
    # judge_disagreement: hits only — (judge_incorrect, total_hits_for_subtype)
    subtype_judge_disagree: dict[str, list[int]] = {st: [0, 0] for st in ALL_SUBTYPES}
    subtype_false_hits: dict[str, list[int]] = {st: [0, 0] for st in ALL_SUBTYPES}
    paraphrase_false_misses: dict[str, list[int]] = {
        st: [0, 0] for st in PARAPHRASE_SUBTYPES
    }

    # false hit by hard-negative type: (fp_count, total_hits_for_subtype)
    hn_false_hits: dict[str, list[int]] = {st: [0, 0] for st in HARD_NEGATIVE_SUBTYPES}

    # Latency savings accumulator
    total_saved_latency_ms = 0.0

    for rec in records.retrieval:
        qid        = rec.query_id
        meta = records.query_meta.get(qid)
        if meta is None:
            raise ValueError(f"missing query metadata for {qid!r}")
        should_hit = meta.should_hit
        role = meta.role
        subtype = meta.subtype
        fb = records.fallback.get(qid)
        if fb is None:
            raise ValueError(f"missing fallback record for query {qid!r}")

        hit = rec.similarity >= threshold

        # Confusion matrix
        if hit and should_hit:
            tp += 1
        elif hit and not should_hit:
            fp += 1
        elif not hit and should_hit:
            fn += 1
        else:
            tn += 1

        if subtype in subtype_false_hits:
            subtype_false_hits[subtype][1] += 1
            if hit and not should_hit:
                subtype_false_hits[subtype][0] += 1
        if subtype in paraphrase_false_misses:
            paraphrase_false_misses[subtype][1] += 1
            if not hit:
                paraphrase_false_misses[subtype][0] += 1
        if subtype in hn_false_hits:
            hn_false_hits[subtype][1] += 1
            if hit and not should_hit:
                hn_false_hits[subtype][0] += 1

        # Paraphrase-restricted false-miss counts
        if role == "paraphrase":
            if hit:
                paraphrase_tp += 1
            else:
                paraphrase_fn += 1

        # Judge-based counts (only relevant when the query is a hit at this τ)
        if hit:
            total_hits += 1
            verdict_key = (qid, rec.strategy, judge_model)
            verdict_rec = records.verdicts.get(verdict_key)
            if verdict_rec is None:
                raise ValueError(
                    f"missing judge verdict for query {qid!r}, "
                    f"configuration {rec.strategy!r}, model {judge_model!r}"
                )
            judge_says_false = verdict_rec.verdict == "false_hit"
            if judge_says_false:
                judge_incorrect_hits += 1
            if judge_says_false != (not should_hit):
                judge_label_disagreements += 1

            # Per-subtype judge disagreement
            if subtype in subtype_judge_disagree:
                subtype_judge_disagree[subtype][1] += 1         # total hits for subtype
                if judge_says_false != (not should_hit):
                    # Disagreement: judge says false_hit but label says should_hit,
                    # OR judge says correct_hit but label says should not hit.
                    subtype_judge_disagree[subtype][0] += 1

        # Latency saving: if hit, we saved the fallback latency for this query
        if hit:
            total_saved_latency_ms += fb.latency_seconds * 1000.0

    total = tp + fp + fn + tn

    # Mean saved latency
    mean_saved_latency_ms = total_saved_latency_ms / total if total > 0 else 0.0

    # Build RateCI objects
    judged_fhr  = RateCI.compute(judge_incorrect_hits, total_hits)
    query_fhr   = RateCI.compute(fp, total)
    fmr         = RateCI.compute(paraphrase_fn, paraphrase_tp + paraphrase_fn)
    csr         = RateCI.compute(tp + fp, total)    # hits / total = cost savings

    # Per-subtype judge disagreement
    judge_disagree_ci: dict[str, RateCI] = {
        st: RateCI.compute(counts[0], counts[1])
        for st, counts in subtype_judge_disagree.items()
    }

    subtype_false_hit_ci = {
        st: RateCI.compute(counts[0], counts[1])
        for st, counts in subtype_false_hits.items()
    }
    paraphrase_false_miss_ci = {
        st: RateCI.compute(counts[0], counts[1])
        for st, counts in paraphrase_false_misses.items()
    }

    # False-hit rate per hard-negative type
    hn_fhr_ci: dict[str, RateCI] = {
        st: RateCI.compute(counts[0], counts[1])
        for st, counts in hn_false_hits.items()
    }

    return MetricSet(
        strategy=records.strategy,
        embedding_backend=records.embedding_backend,
        threshold=round(float(threshold), 6),
        judge_model=judge_model,
        total_queries=total,
        tp=tp, fp=fp, fn=fn, tn=tn,
        judged_false_hit_rate=judged_fhr,
        judge_label_disagreement_rate=RateCI.compute(
            judge_label_disagreements, total_hits
        ),
        query_false_hit_rate=query_fhr,
        false_miss_rate=fmr,
        cost_savings_rate=csr,
        mean_saved_latency_ms=round(mean_saved_latency_ms, 4),
        judge_disagreement_by_subtype=judge_disagree_ci,
        query_false_hit_rate_by_subtype=subtype_false_hit_ci,
        false_miss_rate_by_paraphrase_subtype=paraphrase_false_miss_ci,
        false_hit_rate_by_hard_negative_type=hn_fhr_ci,
    )


def compute_metrics_sweep(
    records: ExperimentRecords,
    thresholds: Iterable[float],
    judge_model: str,
) -> list[MetricSet]:
    """Compute metrics for every threshold in *thresholds*.

    Returns one MetricSet per threshold, sorted ascending by threshold value.
    No model calls are made — all computation is a pure replay of stored records.

    Parameters
    ----------
    records    : ExperimentRecords — loaded by load_experiment_records()
    thresholds : Iterable[float]   — thresholds to evaluate
    judge_model: str               — judge model id from corpus.yaml
    """
    return [
        compute_metrics(records, tau, judge_model)
        for tau in sorted(set(thresholds))
    ]


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------

def write_metrics_jsonl(metrics: list[MetricSet], path: Path | str) -> None:
    """Write a list of MetricSet objects as newline-delimited JSON.

    Creates parent directories if needed.  Overwrites any existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for m in metrics:
            fh.write(json.dumps(m.as_dict(), ensure_ascii=False) + "\n")


def load_metrics_jsonl(path: Path | str) -> list[dict]:
    """Read a metrics JSONL file back as plain dicts for plotting."""
    return _read_jsonl(Path(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

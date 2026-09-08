"""
match_strategies.py

Three pluggable match strategies for SemanticCache.lookup().  All three
share a unified call signature so experiment_runner can call any of them
identically:

    result = strategy_fn(query, query_embedding, index, threshold)

Unified signature
-----------------
query          : str          — raw query text (used by cross_encoder, lexical_match)
query_embedding: np.ndarray   — L2-normalised embedding (used by cosine_threshold,
                                and by cross_encoder for the coarse shortlist)
index          : CacheIndex   — live cache index
threshold      : float        — decision threshold (semantics differ per strategy;
                                see each function's docstring)

Return value (all strategies)
------------------------------
{
    "hit":           bool,
    "cached_answer": str | None,
    "similarity":    float,   # score used for the threshold comparison
    "threshold":     float,
    "strategy":      str,     # name of this strategy
}

Functions
---------
cosine_threshold  — baseline; mirrors GPTCache default
cross_encoder     — two-stage: coarse cosine shortlist → cross-encoder re-rank
lexical_match     — BM25 ablation; embedding-free lower bound
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CacheIndex  # avoid circular import at runtime


# ---------------------------------------------------------------------------
# Strategy 1 — Cosine Threshold (baseline)
# ---------------------------------------------------------------------------

def cosine_threshold(
    query: str,
    query_embedding: np.ndarray,
    index: "CacheIndex",
    threshold: float,
) -> dict:
    """
    Nearest-neighbor cosine search → threshold gate.

    Primary strategy for Experiment A (threshold_sweep).

    Parameters
    ----------
    query : str
        Raw query text (unused by this strategy; kept for uniform signature).
    query_embedding : np.ndarray
        Pre-computed, L2-normalised query vector (shape [dim]).
    index : CacheIndex
        The live cache index holding stored (embedding, metadata) pairs.
    threshold : float
        Cosine similarity threshold in [0, 1].
        Hit declared when top-1 similarity >= threshold.
    """
    # top_result, score = index.search(query_embedding, top_k=1)
    # hit = score >= threshold
    # return {
    #     "hit": hit,
    #     "cached_answer": top_result["answer"] if hit else None,
    #     "similarity": float(score),
    #     "threshold": threshold,
    #     "strategy": "cosine_threshold",
    # }
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy 2 — Cross-Encoder Re-ranking
# ---------------------------------------------------------------------------

def cross_encoder(
    query: str,
    query_embedding: np.ndarray,
    index: "CacheIndex",
    threshold: float,
    top_k: int = 5,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    """
    Two-stage retrieval: coarse cosine shortlist → cross-encoder re-rank.

    Stage 1:  retrieve top_k candidates by cosine similarity (fast, uses
              query_embedding).
    Stage 2:  score each (query, candidate_query) pair with a cross-encoder;
              keep the highest-scoring pair.
    Gate:     declare hit if best cross-encoder score >= threshold.

    Note: cross-encoder outputs are logits, not cosine similarities.
    Calibrate this threshold independently from the cosine threshold.

    Parameters
    ----------
    query : str
        Raw query text — used for cross-encoder pair scoring.
    query_embedding : np.ndarray
        Pre-computed embedding — used for the coarse cosine shortlist.
    index : CacheIndex
        The live cache index.
    threshold : float
        Cross-encoder score threshold.
    top_k : int
        Cosine candidates to re-rank (default 5).
    cross_encoder_model : str
        HuggingFace model ID for the CrossEncoder.
    """
    # from sentence_transformers import CrossEncoder
    # ce = CrossEncoder(cross_encoder_model)
    # candidates = index.search(query_embedding, top_k=top_k)
    #   → returns list of dicts: [{"query": str, "answer": str, "embedding": ndarray}, ...]
    # pairs = [(query, c["query"]) for c in candidates]
    # scores = ce.predict(pairs)                # shape [top_k]
    # best_idx = int(np.argmax(scores))
    # best_score = float(scores[best_idx])
    # hit = best_score >= threshold
    # return {
    #     "hit": hit,
    #     "cached_answer": candidates[best_idx]["answer"] if hit else None,
    #     "similarity": best_score,
    #     "threshold": threshold,
    #     "strategy": "cross_encoder",
    # }
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy 3 — Lexical Match (BM25 ablation baseline)
# ---------------------------------------------------------------------------

def lexical_match(
    query: str,
    query_embedding: np.ndarray,
    index: "CacheIndex",
    threshold: float,
) -> dict:
    """
    BM25-style lexical retrieval over cached query texts.

    Does NOT use embeddings — serves as the embedding-free ablation to
    demonstrate what dense retrieval adds (Experiment B context).

    Requires index.get_all_queries() -> list[dict{"query", "answer"}].

    Parameters
    ----------
    query : str
        Raw query string to match against stored queries.
    query_embedding : np.ndarray
        Unused by this strategy; kept for uniform signature.
    index : CacheIndex
        Must expose get_all_queries().
    threshold : float
        Normalised BM25 score threshold in [0, 1].
        Score is normalised as raw_score / max_score_in_corpus.
    """
    # from rank_bm25 import BM25Okapi
    # stored = index.get_all_queries()   # list[{"query": str, "answer": str}]
    # if not stored:
    #     return {"hit": False, "cached_answer": None,
    #             "similarity": 0.0, "threshold": threshold, "strategy": "lexical_match"}
    # tokenised_corpus = [q["query"].lower().split() for q in stored]
    # bm25 = BM25Okapi(tokenised_corpus)
    # scores = bm25.get_scores(query.lower().split())   # shape [len(stored)]
    # best_idx = int(np.argmax(scores))
    # raw_score = float(scores[best_idx])
    # max_score = float(scores.max()) if scores.max() > 0 else 1.0
    # normalised = raw_score / max_score
    # hit = normalised >= threshold
    # return {
    #     "hit": hit,
    #     "cached_answer": stored[best_idx]["answer"] if hit else None,
    #     "similarity": normalised,
    #     "threshold": threshold,
    #     "strategy": "lexical_match",
    # }
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry — lets experiment_runner resolve a strategy by name
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, callable] = {
    "cosine_threshold": cosine_threshold,
    "cross_encoder":    cross_encoder,
    "lexical_match":    lexical_match,
}

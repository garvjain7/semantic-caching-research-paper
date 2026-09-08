"""
cache.py

Core semantic cache orchestrator: given a query, delegates to one of the
match strategies in match_strategies.py to decide hit vs. miss, then
forwards misses to llm_fallback.py.

Deliberately combines the vector index and hit/miss decision logic in
ONE module rather than splitting them — see PROJECT_STATE.md Decisions
Log (2026-07-15, architecture revision).

Contains:
    - CacheIndex: wraps a vector index (FAISS or plain cosine) over
      cached query embeddings.
    - SemanticCache: exposes lookup(strategy_fn) and insert().
"""

import numpy as np
# import faiss  # TODO: decide FAISS vs plain in-process cosine for v1


class CacheIndex:
    def __init__(self, dim: int):
        self.dim = dim
        # TODO: initialize FAISS index, or a plain numpy matrix of
        # embeddings + parallel list of metadata dicts

    def add(self, embedding: np.ndarray, metadata: dict) -> None:
        """Insert an embedding with its associated (query, answer) metadata."""
        # TODO: implement
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 1):
        """Return the nearest neighbor(s) and similarity score(s)."""
        # TODO: implement
        raise NotImplementedError

    def get_all_queries(self) -> list[dict]:
        """
        Return all stored entries as a list of dicts with at least
        'query' (str) and 'answer' (str) keys.

        Required by match_strategies.lexical_match for BM25 scoring.
        """
        # TODO: implement
        raise NotImplementedError


class SemanticCache:
    def __init__(self, embedder, index: CacheIndex, threshold: float = 0.8):
        self.embedder = embedder
        self.index = index
        self.threshold = threshold

    def lookup(self, query: str, strategy_fn=None) -> dict:
        """
        Embed the query, then delegate to strategy_fn (or fall back to
        cosine_threshold) to decide hit vs. miss.

        All strategies share the unified signature:
            fn(query, query_embedding, index, threshold) -> dict
        where:
            query          : str          — raw text (used by cross_encoder, lexical_match)
            query_embedding: np.ndarray   — L2-normalised vector (used by cosine_threshold
                                            and cross_encoder coarse shortlist)
            index          : CacheIndex
            threshold      : float

        Returns a dict compatible with RequestLogger.log():
            {"hit": bool, "cached_answer": str | None,
             "similarity": float, "threshold": float, "strategy": str}
        """
        # TODO: implement
        # from .match_strategies import cosine_threshold
        # embedding = self.embedder.embed(query)
        # fn = strategy_fn or cosine_threshold
        # result = fn(query, embedding, self.index, self.threshold)
        # return result
        raise NotImplementedError

    def insert(self, query: str, answer: str) -> None:
        """Embed the query and add it to the index with its answer."""
        # TODO: implement
        raise NotImplementedError

"""
cache.py

Core semantic cache logic: given a query embedding, find the nearest
cached entry, apply the similarity threshold, and decide hit vs miss.

Deliberately combines the vector index and the hit/miss decision logic
in ONE module rather than splitting them into separate files — see
PROJECT_STATE.md Decisions Log (2026-07-15, architecture revision) for
why: for a solo research prototype at this scale, splitting them would
be separation without justification.

Contains:
    - CacheIndex: wraps a vector index (FAISS or plain cosine) over
      cached query embeddings.
    - SemanticCache: exposes lookup() and insert().
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


class SemanticCache:
    def __init__(self, embedder, index: CacheIndex, threshold: float = 0.8):
        self.embedder = embedder
        self.index = index
        self.threshold = threshold

    def lookup(self, query: str) -> dict:
        """
        Embed the query, search the index, apply the threshold.

        Should return enough detail for RequestLogger to log a full
        record in one call, e.g.:
            {"hit": bool, "cached_answer": str | None,
             "similarity": float, "threshold": float}
        """
        # TODO: implement
        raise NotImplementedError

    def insert(self, query: str, answer: str) -> None:
        """Embed the query and add it to the index with its answer."""
        # TODO: implement
        raise NotImplementedError

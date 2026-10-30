"""Static FAISS-backed semantic cache used by the experiment runner."""

from __future__ import annotations

from typing import Any

import numpy as np


class CacheIndex:
    """Exact inner-product index over normalized anchor embeddings."""

    def __init__(self, dim: int, embedding_backend: str = ""):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("The cache index requires the 'faiss-cpu' package") from exc
        self.dim = dim
        self.embedding_backend = embedding_backend
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: list[dict[str, Any]] = []

    def add(self, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        vector = self._normalize(embedding)
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        required = {"source_id", "question", "response"}
        if required - metadata.keys():
            raise ValueError(f"metadata must contain {sorted(required)}")
        self._index.add(vector.reshape(1, -1))
        self._metadata.append(dict(metadata))

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._metadata:
            return []
        vector = self._normalize(query_embedding)
        scores, indices = self._index.search(
            vector.reshape(1, -1), min(top_k, len(self._metadata))
        )
        hits = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            hits.append({**self._metadata[int(index)], "similarity": float(score)})
        return hits

    def get_all_queries(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._metadata]

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"embedding dimension {vector.size} != index dimension {self.dim}")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("embedding must have a finite, non-zero norm")
        return np.ascontiguousarray((vector / norm).astype(np.float32))


class SemanticCache:
    """Thin lookup wrapper; thresholding occurs only after nearest retrieval."""

    def __init__(self, embedder, index: CacheIndex, threshold: float = 0.8):
        self.embedder = embedder
        self.index = index
        self.threshold = threshold

    def lookup(self, query: str, strategy_fn=None) -> dict[str, Any]:
        from .match_strategies import cosine_threshold_nn

        embedding = self.embedder.embed(query)
        fn = strategy_fn or cosine_threshold_nn
        result = fn(query, embedding, self.index)
        return {
            "hit": result.similarity >= self.threshold,
            "cached_answer": result.anchor_answer if result.similarity >= self.threshold else None,
            "similarity": result.similarity,
            "threshold": self.threshold,
            "strategy": result.strategy,
            "neighbour_id": result.neighbour_id,
        }

"""Threshold-independent nearest-neighbour strategies for cache experiments."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .cache import CacheIndex


@dataclass(frozen=True)
class StrategyResult:
    query_id: str
    neighbour_id: str
    similarity: float
    strategy: str
    embedding_backend: str
    anchor_question: str
    anchor_answer: str

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "neighbour_id": self.neighbour_id,
            "similarity": self.similarity,
            "strategy": self.strategy,
            "embedding_backend": self.embedding_backend,
            "anchor_question": self.anchor_question,
            "anchor_answer": self.anchor_answer,
        }

    @staticmethod
    def from_dict(row: dict) -> "StrategyResult":
        return StrategyResult(
            query_id=row["query_id"],
            neighbour_id=row["neighbour_id"],
            similarity=float(row["similarity"]),
            strategy=row["strategy"],
            embedding_backend=row["embedding_backend"],
            anchor_question=row["anchor_question"],
            anchor_answer=row["anchor_answer"],
        )


def apply_threshold(result: StrategyResult, threshold: float) -> bool:
    """Apply a threshold to an already-persisted nearest-neighbour result."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return result.similarity >= threshold


def _cosine_result(
    query_embedding: np.ndarray,
    index: "CacheIndex",
    config_name: str,
) -> StrategyResult:
    neighbors = index.search(query_embedding, top_k=1)
    if not neighbors:
        raise ValueError("cannot retrieve from an empty cache index")
    return _result(neighbors[0], config_name, index.embedding_backend)


def minilm_cosine_nn(query: str, query_embedding: np.ndarray, index: "CacheIndex") -> StrategyResult:
    return _cosine_result(query_embedding, index, "minilm_cosine")


def bge_cosine_nn(query: str, query_embedding: np.ndarray, index: "CacheIndex") -> StrategyResult:
    return _cosine_result(query_embedding, index, "bge_cosine")


def cosine_threshold_nn(query: str, query_embedding: np.ndarray, index: "CacheIndex") -> StrategyResult:
    """Compatibility alias for the MiniLM cosine configuration."""
    return _cosine_result(query_embedding, index, "minilm_cosine")


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Reranking requires 'sentence-transformers'") from exc
    return CrossEncoder(model_name)


def cross_encoder_nn(
    query: str,
    query_embedding: np.ndarray,
    index: "CacheIndex",
    top_k: int = 5,
    cross_encoder_model: str = "cross-encoder/quora-distilroberta-base",
) -> StrategyResult:
    """Retrieve five MiniLM candidates and rerank them with a cross-encoder."""
    if index.embedding_backend and "MiniLM" not in index.embedding_backend and "minilm" not in index.embedding_backend.lower():
        raise ValueError("cross-encoder configuration requires a MiniLM cache index")
    candidates = index.search(query_embedding, top_k=top_k)
    if not candidates:
        raise ValueError("cannot rerank an empty cache index")
    model = _load_cross_encoder(cross_encoder_model)
    raw = np.asarray(
        model.predict([(query, candidate["question"]) for candidate in candidates]),
        dtype=np.float64,
    )
    if raw.ndim == 2 and raw.shape[1] == 2:
        shifted = raw - raw.max(axis=1, keepdims=True)
        scores = np.exp(shifted)
        scores = scores[:, 1] / scores.sum(axis=1)
    else:
        scores = raw.reshape(-1)
        if np.any((scores < 0) | (scores > 1)):
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
    best = candidates[int(np.argmax(scores))]
    return _result(
        best,
        "minilm_crossencoder_top5",
        index.embedding_backend,
        float(np.max(scores)),
    )


def lexical_difflib_nn(
    query: str, query_embedding: np.ndarray | None, index: "CacheIndex"
) -> StrategyResult:
    """Select the anchor with the highest difflib.SequenceMatcher ratio."""
    entries = index.get_all_queries()
    if not entries:
        raise ValueError("cannot retrieve from an empty cache index")
    best = max(
        entries,
        key=lambda item: SequenceMatcher(None, query.casefold(), item["question"].casefold()).ratio(),
    )
    score = SequenceMatcher(None, query.casefold(), best["question"].casefold()).ratio()
    return _result(best, "lexical_difflib", "none", score)


def _result(
    item: dict,
    config_name: str,
    backend: str,
    similarity: float | None = None,
) -> StrategyResult:
    return StrategyResult(
        query_id="",
        neighbour_id=item["source_id"],
        similarity=float(item["similarity"] if similarity is None else similarity),
        strategy=config_name,
        embedding_backend=backend,
        anchor_question=item["question"],
        anchor_answer=item["response"],
    )


StrategyFn = Callable[[str, np.ndarray | None, "CacheIndex"], StrategyResult]
STRATEGIES: dict[str, StrategyFn] = {
    "minilm_cosine": minilm_cosine_nn,
    "bge_cosine": bge_cosine_nn,
    "minilm_crossencoder_top5": cross_encoder_nn,
    "lexical_difflib": lexical_difflib_nn,
}

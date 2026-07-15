"""
embedding.py

Wraps the embedding model(s) used to convert queries into vectors for
similarity lookup. Must support swapping the underlying model, since
Experiment B (embedding_comparison) requires running the full pipeline
with at least two different embedding models under identical
conditions.

Baseline model: sentence-transformers/all-MiniLM-L6-v2
(matches GPTCache's default embedding function, giving a fair
comparison point).

Contains:
    - Embedder: load a model by name, embed a single query or a batch,
      return normalized vectors ready for cosine similarity comparison.
"""

import numpy as np
# from sentence_transformers import SentenceTransformer  # TODO: uncomment once installed


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None  # TODO: self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a single query."""
        # TODO: implement
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embedding — needed for efficient dataset generation."""
        # TODO: implement
        raise NotImplementedError

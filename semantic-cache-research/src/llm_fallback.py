"""
llm_fallback.py

Wraps a local LLM (via Ollama) used ONLY to measure real cost/latency
avoided when a cache hit occurs. This is NOT used to determine
correctness — cached answers are the dataset's own gold answers, not
LLM-generated ones (see PROJECT_STATE.md Decisions Log, 2026-07-15).

Contains:
    - OllamaClient: sends a prompt to a fixed local model at
      temperature 0, returns the answer plus measured latency and
      token count.
"""

import time
# TODO: confirm the Ollama Python client / API call pattern


class OllamaClient:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def generate(self, prompt: str) -> dict:
        """
        Call the local model, measure wall-clock latency.

        Should return something like:
            {"answer": str, "latency_seconds": float, "tokens": int}
        """
        # TODO: implement
        raise NotImplementedError

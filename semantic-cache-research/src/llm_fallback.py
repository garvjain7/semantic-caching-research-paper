"""
llm_fallback.py

Wraps the configured local fallback LLM (via Ollama). Its response and
latency are cached once per exact prompt for replay; static cache responses
are generated separately for anchors.

Baseline cost model
-------------------
Ollama is called AT MOST ONCE per unique query string across the entire
experiment, regardless of how many threshold values or strategies are
being swept.  The first call result is stored in `baseline_cache` (a
plain dict keyed by query text).  Subsequent lookups for the same query
return the cached latency/token figures without re-calling Ollama.

This means: when experiment_runner replays the dataset across multiple
threshold values (Experiment A), the LLM cost is measured once per
query and then reused — not re-incurred per threshold.

No external database or persistence module is used; baseline_cache is
in-process memory only.  Results are written to the JSONL log alongside
each request record so they are recoverable from the log file.

Contains:
    - OllamaClient: generate(prompt) with built-in baseline deduplication.
"""

try:
    from .dataset_generation.ollama_generation import generate_text
except ImportError:
    from dataset_generation.ollama_generation import generate_text


class OllamaClient:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        # Keyed by query text.  Value: {"answer": str, "latency_seconds": float, "tokens": int}
        # Populated on first miss; reused on all subsequent calls for the same query.
        self.baseline_cache: dict[str, dict] = {}

    def generate(self, prompt: str) -> dict:
        """
        Return the LLM response for `prompt`, calling Ollama only if this
        exact prompt has not been seen before in this experiment session.

        Returns
        -------
        dict with keys:
            "answer"          : str    — model output text
            "latency_seconds" : float  — wall-clock time of the Ollama call
                                         (0.0 if served from baseline_cache)
            "tokens"          : int    — number of output tokens
                                         (0 if served from baseline_cache)
            "from_baseline_cache": bool — True when result was reused, not re-called
        """
        if prompt in self.baseline_cache:
            cached = dict(self.baseline_cache[prompt])
            cached["from_baseline_cache"] = True
            return cached

        result = self._call_ollama(prompt)
        self.baseline_cache[prompt] = result
        return dict(result) | {"from_baseline_cache": False}

    def _call_ollama(self, prompt: str) -> dict:
        """
        Make one real Ollama call.  Returns the raw result dict (without
        the from_baseline_cache flag) so it can be stored in baseline_cache.
        """
        answer, latency, tokens, _ = generate_text(
            model=self.model_name,
            prompt=prompt,
            temperature=0,
        )
        return {
            "answer": answer,
            "latency_seconds": latency,
            "tokens": tokens,
        }

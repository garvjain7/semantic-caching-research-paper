"""Small Ollama JSON-generation helpers shared by corpus generators."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.request import Request, urlopen


def generate_json(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    response_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Call Ollama once and decode its JSON response.

    Importing the optional Ollama client lazily keeps non-generation stages
    usable without the client installed.
    """
    if not model or not model.strip():
        raise ValueError("model must be a non-empty Ollama model id")
    response_format: str | dict[str, Any] = response_schema or "json"
    options = {"temperature": temperature, "num_predict": 256}
    try:
        import ollama
    except ImportError:
        # Keep local generation usable in minimal Python environments without
        # requiring the optional SDK; this is the same Ollama generate API.
        request = Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps({
                "model": model,
                "prompt": prompt,
                "format": response_format,
                "stream": False,
                "options": options,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        raw = response_payload.get("response", "")
    else:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            format=response_format,
            stream=False,
            options=options,
        )
        raw = (
            response.get("response", "")
            if isinstance(response, dict)
            else getattr(response, "response", "")
        )
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Ollama returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ollama JSON response must be an object")
    return payload, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_text(
    model: str, prompt: str, temperature: float = 0.0
) -> tuple[str, float, int, str]:
    """Call Ollama and return response text, latency, output tokens, prompt hash."""
    if not model or not model.strip():
        raise ValueError("model must be a non-empty Ollama model id")
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Ollama inference requires the 'ollama' Python package") from exc

    started = time.perf_counter()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={"temperature": temperature, "num_predict": 128},
    )
    latency = time.perf_counter() - started
    if isinstance(response, dict):
        answer = response.get("response", "")
        tokens = response.get("eval_count", 0)
    else:
        answer = getattr(response, "response", "")
        tokens = getattr(response, "eval_count", 0)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned an empty text response")
    return (
        answer.strip(),
        latency,
        int(tokens or 0),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def require_text_list(values: list[str] | tuple[str, ...], name: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return [value.strip() for value in values]


def require_answer_list(values: list[str] | tuple[str, ...]) -> list[str]:
    """Validate but preserve an NQ-Open answer list, including blank entries."""
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) for value in values
    ) or not any(value.strip() for value in values):
        raise ValueError("answers must contain at least one non-empty string")
    return list(values)

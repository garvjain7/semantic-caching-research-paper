"""
logger.py

Shared per-request logging schema, used by every experiment. Every
cache lookup — hit or miss — must be logged with enough detail to
recompute any metric later without rerunning the experiment (see
PROJECT_STATE.md, Architecture section).

Logged fields: query, embedding (or its id), retrieved entry,
similarity score, threshold used, hit/miss decision, ground-truth
label, latency, token count.

Contains:
    - RequestLogger: appends one structured record per request to
      results/<experiment_name>/raw_logs/ (JSONL recommended — one
      JSON object per line, easy to append and to stream-read later).
"""

import json
from pathlib import Path


class RequestLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        """Append one record as a JSON line."""
        # TODO: implement (open in append mode, json.dumps + newline)
        raise NotImplementedError

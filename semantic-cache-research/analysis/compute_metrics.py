"""
compute_metrics.py

Reads raw per-request logs from results/<experiment_name>/raw_logs/
and computes hit rate, false-hit rate, true-hit precision, and
cost-savings — with confidence intervals (bootstrap or binomial;
decide once real sample sizes are known, see PROJECT_STATE.md
Questions Needing Research).

Contains:
    - load_logs(experiment_name) -> list[dict]
    - compute_rates(logs) -> dict[str, tuple[float, float, float]]
      (metric -> (point_estimate, ci_low, ci_high))
"""

import json
from pathlib import Path


def load_logs(experiment_name: str) -> list[dict]:
    """Read JSONL logs from results/<experiment_name>/raw_logs/."""
    # TODO: implement
    raise NotImplementedError


def compute_rates(logs: list[dict]) -> dict:
    """Compute hit rate, false-hit rate, precision, cost-savings, and
    confidence intervals from a list of logged request records."""
    # TODO: implement
    raise NotImplementedError

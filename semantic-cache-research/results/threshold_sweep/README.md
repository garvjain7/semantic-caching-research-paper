# threshold_sweep/

Primary experiment output. The saved run is complete: all four configurations
have retrieval records, judge verdicts, fallback records, and threshold metrics.
Re-running the entry point reuses persisted records where available, but may call
Ollama if any required records are missing:

    python scripts/run_experiments.py

The current artifacts include one retrieval per configuration/query, one judge
verdict per configuration/query, one fallback response and latency per query,
and `metrics_<configuration>.jsonl` with ten replayed thresholds and Wilson
intervals. These are the primary 700-query results. See the root
`PROJECT_STATE.md` for a concise summary.

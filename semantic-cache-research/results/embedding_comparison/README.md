# embedding_comparison/

Generated output only — nothing here is hand-written. Safe to delete
and regenerate by running:

    python -m src.experiment_runner --config configs/embedding_comparison.yaml

Expected contents once populated:
    - raw_logs/    per-request JSONL logs (see src/logger.py schema)
    - metrics.csv  computed hit rate / false-hit rate / precision /
                    cost-savings, with confidence intervals
                    (see analysis/compute_metrics.py)

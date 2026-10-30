# Embedding Comparison Status

This output directory is empty. The separate `embedding_comparison` experiment configuration is retained, but the main completed runner already evaluates both embedding models as part of the four-configuration threshold sweep. Those results are in `../threshold_sweep/metrics_minilm_cosine.jsonl` and `metrics_bge_cosine.jsonl`; there is no separate embedding-comparison run or output set here.

The old `src.experiment_runner --config ...` command is not a valid reproduction command. Use `python scripts/run_experiments.py` from the repository root to run/reuse the main sweep; it evaluates all configured matchers.

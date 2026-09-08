"""
experiment_runner.py

Single reusable entrypoint for running any experiment. Takes a config
(from configs/) describing the experiment — which variable to sweep,
fixed parameters, dataset path — and produces logs + metrics under
results/<experiment_name>/.

One runner, three configs (threshold_sweep, embedding_comparison,
cache_scale) — not three separate scripts. Keeps the experiment logic
in one place so all three experiments stay methodologically consistent
(same logging schema, same lookup path) by construction.

Usage (planned):
    python -m src.experiment_runner --config configs/threshold_sweep.yaml
    python -m src.experiment_runner --config configs/threshold_sweep.yaml \
        --match-strategy cross_encoder

Contains:
    - load_config(path) -> dict
    - run_experiment(config, match_strategy, progress_callback) -> None

Available match strategies (see src/match_strategies.py):
    "cosine_threshold"  — baseline; used for Experiment A sweep
    "cross_encoder"     — two-stage re-ranking; Experiment B comparison
    "lexical_match"     — BM25 ablation; embedding-free lower bound
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Optional
# import yaml  # TODO: uncomment once installed

# from .match_strategies import STRATEGIES  # TODO: uncomment once implemented


def load_config(path: str) -> dict:
    """Load and validate a YAML experiment config."""
    # TODO: implement
    raise NotImplementedError


def run_experiment(
    config: dict,
    match_strategy: str = "cosine_threshold",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Wire together Embedder, SemanticCache, OllamaClient, and
    RequestLogger per the config, run the sweep/comparison/scale
    experiment, write results to results/<experiment_name>/.

    Parameters
    ----------
    config : dict
        Loaded and validated experiment config (from load_config).
    match_strategy : str
        One of the keys in match_strategies.STRATEGIES.  Defaults to
        "cosine_threshold" which mirrors GPTCache behaviour and is the
        baseline for Experiment A.
    progress_callback : callable, optional
        Called as callback(queries_done: int, total: int) after each
        query — used by the API layer to update /status without polling
        the filesystem.
    """
    # TODO: implement
    # strategy_fn = STRATEGIES[match_strategy]
    # embedder  = Embedder(config["embedding_model"])
    # index     = CacheIndex(dim=embedder.model.get_sentence_embedding_dimension())
    # cache     = SemanticCache(embedder, index, threshold=config.get("threshold", 0.8))
    # ollama    = OllamaClient(config["llm_model"])   # baseline_cache lives here
    # logger    = RequestLogger(f"results/{config['experiment_name']}/raw_logs/run.jsonl")
    #
    # dataset = load_dataset(config["dataset_path"])
    #   Each item: {"query": str, "gold_answer": str,
    #               "label": "paraphrase" | "hard_negative" | "unrelated"}
    # total = len(dataset)
    #
    # # Confusion matrix accumulators
    # # Ground truth: paraphrase → should_hit=True
    # #               hard_negative / unrelated → should_hit=False
    # cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    #
    # for i, item in enumerate(dataset):
    #     query, label = item["query"], item["label"]
    #     should_hit = (label == "paraphrase")
    #
    #     # Strategy call — unified signature for all three strategies
    #     embedding = embedder.embed(query)
    #     result    = strategy_fn(query, embedding, cache.index, cache.threshold)
    #
    #     # Baseline LLM cost: called at most once per unique query string.
    #     # OllamaClient.generate() checks its own baseline_cache before calling Ollama.
    #     if not result["hit"]:
    #         llm_resp = ollama.generate(query)          # deduped internally
    #         cache.insert(query, llm_resp["answer"])
    #         result["latency_seconds"] = llm_resp["latency_seconds"]
    #         result["tokens"]          = llm_resp["tokens"]
    #         result["from_baseline_cache"] = llm_resp["from_baseline_cache"]
    #     else:
    #         result["latency_seconds"] = 0.0
    #         result["tokens"]          = 0
    #         result["from_baseline_cache"] = None  # not applicable on cache hit
    #
    #     # Confusion matrix update
    #     if result["hit"] and should_hit:      cm["TP"] += 1
    #     elif result["hit"] and not should_hit: cm["FP"] += 1
    #     elif not result["hit"] and should_hit: cm["FN"] += 1
    #     else:                                  cm["TN"] += 1
    #
    #     result.update({"query": query, "ground_truth_label": label,
    #                    "should_hit": should_hit})
    #     logger.log(result)
    #     if progress_callback:
    #         progress_callback(i + 1, total)
    #
    # # Derived metrics from confusion matrix
    # tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    # precision    = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    # recall       = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    # hit_rate     = (tp + fp) / total
    # false_hit_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # cost_savings   = (tp + tn) / total  # requests not sent to Ollama
    #
    # summary = {
    #     "experiment": config["experiment_name"],
    #     "strategy":   match_strategy,
    #     "threshold":  cache.threshold,
    #     "total_queries": total,
    #     "confusion_matrix": cm,
    #     "precision":     precision,
    #     "recall":        recall,
    #     "hit_rate":      hit_rate,
    #     "false_hit_rate": false_hit_rate,
    #     "cost_savings":   cost_savings,
    # }
    # summary_path = Path(f"results/{config['experiment_name']}/summary.json")
    # summary_path.parent.mkdir(parents=True, exist_ok=True)
    # summary_path.write_text(json.dumps(summary, indent=2))
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a semantic cache experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument(
        "--match-strategy",
        default="cosine_threshold",
        choices=["cosine_threshold", "cross_encoder", "lexical_match"],
        help="Match strategy to use for cache lookups (default: cosine_threshold)",
    )
    args = parser.parse_args()
    # TODO:
    # cfg = load_config(args.config)
    # run_experiment(cfg, match_strategy=args.match_strategy)

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

Contains:
    - load_config(path) -> dict
    - run_experiment(config) -> None
"""

import argparse
# import yaml  # TODO: uncomment once installed


def load_config(path: str) -> dict:
    """Load and validate a YAML experiment config."""
    # TODO: implement
    raise NotImplementedError


def run_experiment(config: dict) -> None:
    """
    Wire together Embedder, SemanticCache, OllamaClient, and
    RequestLogger per the config, run the sweep/comparison/scale
    experiment, write results to results/<experiment_name>/.
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    args = parser.parse_args()
    # TODO: config = load_config(args.config); run_experiment(config)

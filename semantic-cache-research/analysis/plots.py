"""
plots.py

Produces the headline and secondary plots from computed metrics.

Contains:
    - plot_tradeoff_curve(metrics, output_path)
        false-hit rate vs. cost-savings across the threshold sweep
        -> the paper's headline figure (Experiment A)
    - plot_embedding_comparison(metrics_by_model, output_path)
        same curve, overlaid per embedding model (Experiment B)
    - plot_scale_effect(metrics_by_scale, output_path)
        false-hit rate vs. cache size at fixed threshold (Experiment C)
"""

# import matplotlib.pyplot as plt  # TODO: uncomment once installed


def plot_tradeoff_curve(metrics, output_path: str) -> None:
    # TODO: implement
    raise NotImplementedError


def plot_embedding_comparison(metrics_by_model, output_path: str) -> None:
    # TODO: implement
    raise NotImplementedError


def plot_scale_effect(metrics_by_scale, output_path: str) -> None:
    # TODO: implement
    raise NotImplementedError

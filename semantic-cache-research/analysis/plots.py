"""Publication figures generated from persisted metric rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Figure generation requires matplotlib") from exc
    return plt


def _save(figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    _pyplot().close(figure)


def plot_tradeoff_curve(metrics: list[dict[str, Any]], output_path: str | Path) -> None:
    """Figure 1: judged false-hit rate versus cost savings for one config."""
    plt = _pyplot()
    rows = sorted(metrics, key=lambda row: row["threshold"])
    x = [row["cost_savings_rate"]["point"] for row in rows]
    y = [row["judged_false_hit_rate"]["point"] for row in rows]
    low = [point - row["judged_false_hit_rate"]["ci_low"] for point, row in zip(y, rows)]
    high = [row["judged_false_hit_rate"]["ci_high"] - point for point, row in zip(y, rows)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3)
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(f"τ={row['threshold']:.2f}", (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Cost savings", ylabel="Judged false-hit rate", title="Threshold trade-off: MiniLM cosine")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    _save(fig, output_path)


def plot_embedding_comparison(
    metrics_by_model: dict[str, list[dict[str, Any]]], output_path: str | Path
) -> None:
    """Figure 2: threshold trade-off for all four retrieval configurations."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, metrics in metrics_by_model.items():
        rows = sorted(metrics, key=lambda row: row["threshold"])
        ax.plot(
            [row["cost_savings_rate"]["point"] for row in rows],
            [row["judged_false_hit_rate"]["point"] for row in rows],
            marker="o",
            label=name,
        )
    ax.set(xlabel="Cost savings", ylabel="Judged false-hit rate", title="Retrieval configuration comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_path)


def plot_scale_effect(metrics_by_scale, output_path: str | Path) -> None:
    """Plot false-hit rate by cache size for the optional scale experiment."""
    plt = _pyplot()
    rows = sorted(metrics_by_scale, key=lambda row: row["cache_size"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [row["cache_size"] for row in rows],
        [row["false_hit_rate"] for row in rows],
        marker="o",
    )
    ax.set(xlabel="Cache size", ylabel="False-hit rate", title="False-hit rate by cache size")
    ax.grid(alpha=0.25)
    _save(fig, output_path)


def plot_subtype_disagreement(
    metrics: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Figure 3: judge-versus-label disagreement by subtype over thresholds."""
    plt = _pyplot()
    rows = sorted(metrics, key=lambda row: row["threshold"])
    subtypes = tuple(rows[0]["judge_disagreement_by_subtype"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for subtype in subtypes:
        ax.plot(
            [row["threshold"] for row in rows],
            [row["judge_disagreement_by_subtype"][subtype]["point"] for row in rows],
            marker="o",
            label=subtype,
        )
    ax.set(xlabel="Threshold τ", ylabel="Judge-label disagreement among hits", title="Judge disagreement by query subtype")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    _save(fig, output_path)

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


CONFIG_LABELS = {
    "minilm_cosine": "MiniLM cosine",
    "bge_cosine": "BGE cosine",
    "minilm_crossencoder_top5": "MiniLM + cross-encoder",
    "lexical_difflib": "Lexical (difflib)",
}


def plot_jfh_threshold(metrics_by_config, output_path):
    """Judged false-hit rate by threshold, with Wilson 95% intervals."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for name, metrics in metrics_by_config.items():
        rows = sorted(metrics, key=lambda row: row["threshold"])
        x = [r["threshold"] for r in rows]
        y = [r["judged_false_hit_rate"]["point"] for r in rows]
        lo = [v - r["judged_false_hit_rate"]["ci_low"] for v, r in zip(y, rows)]
        hi = [r["judged_false_hit_rate"]["ci_high"] - v for v, r in zip(y, rows)]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=2.5,
                    linewidth=1.7, markersize=4, label=CONFIG_LABELS.get(name, name))
    ax.set(xlabel="Similarity threshold (τ)", ylabel="Judged false-hit rate among cache hits",
           title="Judged false-hit rate across the threshold sweep")
    ax.set_xticks([0.50 + i * 0.05 for i in range(10)])
    ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_path)


def plot_savings_threshold(metrics_by_config, output_path):
    """Cache-call savings by threshold."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for name, metrics in metrics_by_config.items():
        rows = sorted(metrics, key=lambda row: row["threshold"])
        y = [r["cost_savings_rate"]["point"] for r in rows]
        lo = [v - r["cost_savings_rate"]["ci_low"] for v, r in zip(y, rows)]
        hi = [r["cost_savings_rate"]["ci_high"] - v for v, r in zip(y, rows)]
        ax.errorbar([r["threshold"] for r in rows], y, yerr=[lo, hi],
                    marker="o", capsize=2, linewidth=1.7, markersize=4,
                    label=CONFIG_LABELS.get(name, name))
    ax.set(xlabel="Similarity threshold (τ)", ylabel="Model-call savings (cache-hit share)",
           title="Model-call savings across the threshold sweep")
    ax.set_xticks([0.50 + i * 0.05 for i in range(10)])
    ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_path)


def plot_tradeoff_frontier(metrics_by_config, output_path):
    """JFH-cost frontier with threshold labels and Wilson intervals."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for name, metrics in metrics_by_config.items():
        rows = sorted(metrics, key=lambda row: row["threshold"])
        x = [r["cost_savings_rate"]["point"] for r in rows]
        y = [r["judged_false_hit_rate"]["point"] for r in rows]
        lo = [v - r["judged_false_hit_rate"]["ci_low"] for v, r in zip(y, rows)]
        hi = [r["judged_false_hit_rate"]["ci_high"] - v for v, r in zip(y, rows)]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=2,
                    linewidth=1.5, markersize=4, label=CONFIG_LABELS.get(name, name))
    ax.set(xlabel="Model-call savings", ylabel="Judged false-hit rate among cache hits",
           title="Correctness–cost trade-off across tested thresholds")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_path)


def plot_hard_negative_heatmap(metrics_by_config, output_path, threshold=.75):
    """Label-based hard-negative hit rates at one shared threshold."""
    plt = _pyplot()
    configs = list(metrics_by_config)
    types = ("E", "N", "T", "S")
    vals = []
    for config in configs:
        row = min(metrics_by_config[config], key=lambda r: abs(r["threshold"]-threshold))
        vals.append([row["false_hit_rate_by_hard_negative_type"][t]["point"] for t in types])
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    image = ax.imshow(vals, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(types)), labels=types)
    ax.set_yticks(range(len(configs)), labels=[CONFIG_LABELS.get(c,c) for c in configs])
    ax.set_xlabel("Hard-negative type"); ax.set_title(f"Label false-hit rate by hard-negative type (τ={threshold:.2f})")
    for i, row in enumerate(vals):
        for j, value in enumerate(row): ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < .75 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="False-hit queries / queries in type", pad=.03)
    _save(fig, output_path)


def plot_disagreement_subtypes(metrics_by_config, output_path, threshold=.75):
    """Judge/label disagreement among hits; zero-denominator groups are omitted."""
    plt = _pyplot()
    configs = list(metrics_by_config)
    subtypes = ("P1", "P2", "P3", "P4", "E", "N", "T", "S", "unrelated")
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    width = .19
    for ci, config in enumerate(configs):
        row = min(metrics_by_config[config], key=lambda r: abs(r["threshold"]-threshold))
        ys = []
        for st in subtypes:
            item = row["judge_disagreement_by_subtype"][st]
            ys.append(item["point"] if item["denominator"] else float("nan"))
        xs = [i + (ci - (len(configs)-1)/2)*width for i in range(len(subtypes))]
        ax.bar(xs, ys, width=width, label=CONFIG_LABELS.get(config,config))
        for x, st in zip(xs, subtypes):
            item = row["judge_disagreement_by_subtype"][st]
            if item["denominator"]:
                y = item["point"]
                ax.errorbar(x, y, yerr=[[y-item["ci_low"]],[item["ci_high"]-y]],
                            fmt="none", ecolor="black", capsize=1.5, linewidth=.7)
    ax.set_xticks(range(len(subtypes)), labels=subtypes)
    ax.set(xlabel="Query subtype", ylabel="Judge–label disagreement among hits",
           title=f"Judge–label disagreement by query subtype (τ={threshold:.2f})")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2)
    _save(fig, output_path)


def plot_distance_error(records_by_config, metadata, output_path, threshold=.75):
    """Binned label false-hit incidence by lexical distance, at a shared τ."""
    plt = _pyplot()
    bins = [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.000001)]
    labels = ["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1.0"]
    metrics = {}
    for config, records in records_by_config.items():
        totals = [[0,0] for _ in bins]
        for rec in records:
            m = metadata.get(rec["query_id"])
            if not m or m.get("jaccard_to_anchor") is None or m.get("edit_dist_to_anchor") is None: continue
            if float(rec["similarity"]) < threshold: continue
            correct = bool(m.get("should_hit")) and rec["neighbour_id"] == m.get("anchor_id")
            for key, value in (("jaccard", float(m["jaccard_to_anchor"])), ("edit", float(m["edit_dist_to_anchor"]))):
                for i, (lo, hi) in enumerate(bins):
                    if lo <= value < hi:
                        metrics.setdefault((config,key), [[0,0] for _ in bins])[i][1] += 1
                        if not correct: metrics[(config,key)][i][0] += 1
                        break
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, key, title in zip(axes, ("jaccard","edit"), ("Token Jaccard similarity to anchor", "Normalized edit distance to anchor")):
        for ci, config in enumerate(records_by_config):
            arr = metrics.get((config,key), [[0,0] for _ in bins])
            rates = [a/b if b else float("nan") for a,b in arr]
            lows, highs = [], []
            for a, b in arr:
                if b:
                    p = a / b; z = 1.96; den = 1 + z*z/b
                    center = (p + z*z/(2*b))/den
                    half = z*((p*(1-p)/b + z*z/(4*b*b))**.5)/den
                    lows.append(p-max(0,center-half)); highs.append(min(1,center+half)-p)
                else:
                    lows.append(0); highs.append(0)
            xs = [i + (ci - 1.5)*.035 for i in range(len(labels))]
            ax.errorbar(xs, rates, yerr=[lows,highs], marker="o", linewidth=1.4, capsize=2,
                        markersize=4, label=CONFIG_LABELS.get(config,config))
        ax.set_title(title); ax.set_xticks(range(len(labels)),labels); ax.set_ylim(0,1); ax.grid(alpha=.25)
        ax.set_xlabel("Distance bin")
    axes[0].set_ylabel("Label false hits / hits in bin")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Label false-hit rate by lexical covariate among cache hits (τ={threshold:.2f})")
    _save(fig, output_path)
    return metrics

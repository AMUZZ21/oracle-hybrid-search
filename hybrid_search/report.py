"""Turns raw per-query metrics into results.md / results.json / per_query.csv / plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import Dataset
from .pipeline import SystemRun
from .stats import paired_bootstrap

COLUMNS = ["P@1", "P@5", "P@10", "nDCG@5", "nDCG@10", "R@10", "R@100", "MRR@10", "MAP@100"]


def _table(systems: list[SystemRun], metrics: dict[str, dict[str, np.ndarray]]) -> str:
    means = {s.name: {c: float(metrics[s.name][c].mean()) for c in COLUMNS} for s in systems}
    best = {c: max(m[c] for m in means.values()) for c in COLUMNS}
    head = "| System | " + " | ".join(COLUMNS) + " | ms/query |"
    sep = "|---|" + "---:|" * (len(COLUMNS) + 1)
    rows = []
    for s in systems:
        cells = []
        for c in COLUMNS:
            v = f"{means[s.name][c]:.3f}"
            cells.append(f"**{v}**" if abs(means[s.name][c] - best[c]) < 1e-12 else v)
        rows.append(f"| {s.name} | " + " | ".join(cells) + f" | {s.ms_per_query:.1f} |")
    return "\n".join([head, sep, *rows])


def _significance(metrics, pairs, metric="nDCG@10") -> str:
    lines = [f"| Comparison (A vs B) | Δ {metric} (A−B) | 95% CI | p-value |", "|---|---:|---:|---:|"]
    for a, b in pairs:
        if a in metrics and b in metrics:
            r = paired_bootstrap(metrics[a][metric], metrics[b][metric])
            star = " *" if r["p"] < 0.05 else ""
            lines.append(f"| {a} vs {b} | {r['mean_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p']:.4f}{star} |")
    return "\n".join(lines)


def _examples(ds: Dataset, metrics, a: str, b: str, n: int = 3) -> str:
    """Queries where system a gained / lost the most nDCG@10 relative to system b."""
    if a not in metrics or b not in metrics:
        return ""
    d = metrics[a]["nDCG@10"] - metrics[b]["nDCG@10"]
    order = np.argsort(d)
    out = [f"### Biggest gains of `{a}` over `{b}` (nDCG@10)\n"]
    for i in order[::-1][:n]:
        out.append(f"- {metrics[b]['nDCG@10'][i]:.2f} → {metrics[a]['nDCG@10'][i]:.2f}  ·  {ds.query_texts[i]}")
    out.append(f"\n### Biggest losses of `{a}` vs `{b}`\n")
    for i in order[:n]:
        out.append(f"- {metrics[b]['nDCG@10'][i]:.2f} → {metrics[a]['nDCG@10'][i]:.2f}  ·  {ds.query_texts[i]}")
    return "\n".join(out)


def _plots(out: Path, systems, metrics, sweep, alpha) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s.name for s in systems]
    show = ["P@10", "nDCG@10", "R@100"]
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#9D755D"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    width = 0.8 / len(names)
    x = np.arange(len(show))
    for i, n in enumerate(names):
        vals = [metrics[n][m].mean() for m in show]
        bars = ax.bar(x + i * width, vals, width, label=n, color=palette[i % len(palette)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(show)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality by system")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    alphas, vals = zip(*sweep)
    ax.plot(alphas, vals, marker="o", color="#4C78A8", label="linear fusion")
    for n, c in (("bm25", "#999999"), ("dense", "#F58518")):
        if n in metrics:
            ax.axhline(metrics[n]["nDCG@10"].mean(), ls="--", color=c, lw=1, label=n)
    ax.axvline(alpha, color="#cccccc", lw=0.8)
    ax.set_xlabel("dense weight α  (0 = pure BM25, 1 = pure dense)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Linear-fusion weight sweep")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "results.png", dpi=160)
    plt.close(fig)


def write_report(out_dir, ds: Dataset, systems: list[SystemRun], metrics, sweep, config: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alpha = config.get("alpha", 0.5)
    lin = next((s.name for s in systems if s.name.startswith("hybrid_linear")), None)

    # results.json (means + latency + config)
    summary = {
        "config": config,
        "dataset": {"name": ds.name, "n_docs": len(ds.doc_ids), "n_queries": len(ds.query_ids)},
        "systems": {
            s.name: {"metrics": {c: float(metrics[s.name][c].mean()) for c in COLUMNS},
                     "ms_per_query": s.ms_per_query, "stage_ms": s.stages}
            for s in systems
        },
        "alpha_sweep_ndcg10": [{"alpha": a, "ndcg10": v} for a, v in sweep],
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    # per_query.csv
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", *[f"{s.name}:{c}" for s in systems for c in COLUMNS]])
        for i, qid in enumerate(ds.query_ids):
            w.writerow([qid, ds.query_texts[i], *[f"{metrics[s.name][c][i]:.4f}" for s in systems for c in COLUMNS]])

    pairs = [("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense"), ("dense", "bm25"),
             ("hybrid_rrf+rerank", "hybrid_rrf"), ("hybrid_rrf+rerank", "dense"), ("hybrid_rrf+rerank", "bm25")]
    if lin:
        pairs.insert(3, (lin, "hybrid_rrf"))

    md = [
        f"# Results: {ds.name}",
        f"{len(ds.doc_ids):,} documents, {len(ds.query_ids)} queries. "
        f"Dense: `{config['dense_model']}`. Reranker: `{config.get('rerank_model') or 'none'}` "
        f"(rerank depth {config['rerank_depth']}). RRF k={config['rrf_k']}.\n",
        "## Main table (best per column in bold)\n",
        _table(systems, metrics),
        "\nLatency is the cumulative cost of every stage a system needs (hybrid = BM25 + dense + fusion), "
        "measured on this machine; dense query encoding is batched, so it is amortised per query.\n",
        "## Is the difference real? Paired bootstrap on nDCG@10 (10,000 resamples; * = p < 0.05)\n",
        _significance(metrics, pairs),
        "\n## Linear-fusion weight sweep (nDCG@10)\n",
        "| dense weight α | " + " | ".join(f"{a:.1f}" for a, _ in sweep) + " |",
        "|---|" + "---:|" * len(sweep),
        "| nDCG@10 | " + " | ".join(f"{v:.3f}" for _, v in sweep) + " |",
        "\nThe sweep is computed on the evaluation queries, so it shows sensitivity, not a tuned result.\n",
    ]
    if "hybrid_rrf+rerank" in metrics:
        md.append("## Qualitative: where the reranker helps and hurts\n")
        md.append(_examples(ds, metrics, "hybrid_rrf+rerank", "hybrid_rrf"))
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    _plots(out, systems, metrics, sweep, alpha)
    return out / "results.md"

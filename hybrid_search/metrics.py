"""Ranking metrics, written from scratch (binary relevance = grade > 0; nDCG uses graded gains).

P@k     = (# relevant in top k) / k
R@k     = (# relevant in top k) / (# relevant overall)
DCG@k   = sum_{i=1..k} rel_i / log2(i + 1)         (linear gain, same convention as BEIR / pytrec_eval)
nDCG@k  = DCG@k / IDCG@k                            (IDCG = DCG of the ideal ordering)
MRR@k   = 1 / rank of first relevant doc in top k (0 if none)
AP@k    = sum_{i<=k, rel_i} P@i / min(#relevant, k)
"""
from __future__ import annotations

import math

import numpy as np

Qrel = dict[str, int]


def precision_at_k(ranked: list[str], rel: Qrel, k: int) -> float:
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked: list[str], rel: Qrel, k: int) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rel: Qrel, k: int) -> float:
    gains = [float(rel.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in rel.values() if v > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], rel: Qrel, k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranked: list[str], rel: Qrel, k: int) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if rel.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / min(n_rel, k)


def evaluate(
    qids: list[str],
    runs: list[list[str]],
    qrels: dict[str, Qrel],
    p_ks=(1, 5, 10),
    ndcg_ks=(5, 10),
    r_ks=(10, 100),
    mrr_k: int = 10,
    map_k: int = 100,
) -> dict[str, np.ndarray]:
    """Per-query metric arrays (aligned with qids). Means are taken by the caller."""
    out: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        out.setdefault(name, []).append(value)

    for qid, ranked in zip(qids, runs):
        rel = qrels[qid]
        for k in p_ks:
            add(f"P@{k}", precision_at_k(ranked, rel, k))
        for k in ndcg_ks:
            add(f"nDCG@{k}", ndcg_at_k(ranked, rel, k))
        for k in r_ks:
            add(f"R@{k}", recall_at_k(ranked, rel, k))
        add(f"MRR@{mrr_k}", mrr_at_k(ranked, rel, mrr_k))
        add(f"MAP@{map_k}", average_precision_at_k(ranked, rel, map_k))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

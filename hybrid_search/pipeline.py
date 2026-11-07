"""Runs every retrieval system on one dataset and records rankings + latency.

Systems
  bm25             lexical baseline
  dense            bi-encoder baseline
  hybrid_rrf       Reciprocal Rank Fusion of bm25 + dense
  hybrid_linear    min-max weighted fusion (alpha * dense + (1 - alpha) * bm25)
  hybrid_rrf+rerank   cross-encoder re-scores the top `rerank_depth` of hybrid_rrf
  bm25+rerank      (optional ablation) same reranker on BM25 candidates only
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import fusion
from .data import Dataset
from .metrics import evaluate


@dataclass
class SystemRun:
    name: str
    runs: list[np.ndarray]          # per query: doc indices, best first
    ms_per_query: float             # cumulative latency of every stage this system needs
    stages: dict[str, float] = field(default_factory=dict)


def _ids(ds: Dataset, runs: list[np.ndarray]) -> list[list[str]]:
    return [[ds.doc_ids[i] for i in r] for r in runs]


def rerank_runs(reranker, ds: Dataset, runs: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """Re-order the top `depth` of each run by cross-encoder score; keep the tail as-is."""
    cands = [r[:depth] for r in runs]
    texts = [[ds.doc_texts[i] for i in c] for c in cands]
    scores = reranker.score(ds.query_texts, texts)
    out = []
    for r, c, s in zip(runs, cands, scores):
        order = np.argsort(-s, kind="stable")
        out.append(np.concatenate([c[order], r[depth:]]))
    return out


def run_pipeline(
    ds: Dataset,
    bm25,
    dense,
    reranker=None,
    depth: int = 100,
    rerank_depth: int = 50,
    rrf_k: int = 60,
    alpha: float = 0.5,
    sweep_alphas: tuple[float, ...] = tuple(round(x * 0.1, 1) for x in range(11)),
    ablate_rerank: bool = False,
) -> tuple[list[SystemRun], list[tuple[float, float]]]:
    nq = len(ds.query_ids)
    ms = lambda t: 1000.0 * t / nq  # noqa: E731

    t = time.perf_counter(); bm = bm25.search(ds.query_texts, depth); t_bm = ms(time.perf_counter() - t)
    t = time.perf_counter(); dn = dense.search(ds.query_texts, depth); t_dn = ms(time.perf_counter() - t)

    t = time.perf_counter()
    rrf_res = [fusion.rrf([b[0], d[0]], k=rrf_k, top=depth) for b, d in zip(bm, dn)]
    t_rrf = ms(time.perf_counter() - t)

    t = time.perf_counter()
    lin_res = [fusion.linear([b, d], [1 - alpha, alpha], top=depth) for b, d in zip(bm, dn)]
    t_lin = ms(time.perf_counter() - t)

    systems = [
        SystemRun("bm25", [r[0] for r in bm], t_bm, {"bm25": t_bm}),
        SystemRun("dense", [r[0] for r in dn], t_dn, {"dense": t_dn}),
        SystemRun("hybrid_rrf", [r[0] for r in rrf_res], t_bm + t_dn + t_rrf,
                  {"bm25": t_bm, "dense": t_dn, "fusion": t_rrf}),
        SystemRun(f"hybrid_linear(a={alpha})", [r[0] for r in lin_res], t_bm + t_dn + t_lin,
                  {"bm25": t_bm, "dense": t_dn, "fusion": t_lin}),
    ]

    if reranker is not None:
        hybrid = systems[2]
        t = time.perf_counter()
        rr = rerank_runs(reranker, ds, hybrid.runs, rerank_depth)
        t_rr = ms(time.perf_counter() - t)
        systems.append(SystemRun("hybrid_rrf+rerank", rr, hybrid.ms_per_query + t_rr,
                                 {**hybrid.stages, "rerank": t_rr}))
        if ablate_rerank:
            t = time.perf_counter()
            rb = rerank_runs(reranker, ds, systems[0].runs, rerank_depth)
            t_rb = ms(time.perf_counter() - t)
            systems.append(SystemRun("bm25+rerank", rb, t_bm + t_rb, {"bm25": t_bm, "rerank": t_rb}))

    # Linear-fusion weight sweep (nDCG@10 as a function of the dense weight alpha).
    # NOTE: evaluated on the test set, so it is a sensitivity analysis, not a tuned result.
    sweep = []
    for a in sweep_alphas:
        res = [fusion.linear([b, d], [1 - a, a], top=10) for b, d in zip(bm, dn)]
        m = evaluate(ds.query_ids, _ids(ds, [r[0] for r in res]), ds.qrels, p_ks=(), ndcg_ks=(10,),
                     r_ks=(), mrr_k=10, map_k=10)["nDCG@10"].mean()
        sweep.append((a, float(m)))
    return systems, sweep


def score_systems(ds: Dataset, systems: list[SystemRun]) -> dict[str, dict[str, np.ndarray]]:
    return {s.name: evaluate(ds.query_ids, _ids(ds, s.runs), ds.qrels) for s in systems}

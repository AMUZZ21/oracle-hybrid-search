#!/usr/bin/env python
"""Run the full BM25 / dense / hybrid / reranked comparison.

    python run_experiment.py                       # SciFact, default models
    python run_experiment.py --dataset nfcorpus    # graded relevance, good for nDCG
    python run_experiment.py --limit-queries 50    # quick sanity run
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from hybrid_search.bm25 import BM25
from hybrid_search.data import load_beir
from hybrid_search.pipeline import run_pipeline, score_systems
from hybrid_search.report import COLUMNS, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", help="BEIR dataset name (scifact, nfcorpus, fiqa, ...)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--query-prefix", default=None, help="override the query instruction prefix")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--ablate-rerank", action="store_true", help="also rerank BM25-only candidates")
    p.add_argument("--depth", type=int, default=100, help="candidates retrieved per system")
    p.add_argument("--rerank-depth", type=int, default=50)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.5, help="dense weight for linear fusion")
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    ds = load_beir(a.dataset, a.data_dir)
    if a.limit_queries:
        ds = ds.limit_queries(a.limit_queries)
    print(ds)

    print("Indexing BM25 ...")
    bm25 = BM25(a.bm25_k1, a.bm25_b).fit(ds.doc_texts)

    print("Indexing dense embeddings ...")
    from hybrid_search.dense import DenseRetriever
    dense = DenseRetriever(a.dense_model, a.query_prefix, max_seq_length=a.max_seq_length, cache_dir=a.cache_dir)
    dense.index(ds.doc_texts, cache_tag=a.dataset)

    reranker = None
    if not a.no_rerank:
        from hybrid_search.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(a.rerank_model)

    systems, sweep = run_pipeline(
        ds, bm25, dense, reranker, depth=a.depth, rerank_depth=a.rerank_depth,
        rrf_k=a.rrf_k, alpha=a.alpha, ablate_rerank=a.ablate_rerank,
    )
    metrics = score_systems(ds, systems)

    config = {**vars(a), "rerank_model": None if a.no_rerank else a.rerank_model}
    path = write_report(f"{a.out}/{a.dataset}", ds, systems, metrics, sweep, config)

    print("\n" + "=" * 78)
    print(f"{'system':<26}" + "".join(f"{c:>9}" for c in ["P@10", "nDCG@10", "R@100", "MRR@10"]) + f"{'ms/q':>9}")
    for s in systems:
        m = metrics[s.name]
        print(f"{s.name:<26}" + "".join(f"{m[c].mean():>9.3f}" for c in ["P@10", "nDCG@10", "R@100", "MRR@10"])
              + f"{s.ms_per_query:>9.1f}")
    print("=" * 78)
    print(f"Full report: {path}  (also results.json, per_query.csv, results.png)")


if __name__ == "__main__":
    main()

"""End-to-end offline test of the experiment pipeline with stand-in encoders."""
from fakes import FakeDense, FakeReranker, synthetic_dataset

from hybrid_search.bm25 import BM25
from hybrid_search.pipeline import run_pipeline, score_systems
from hybrid_search.report import write_report


def test_full_pipeline_and_report(tmp_path):
    ds = synthetic_dataset()
    bm25 = BM25().fit(ds.doc_texts)
    dense = FakeDense().index(ds.doc_texts, "syn")
    systems, sweep = run_pipeline(ds, bm25, dense, FakeReranker(), depth=50, rerank_depth=20, ablate_rerank=True)
    names = [s.name for s in systems]
    assert names[:3] == ["bm25", "dense", "hybrid_rrf"] and "hybrid_rrf+rerank" in names and "bm25+rerank" in names

    metrics = score_systems(ds, systems)
    for s in systems:
        assert all(len(r) > 0 for r in s.runs)
        assert 0.0 <= metrics[s.name]["nDCG@10"].mean() <= 1.0
    # sanity: on this easy synthetic data lexical retrieval must be far above chance
    assert metrics["bm25"]["nDCG@10"].mean() > 0.5
    # reranker keeps the same candidate set (only reorders the head)
    h = next(s for s in systems if s.name == "hybrid_rrf")
    r = next(s for s in systems if s.name == "hybrid_rrf+rerank")
    assert all(set(a[:20]) == set(b[:20]) for a, b in zip(h.runs, r.runs))
    assert len(sweep) == 11

    cfg = dict(dense_model="fake", rerank_model="fake", rerank_depth=20, rrf_k=60, alpha=0.5)
    path = write_report(tmp_path, ds, systems, metrics, sweep, cfg)
    for f in ("results.md", "results.json", "per_query.csv", "results.png"):
        assert (tmp_path / f).exists()
    assert "hybrid_rrf+rerank" in path.read_text()

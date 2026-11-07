"""API tests: engine + FastAPI routes, using stand-in encoders and the bundled demo corpus."""
import pytest
from fakes import FakeDense, FakeReranker
from fastapi.testclient import TestClient

from hybrid_search.data import load_jsonl_corpus
from hybrid_search.engine import SAMPLE_CORPUS, SearchEngine
from webapp.server import create_app


def make_engine(dense=True, rerank=True):
    ids, titles, texts = load_jsonl_corpus(SAMPLE_CORPUS)
    d = FakeDense().index(texts) if dense else None
    r = FakeReranker() if (rerank and dense) else None
    return SearchEngine(ids, texts, titles, dense=d, reranker=r, corpus_name="demo",
                        examples=["heart attack"], unavailable={} if dense else {"dense": "off for test"})


@pytest.fixture()
def client():
    return TestClient(create_app(make_engine()))


def test_info_lists_all_modes(client):
    j = client.get("/api/info").json()
    assert j["n_docs"] == 36 and j["capabilities"] == {"bm25": True, "dense": True, "rerank": True}
    assert all(m["available"] for m in j["modes"])
    assert j["compare_modes"] == ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid_rrf", "hybrid_linear", "hybrid_rerank"])
def test_every_mode_returns_ranked_results(client, mode):
    r = client.get("/api/search", params={"q": "BRCA1 breast cancer", "mode": mode, "k": 5})
    assert r.status_code == 200
    j = r.json()
    res = j["results"]
    # BM25 only returns documents sharing a term with the query, so it may return fewer than k
    assert len(res) == 5 if mode != "bm25" else 1 <= len(res) <= 5
    assert [x["rank"] for x in res] == list(range(1, len(res) + 1))
    assert res[0]["title"] == "BRCA1 and breast cancer"
    assert not res[0]["text"].startswith("BRCA1 and breast cancer.")  # title stripped for display
    assert j["ms"] >= 0 and j["label"]


def test_rerank_reports_rank_before(client):
    res = client.get("/api/search", params={"q": "heart attack clot", "mode": "hybrid_rerank", "k": 5}).json()["results"]
    assert all(x["rank_before"] is not None for x in res)


def test_hybrid_explains_component_ranks(client):
    res = client.get("/api/search", params={"q": "antibiotic resistance", "mode": "hybrid_rrf"}).json()["results"]
    assert res[0]["bm25_rank"] is not None and res[0]["dense_rank"] is not None


def test_compare_runs_each_method_once(client):
    j = client.get("/api/compare", params={"q": "cholesterol", "k": 6}).json()
    assert [c["mode"] for c in j["columns"]] == ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]
    sizes = {c["mode"]: len(c["results"]) for c in j["columns"]}
    assert sizes["dense"] == sizes["hybrid_rrf"] == sizes["hybrid_rerank"] == 6
    assert 1 <= sizes["bm25"] <= 6  # lexical: only docs that share a term


def test_validation_errors(client):
    assert client.get("/api/search", params={"q": "   "}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400
    assert client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/api/search", params={"q": "x" * 501}).status_code == 422
    assert client.get("/api/search").status_code == 422


def test_no_match_returns_empty_list_not_error(client):
    r = client.get("/api/search", params={"q": "zzzzqqqq", "mode": "bm25"})
    assert r.status_code == 200 and r.json()["results"] == []


def test_bm25_only_engine_degrades_gracefully():
    c = TestClient(create_app(make_engine(dense=False)))
    info = c.get("/api/info").json()
    assert info["capabilities"] == {"bm25": True, "dense": False, "rerank": False}
    assert "dense" in info["unavailable"]
    assert c.get("/api/search", params={"q": "cancer", "mode": "dense"}).status_code == 400
    assert c.get("/api/search", params={"q": "cancer", "mode": "bm25"}).status_code == 200
    cols = c.get("/api/compare", params={"q": "cancer"}).json()["columns"]
    assert [x["mode"] for x in cols] == ["bm25"]


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]

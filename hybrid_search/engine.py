"""Serving layer: one object that answers a query with any retrieval strategy.

It reuses the exact same BM25 / dense / fusion / reranker code as the offline experiment, so what
the web UI shows is what the benchmark measured. Dense search and reranking are optional: if the
models cannot be loaded the engine still serves BM25 and reports why the other modes are off.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from . import fusion
from .bm25 import BM25
from .data import load_beir_corpus, load_jsonl_corpus

MODES = {
    "bm25": "Keyword (BM25)",
    "dense": "Meaning (dense)",
    "hybrid_rrf": "Hybrid (RRF)",
    "hybrid_linear": "Hybrid (weighted)",
    "hybrid_rerank": "Hybrid + rerank",
}
SCORE_LABELS = {
    "bm25": "BM25 score", "dense": "cosine", "hybrid_rrf": "RRF score",
    "hybrid_linear": "fused score", "hybrid_rerank": "reranker score",
}
COMPARE_MODES = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]
SAMPLE_CORPUS = Path(__file__).parent / "sample_corpus.jsonl"

EXAMPLES = {
    "demo": [
        "heart attack treatment",
        "BRCA1",
        "why bacteria stop responding to drugs",
        "how do search engines rank documents",
        "rising seas",
    ],
    "biomedical": [
        "vitamin D deficiency and bone health",
        "BRCA1 mutations and cancer risk",
        "antibiotic resistance in bacteria",
        "statins and cardiovascular risk",
    ],
}


def _derive_title(text: str) -> str:
    first = text.split(". ")[0].strip()
    return first if len(first) <= 90 else first[:87].rstrip() + "..."


def _body(title: str, text: str) -> str:
    """Display text without the 'title. ' prefix that was added for indexing."""
    if title and text.startswith(title):
        rest = text[len(title):].lstrip(" .:-\n")
        return rest or text
    return text


class SearchEngine:
    def __init__(self, doc_ids, doc_texts, doc_titles=None, bm25=None, dense=None, reranker=None,
                 corpus_name="corpus", examples=None, unavailable=None):
        self.doc_ids = list(doc_ids)
        self.texts = list(doc_texts)
        titles = list(doc_titles) if doc_titles else [""] * len(self.texts)
        self.titles = [t or _derive_title(x) for t, x in zip(titles, self.texts)]
        self.bodies = [_body(t, x) for t, x in zip(titles, self.texts)]
        self.bm25 = bm25 if bm25 is not None else BM25().fit(self.texts)
        self.dense = dense
        self.reranker = reranker
        self.corpus_name = corpus_name
        self.examples = list(examples or [])
        self.unavailable = dict(unavailable or {})  # capability -> human-readable reason
        self._lock = threading.Lock()  # torch models are not worth sharing across threads

    # ---- introspection -----------------------------------------------------------------
    def available_modes(self) -> list[str]:
        modes = ["bm25"]
        if self.dense is not None:
            modes += ["dense", "hybrid_rrf", "hybrid_linear"]
            if self.reranker is not None:
                modes.append("hybrid_rerank")
        return modes

    def info(self) -> dict:
        avail = set(self.available_modes())
        return {
            "corpus": self.corpus_name,
            "n_docs": len(self.doc_ids),
            "capabilities": {"bm25": True, "dense": self.dense is not None,
                             "rerank": self.reranker is not None},
            "dense_model": getattr(self.dense, "model_name", None),
            "rerank_model": getattr(self.reranker, "model_name", None),
            "modes": [{"id": m, "label": label, "available": m in avail} for m, label in MODES.items()],
            "compare_modes": [m for m in COMPARE_MODES if m in avail],
            "unavailable": self.unavailable,
            "examples": self.examples,
        }

    # ---- public search API -------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid_rrf", k: int = 10, depth: int = 100,
               alpha: float = 0.5, rrf_k: int = 60, rerank_depth: int = 30) -> dict:
        return self._run(query, [mode], k, depth, alpha, rrf_k, rerank_depth)[0]

    def compare(self, query: str, k: int = 10, depth: int = 100, alpha: float = 0.5,
                rrf_k: int = 60, rerank_depth: int = 30) -> dict:
        modes = [m for m in COMPARE_MODES if m in self.available_modes()]
        cols = self._run(query, modes, k, depth, alpha, rrf_k, rerank_depth)
        return {"query": query, "k": k, "columns": cols}

    # ---- internals ---------------------------------------------------------------------
    def _item(self, rank, d, score, bm_rank, dn_rank, rank_before=None):
        return {
            "rank": rank,
            "doc_id": self.doc_ids[d],
            "title": self.titles[d],
            "text": self.bodies[d][:2500],
            "score": None if score is None or not np.isfinite(score) else round(float(score), 4),
            "bm25_rank": bm_rank.get(d),
            "dense_rank": dn_rank.get(d) if dn_rank is not None else None,
            "rank_before": rank_before,
        }

    def _run(self, query, modes, k, depth, alpha, rrf_k, rerank_depth) -> list[dict]:
        avail = self.available_modes()
        for m in modes:
            if m not in MODES:
                raise ValueError(f"Unknown method '{m}'.")
            if m not in avail:
                raise ValueError(f"'{MODES[m]}' is not available on this server.")
        depth = max(depth, k, rerank_depth)

        def lap(t0):
            return 1000.0 * (time.perf_counter() - t0)

        with self._lock:
            t = time.perf_counter(); bm = self.bm25.search([query], depth)[0]; t_bm = lap(t)
            dn, t_dn = None, 0.0
            if any(m != "bm25" for m in modes):
                t = time.perf_counter(); dn = self.dense.search([query], depth)[0]; t_dn = lap(t)
            bm_rank = {int(d): i + 1 for i, d in enumerate(bm[0])}
            dn_rank = {int(d): i + 1 for i, d in enumerate(dn[0])} if dn is not None else None

            rrf_res, t_rrf = None, 0.0
            if "hybrid_rrf" in modes or "hybrid_rerank" in modes:
                t = time.perf_counter(); rrf_res = fusion.rrf([bm[0], dn[0]], k=rrf_k, top=depth); t_rrf = lap(t)

            out = []
            for m in modes:
                stages = {}
                before = None
                if m == "bm25":
                    idx, sc = bm; stages = {"bm25": t_bm}
                elif m == "dense":
                    idx, sc = dn; stages = {"dense": t_dn}
                elif m == "hybrid_rrf":
                    idx, sc = rrf_res; stages = {"bm25": t_bm, "dense": t_dn, "fusion": t_rrf}
                elif m == "hybrid_linear":
                    t = time.perf_counter()
                    idx, sc = fusion.linear([bm, dn], [1 - alpha, alpha], top=depth)
                    stages = {"bm25": t_bm, "dense": t_dn, "fusion": lap(t)}
                else:  # hybrid_rerank
                    r_idx = rrf_res[0]
                    cand = r_idx[:rerank_depth]
                    before = {int(d): i + 1 for i, d in enumerate(r_idx)}
                    t = time.perf_counter()
                    scores = self.reranker.score([query], [[self.texts[int(i)] for i in cand]])[0] if len(cand) else np.array([])
                    order = np.argsort(-scores, kind="stable")
                    idx = np.concatenate([cand[order], r_idx[rerank_depth:]]).astype(np.int64)
                    sc = np.concatenate([scores[order], np.full(len(r_idx) - len(cand), np.nan)])
                    stages = {"bm25": t_bm, "dense": t_dn, "fusion": t_rrf, "rerank": lap(t)}
                results = [
                    self._item(rank, int(d), s, bm_rank, dn_rank,
                               before.get(int(d)) if before else None)
                    for rank, (d, s) in enumerate(zip(idx[:k], sc[:k]), start=1)
                ]
                out.append({
                    "mode": m, "label": MODES[m], "score_label": SCORE_LABELS[m],
                    "ms": round(sum(stages.values()), 1),
                    "stages": {a: round(b, 1) for a, b in stages.items()},
                    "results": results,
                })
        return out


def build_engine(corpus: str = "demo", data_dir: str = "data", cache_dir: str = ".cache",
                 dense_model: str = "BAAI/bge-small-en-v1.5",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_dense: bool = True, use_rerank: bool = True, max_seq_length: int = 256,
                 log=print) -> SearchEngine:
    """corpus: 'demo' (bundled), a BEIR dataset name (e.g. 'scifact'), or a path to a .jsonl file."""
    if corpus == "demo":
        ids, titles, texts = load_jsonl_corpus(SAMPLE_CORPUS)
        name, examples = "demo", EXAMPLES["demo"]
    elif Path(corpus).is_file():
        ids, titles, texts = load_jsonl_corpus(corpus)
        name, examples = Path(corpus).stem, []
    else:
        ids, titles, texts = load_beir_corpus(corpus, data_dir)
        name = corpus
        examples = EXAMPLES["biomedical"] if corpus in ("scifact", "nfcorpus") else []
    log(f"Corpus '{name}': {len(ids):,} documents")

    log("Building BM25 index ...")
    bm25 = BM25().fit(texts)
    tag = f"{name}-" + hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:10]

    dense, reranker, unavailable = None, None, {}
    if use_dense:
        try:
            from .dense import DenseRetriever
            log(f"Loading dense model {dense_model} ...")
            dense = DenseRetriever(dense_model, max_seq_length=max_seq_length, cache_dir=cache_dir)
            dense.index(texts, cache_tag=tag)
        except Exception as e:  # missing package, offline, bad model name...
            dense = None
            unavailable["dense"] = f"Dense search is off: {type(e).__name__}: {e}"
            log(unavailable["dense"])
    else:
        unavailable["dense"] = "Dense search was turned off when the server started."

    if dense is not None and use_rerank:
        try:
            from .rerank import CrossEncoderReranker
            log(f"Loading reranker {rerank_model} ...")
            reranker = CrossEncoderReranker(rerank_model, show_progress=False)
        except Exception as e:
            reranker = None
            unavailable["rerank"] = f"Reranking is off: {type(e).__name__}: {e}"
            log(unavailable["rerank"])
    elif dense is None:
        unavailable["rerank"] = "Reranking needs the hybrid pipeline, which needs dense search."
    else:
        unavailable["rerank"] = "Reranking was turned off when the server started."

    return SearchEngine(ids, texts, titles, bm25, dense, reranker, name, examples, unavailable)

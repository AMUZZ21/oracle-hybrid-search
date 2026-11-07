"""BM25 (Okapi) implemented on a sparse term-document matrix.

score(q, d) = sum_{t in q} IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene-style, never negative)

All per-document weights are precomputed at index time, so a query is just a sum of sparse rows.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .tokenize import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W: sp.csr_matrix | None = None
        self.n_docs = 0

    def fit(self, texts: list[str]) -> "BM25":
        n = len(texts)
        self.n_docs = n
        self.vocab = {}
        rows, cols, vals = [], [], []
        doc_len = np.zeros(n, dtype=np.float64)

        for j, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len[j] = len(tokens)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                i = self.vocab.setdefault(tok, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)

        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(self.vocab), n), dtype=np.float64)
        df = np.diff(tf.indptr)  # number of docs containing each term
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = max(doc_len.mean(), 1e-9)

        coo = tf.tocoo()
        denom = coo.data + self.k1 * (1.0 - self.b + self.b * doc_len[coo.col] / avgdl)
        weights = idf[coo.row] * coo.data * (self.k1 + 1.0) / denom
        self.W = sp.csr_matrix((weights, (coo.row, coo.col)), shape=coo.shape, dtype=np.float32)
        return self

    def score(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.W[ids].sum(axis=0)).ravel()

    def search(self, queries: list[str], k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (doc_indices, scores) per query, best first. Zero-score docs are dropped."""
        out = []
        for q in queries:
            s = self.score(q)
            kk = min(k, len(s))
            idx = np.argpartition(-s, kk - 1)[:kk]
            idx = idx[np.argsort(-s[idx], kind="stable")]
            keep = s[idx] > 0
            out.append((idx[keep], s[idx][keep]))
        return out

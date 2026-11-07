"""Rank fusion for combining lexical and dense result lists.

RRF (Cormack et al. 2009):  score(d) = sum_over_lists 1 / (k + rank_list(d)),  rank starting at 1.
  Uses only ranks, so it needs no score calibration between BM25 (unbounded) and cosine ([-1, 1]).

Linear: min-max normalise each list's scores to [0, 1] over its retrieved candidates, then
  score(d) = sum_i w_i * norm_i(d); a doc missing from a list contributes 0 for that list.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

Result = tuple[np.ndarray, np.ndarray]  # (doc_indices, scores), sorted best-first


def _finish(scores: dict[int, float], top: int) -> Result:
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]  # deterministic ties
    idx = np.array([d for d, _ in items], dtype=np.int64)
    sc = np.array([s for _, s in items], dtype=np.float64)
    return idx, sc


def rrf(rankings: list[np.ndarray], k: int = 60, top: int = 100) -> Result:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[int(d)] += 1.0 / (k + rank)
    return _finish(scores, top)


def minmax(s: np.ndarray) -> np.ndarray:
    if len(s) == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.ones_like(s, dtype=np.float64)


def linear(results: list[Result], weights: list[float], top: int = 100) -> Result:
    scores: dict[int, float] = defaultdict(float)
    for (idx, sc), w in zip(results, weights):
        for d, v in zip(idx, minmax(np.asarray(sc, dtype=np.float64))):
            scores[int(d)] += w * float(v)
    return _finish(scores, top)

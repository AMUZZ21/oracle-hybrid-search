import numpy as np
import pytest

from hybrid_search import fusion
from hybrid_search.bm25 import BM25
from hybrid_search.metrics import (average_precision_at_k, mrr_at_k, ndcg_at_k,
                                   precision_at_k, recall_at_k)
from hybrid_search.stats import paired_bootstrap


# ---- metrics: hand-computed example -------------------------------------------------
RANKED = ["d1", "d2", "d3"]
REL = {"d1": 1, "d3": 1}


def test_precision_recall():
    assert precision_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 1) == 1.0
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(0.5)


def test_ndcg_matches_hand_calculation():
    dcg = 1 / np.log2(2) + 1 / np.log2(4)          # = 1.5
    idcg = 1 / np.log2(2) + 1 / np.log2(3)         # = 1.6309
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(dcg / idcg)


def test_ndcg_graded_gain():
    # grade 2 doc at rank 2, grade 1 doc at rank 1 -> worse than ideal order
    rel = {"a": 1, "b": 2}
    val = ndcg_at_k(["a", "b"], rel, 2)
    ideal = ndcg_at_k(["b", "a"], rel, 2)
    assert ideal == pytest.approx(1.0) and val < 1.0


def test_mrr_and_ap():
    assert mrr_at_k(["x", "d1"], REL, 10) == 0.5
    assert mrr_at_k(["x", "y"], REL, 10) == 0.0
    assert average_precision_at_k(RANKED, REL, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_perfect_and_empty():
    assert ndcg_at_k(["d1", "d3"], REL, 10) == pytest.approx(1.0)
    assert ndcg_at_k([], REL, 10) == 0.0


# ---- BM25 ---------------------------------------------------------------------------
DOCS = [
    "the cat sat on the mat",
    "dogs are loyal animals",
    "cats and kittens are playful pets",
    "stock markets fell sharply today",
]


def test_bm25_ranks_matching_docs_first_and_stems():
    bm = BM25().fit(DOCS)
    (idx, sc), = bm.search(["cat"], 4)
    assert set(idx[:2]) == {0, 2}          # "cats" stems to "cat" when nltk is installed
    assert all(sc[i] >= sc[i + 1] for i in range(len(sc) - 1))


def test_bm25_unknown_terms_return_nothing():
    bm = BM25().fit(DOCS)
    (idx, sc), = bm.search(["zebra quantum"], 4)
    assert len(idx) == 0


def test_bm25_rarer_term_scores_higher():
    docs = ["common rare", "common common", "common other"]
    bm = BM25().fit(docs)
    assert bm.score("rare")[0] > bm.score("common")[0]


# ---- fusion -------------------------------------------------------------------------
def test_rrf_rewards_agreement():
    idx, sc = fusion.rrf([np.array([1, 2]), np.array([3, 2])], k=60, top=3)
    assert idx[0] == 2                      # appears in both lists
    assert sc[0] == pytest.approx(1 / 62 + 1 / 62)


def test_rrf_is_deterministic_on_ties():
    a = fusion.rrf([np.array([5, 6]), np.array([6, 5])], top=2)
    b = fusion.rrf([np.array([5, 6]), np.array([6, 5])], top=2)
    assert list(a[0]) == list(b[0]) == [5, 6]


def test_linear_extremes_equal_single_system():
    bm = (np.array([0, 1, 2]), np.array([9.0, 5.0, 1.0]))
    dn = (np.array([2, 1, 0]), np.array([0.9, 0.5, 0.1]))
    assert list(fusion.linear([bm, dn], [1.0, 0.0], top=3)[0]) == [0, 1, 2]
    assert list(fusion.linear([bm, dn], [0.0, 1.0], top=3)[0]) == [2, 1, 0]


def test_minmax_constant_input():
    assert list(fusion.minmax(np.array([2.0, 2.0]))) == [1.0, 1.0]


# ---- stats --------------------------------------------------------------------------
def test_bootstrap_identical_systems():
    x = np.random.default_rng(0).random(50)
    r = paired_bootstrap(x, x.copy())
    assert r["mean_diff"] == 0.0 and r["p"] == 1.0


def test_bootstrap_detects_clear_win():
    x = np.random.default_rng(0).random(100)
    r = paired_bootstrap(x + 0.2, x)
    assert r["p"] < 0.01 and r["ci_low"] > 0

"""Stand-in encoders so the whole stack can be tested without downloading any model."""
import hashlib

import numpy as np

from hybrid_search.data import Dataset


def synthetic_dataset(n_topics=12, docs_per_topic=8, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(400)]
    doc_ids, doc_texts, qids, qtexts, qrels = [], [], [], [], {}
    for t in range(n_topics):
        topic_words = list(rng.choice(vocab, 6, replace=False))
        rel = {}
        for j in range(docs_per_topic):
            words = topic_words + list(rng.choice(vocab, 15))
            rng.shuffle(words)
            did = f"t{t}d{j}"
            doc_ids.append(did)
            doc_texts.append(" ".join(words))
            rel[did] = 1
        qids.append(f"q{t}")
        qtexts.append(" ".join(rng.choice(topic_words, 3, replace=False)))
        qrels[f"q{t}"] = rel
    return Dataset("synthetic", doc_ids, doc_texts, qids, qtexts, qrels)


def _embed(text: str, dim=128) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeDense:
    model_name = "fake-hash-encoder"

    def index(self, texts, cache_tag=None):
        self.emb = np.stack([_embed(t) for t in texts])
        return self

    def search(self, queries, k):
        out = []
        for q in queries:
            s = self.emb @ _embed(q)
            idx = np.argsort(-s, kind="stable")[:k]
            out.append((idx, s[idx]))
        return out


class FakeReranker:
    model_name = "fake-overlap-reranker"

    def score(self, queries, candidates):
        return [np.array([len(set(q.lower().split()) & set(c.lower().split())) for c in cs], dtype=float)
                for q, cs in zip(queries, candidates)]

"""Dense bi-encoder retrieval: embed once, cache to disk, brute-force cosine similarity."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str | None = None,
        batch_size: int = 64,
        max_seq_length: int = 256,
        cache_dir: str | Path = ".cache",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.emb: np.ndarray | None = None

    def index(self, texts: list[str], cache_tag: str) -> "DenseRetriever":
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model_name)
        path = self.cache_dir / f"{slug}__{cache_tag}__{len(texts)}__L{self.max_seq_length}.npy"
        if path.exists():
            self.emb = np.load(path)
            print(f"Loaded cached embeddings: {path}")
            return self
        self.emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, self.emb)
        return self

    def search(self, queries: list[str], k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        q = self.model.encode(
            [self.query_prefix + x for x in queries], batch_size=self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        sims = q @ self.emb.T  # cosine similarity (both sides L2-normalised)
        kk = min(k, sims.shape[1])
        out = []
        for row in sims:
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            out.append((idx, row[idx]))
        return out

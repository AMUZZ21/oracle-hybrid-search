"""Cross-encoder reranker: scores (query, passage) jointly, unlike the bi-encoder."""
from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        show_progress: bool = True,
    ):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, queries: list[str], candidates: list[list[str]]) -> list[np.ndarray]:
        """candidates[i] are passage texts for queries[i]. Returns one score array per query."""
        pairs, sizes = [], []
        for q, cands in zip(queries, candidates):
            pairs.extend([(q, c) for c in cands])
            sizes.append(len(cands))
        flat = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=self.show_progress),
            dtype=np.float64,
        )
        out, pos = [], 0
        for n in sizes:
            out.append(flat[pos:pos + n])
            pos += n
        return out

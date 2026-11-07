"""FastAPI app: JSON API + the single-page UI.

Configuration comes from environment variables so the same image runs anywhere:
  HS_CORPUS        demo | scifact | nfcorpus | /path/to/corpus.jsonl      (default: demo)
  HS_DENSE         1/0   load the dense model                              (default: 1)
  HS_RERANK        1/0   load the cross-encoder                            (default: 1)
  HS_DENSE_MODEL / HS_RERANK_MODEL / HS_DATA_DIR / HS_CACHE_DIR
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from hybrid_search.engine import SearchEngine, build_engine

STATIC = Path(__file__).parent / "static"


def _flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "off")


def engine_from_env() -> SearchEngine:
    return build_engine(
        corpus=os.environ.get("HS_CORPUS", "demo"),
        data_dir=os.environ.get("HS_DATA_DIR", "data"),
        cache_dir=os.environ.get("HS_CACHE_DIR", ".cache"),
        dense_model=os.environ.get("HS_DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=os.environ.get("HS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        use_dense=_flag("HS_DENSE"),
        use_rerank=_flag("HS_RERANK"),
    )


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            app.state.engine = engine_from_env()
        yield

    app = FastAPI(title="Hybrid Search Lab", lifespan=lifespan)
    app.state.engine = engine

    def eng() -> SearchEngine:
        if app.state.engine is None:
            raise HTTPException(503, "The search engine is still starting. Try again in a moment.")
        return app.state.engine

    def clean(q: str) -> str:
        q = " ".join(q.split())
        if not q:
            raise HTTPException(400, "Enter a query to search.")
        return q

    @app.get("/api/info")
    def info():
        return eng().info()

    @app.get("/api/search")
    def search(
        q: str = Query(..., max_length=500),
        mode: str = "hybrid_rrf",
        k: int = Query(10, ge=1, le=50),
        alpha: float = Query(0.5, ge=0.0, le=1.0),
        rrf_k: int = Query(60, ge=1, le=1000),
        rerank_depth: int = Query(30, ge=5, le=200),
    ):
        try:
            return {"query": clean(q), **eng().search(clean(q), mode, k, alpha=alpha, rrf_k=rrf_k,
                                                       rerank_depth=rerank_depth)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/compare")
    def compare(
        q: str = Query(..., max_length=500),
        k: int = Query(10, ge=1, le=50),
        alpha: float = Query(0.5, ge=0.0, le=1.0),
        rrf_k: int = Query(60, ge=1, le=1000),
        rerank_depth: int = Query(30, ge=5, le=200),
    ):
        return eng().compare(clean(q), k, alpha=alpha, rrf_k=rrf_k, rerank_depth=rerank_depth)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app

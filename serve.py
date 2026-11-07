#!/usr/bin/env python
"""Start the web UI.

    python serve.py                      # bundled 36-document demo corpus, opens on :8000
    python serve.py --corpus scifact     # real benchmark corpus (downloads once)
    python serve.py --corpus my.jsonl    # your own documents: {"_id": "...", "title": "...", "text": "..."}
    python serve.py --no-dense           # BM25 only, no model downloads
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="demo", help="demo | BEIR name (scifact, nfcorpus, ...) | path to .jsonl")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on your network")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-dense", action="store_true", help="BM25 only (also disables reranking)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--dense-model", default=None)
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    a = p.parse_args()

    os.environ["HS_CORPUS"] = a.corpus
    os.environ["HS_DENSE"] = "0" if a.no_dense else "1"
    os.environ["HS_RERANK"] = "0" if a.no_rerank else "1"
    for env, val in (("HS_DENSE_MODEL", a.dense_model), ("HS_RERANK_MODEL", a.rerank_model),
                     ("HS_DATA_DIR", a.data_dir), ("HS_CACHE_DIR", a.cache_dir)):
        if val:
            os.environ[env] = val

    import uvicorn
    print(f"\n  Loading models, then open  http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}\n")
    uvicorn.run("webapp.server:create_app", factory=True, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

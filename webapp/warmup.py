"""Download models + dataset and build the embedding cache ahead of time (used by the Dockerfile).

    python -m webapp.warmup
"""
from webapp.server import engine_from_env

if __name__ == "__main__":
    e = engine_from_env()
    print("Ready:", e.info()["capabilities"], f"{e.info()['n_docs']:,} docs")

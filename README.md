# Hybrid Search Lab: BM25 vs Dense vs Hybrid vs Reranked

Two things in one project, built on the same retrieval code:

1. **A web app** to search a corpus with each method and see how they rank documents differently.
2. **A benchmark** that measures them on BEIR (default SciFact): P@k, nDCG@k, Recall, MRR, MAP, paired-bootstrap significance tests and latency.

| System | What it does |
|---|---|
| `bm25` | Okapi BM25 written from scratch on a sparse matrix (Porter stemming, stopwords) |
| `dense` | `BAAI/bge-small-en-v1.5` bi-encoder, cosine similarity, embeddings cached to disk |
| `hybrid_rrf` | Reciprocal Rank Fusion of BM25 + dense (`1/(60+rank)` per list) |
| `hybrid_linear` | Min-max normalised score fusion, `alpha*dense + (1-alpha)*bm25` |
| `hybrid_rrf+rerank` | `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 30-50 of `hybrid_rrf` |

## Part 1. Setup (once)

Python 3.10+ recommended.

```bash
cd hybrid-search
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                # expect: 28 passed (offline, a few seconds)
```

## Part 2. Run the web app

```bash
python serve.py
```

Open **http://localhost:8000**. The first start downloads about 150 MB of models (once) and indexes a bundled 36-document demo corpus in a few seconds.

What you get:

- **One method** view: tabs for Keyword, Meaning, Hybrid (RRF), Hybrid (weighted) and Hybrid + rerank. Each result shows its rank in the keyword and meaning lists, so you can see *why* it landed where it did. In rerank mode it also shows how far the cross-encoder moved it.
- **Compare methods** view: four ranked columns with a line joining the same document across them. Point at a document to follow it; select it for full text and its rank in every method.
- **Settings**: results shown, the weighted-hybrid mix, the RRF constant, the rerank depth.
- Per-stage latency on every search, shareable URLs (`?q=...&view=compare`), `/` to focus the search box, light and dark themes.

Try these on the demo corpus: `heart attack treatment` (meaning finds *myocardial infarction*), `BRCA1` (exact term, keyword wins), `why bacteria stop responding to drugs`.

### Use a bigger or your own corpus

```bash
python serve.py --corpus scifact          # 5,183 scientific abstracts; downloads once, embeds in a few minutes on CPU
python serve.py --corpus nfcorpus
python serve.py --corpus my_docs.jsonl    # one JSON object per line: {"_id": "1", "title": "...", "text": "..."}
```

### Options

| Flag | Effect |
|---|---|
| `--no-dense` | Keyword search only. No model download, starts instantly. |
| `--no-rerank` | Skip the cross-encoder. |
| `--host 0.0.0.0 --port 8080` | Reachable from other devices on your network. |
| `--dense-model`, `--rerank-model` | Swap models (any sentence-transformers bi-encoder / cross-encoder). |

If a model can't load (package missing, offline), the server still starts, keeps the methods that work, and the page says why the others are off.

### API

`GET /api/info`, `GET /api/search?q=...&mode=hybrid_rrf&k=10`, `GET /api/compare?q=...`. Interactive docs are at `/docs`.

## Part 3. Deploy

The app is one process serving the API and the page. Pick one route.

### Option A. Docker, anywhere

```bash
docker build -t hybrid-search .                              # bakes models + embeddings into the image
docker build -t hybrid-search --build-arg CORPUS=scifact .   # or with the real benchmark corpus
docker run -p 7860:7860 hybrid-search                        # http://localhost:7860
```

### Option B. Hugging Face Spaces (free CPU, enough RAM for the models)

1. Create a new Space at huggingface.co/new-space and choose **Docker** as the SDK.
2. Clone the Space, then copy this project's files into it.
3. Add this block at the very top of the Space's `README.md`:
   ```
   ---
   title: Hybrid Search Lab
   sdk: docker
   app_port: 7860
   ---
   ```
4. Commit and push. The Space builds the Dockerfile and serves the app.

### Memory notes

With the neural models loaded, plan for roughly 1.5 GB of RAM, so free hosting tiers with 512 MB will not fit it. On such a host, set `HS_DENSE=0` to run keyword-only, or pick a host with more memory. Configuration is by environment variable: `HS_CORPUS`, `HS_DENSE`, `HS_RERANK`, `HS_DENSE_MODEL`, `HS_RERANK_MODEL`.

## Part 4. Run the benchmark

```bash
python run_experiment.py --limit-queries 50     # quick check first
python run_experiment.py                        # full SciFact run (300 queries)
python run_experiment.py --dataset nfcorpus --ablate-rerank
```

Outputs land in `results/<dataset>/`: `results.md` (main table, significance tests, alpha sweep, where the reranker helped or hurt), `results.png`, `results.json`, `per_query.csv`.

How to read it:

- **nDCG@10** is the headline metric (BEIR convention). **P@k** is the intuitive one.
- **R@100 of the candidate pool** caps what the reranker can achieve: it reorders, it never retrieves.
- With 300 queries, gaps under about 0.01 nDCG are usually noise. Check the p-values.
- Use `--ablate-rerank` to compare reranking BM25-only candidates against the hybrid pool.

## Design notes and caveats

- Defaults (`k=60`, `alpha=0.5`) are fixed, not tuned on the test set. The alpha sweep is a sensitivity analysis on the same queries.
- The reranker is an MS MARCO model applied zero-shot to scientific claims. It may not help on SciFact, and that is itself a finding.
- The web app and the benchmark share `bm25.py`, `dense.py`, `fusion.py` and `rerank.py`, so the UI shows what was measured.
- BM25 only returns documents that share a stemmed term with the query, so its column can be short or empty. That is correct behaviour, not a bug.
- Dense query encoding in the benchmark is batched, so its per-query latency is amortised. The web app's latencies are single-query.

## Layout

```
serve.py                 web app launcher
run_experiment.py        benchmark CLI
webapp/                  FastAPI server + static/index.html (the whole UI, no build step)
hybrid_search/
  engine.py              serving layer: any method for one query
  bm25.py  dense.py  fusion.py  rerank.py
  data.py  metrics.py  stats.py  pipeline.py  report.py
  sample_corpus.jsonl    36-document demo corpus
tests/                   28 tests, all offline (stand-in encoders)
Dockerfile
```

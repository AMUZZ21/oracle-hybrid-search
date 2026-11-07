"""Corpus + benchmark loading (BEIR datasets and plain JSONL files) with no extra dependencies."""
from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]  # qid -> {doc_id: graded relevance}
    doc_titles: list[str] | None = None

    def limit_queries(self, n: int) -> "Dataset":
        return Dataset(
            self.name, self.doc_ids, self.doc_texts,
            self.query_ids[:n], self.query_texts[:n], self.qrels, self.doc_titles,
        )

    def __repr__(self) -> str:
        return f"Dataset({self.name}: {len(self.doc_ids)} docs, {len(self.query_ids)} queries)"


def _compose(title: str, text: str) -> str:
    return f"{title}. {text}" if title else text


def _download(name: str, root: Path) -> Path:
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{name}.zip"
    url = BEIR_URL.format(name=name)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    return target


def load_jsonl_corpus(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """Read a BEIR-style corpus file: one JSON object per line with `_id`, optional `title`, `text`.

    Returns (doc_ids, doc_titles, doc_texts); texts are "title. text", the form used for indexing.
    """
    ids, titles, texts = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            ids.append(str(row["_id"]))
            titles.append(title)
            texts.append(_compose(title, text))
    return ids, titles, texts


def load_beir_corpus(name: str = "scifact", root: str | Path = "data"):
    base = _download(name, Path(root))
    return load_jsonl_corpus(base / "corpus.jsonl")


def load_beir(name: str = "scifact", root: str | Path = "data", split: str = "test") -> Dataset:
    base = _download(name, Path(root))
    doc_ids, doc_titles, doc_texts = load_jsonl_corpus(base / "corpus.jsonl")

    queries = {}
    with open(base / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    qids = [q for q in queries if q in qrels]  # only queries with >=1 relevant doc
    return Dataset(name, doc_ids, doc_texts, qids, [queries[q] for q in qids], qrels, doc_titles)

"""Tokenizer for the lexical retriever: lowercase, alnum split, stopwords, Porter stemming."""
from __future__ import annotations

import re
from functools import lru_cache

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have he in is it its of on that the to was were "
    "will with this these those or not but if than then so such can".split()
)

try:
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    @lru_cache(maxsize=None)
    def _stem(word: str) -> str:
        return _stemmer.stem(word)

except ImportError:  # stemming is optional; BM25 still works without it

    def _stem(word: str) -> str:
        return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]

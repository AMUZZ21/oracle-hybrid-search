"""Paired bootstrap over queries: is system A really better than system B?"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Resample queries with replacement; return mean(a-b), 95% CI, two-sided p-value."""
    assert a.shape == b.shape
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(min(p, 1.0))}

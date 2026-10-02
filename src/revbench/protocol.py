"""Evaluation protocol: a fixed stratified test split and nested training subsamples."""

import numpy as np
from sklearn.model_selection import train_test_split

TEST_FRACTION = 0.3
TEST_CAP = 2000
MIN_POSITIVES = 5
SIZES = (50, 100, 200, 500, 1000, 2000, None)  # None = the full training pool


def split(y, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (pool, test) positional indices: 30% test, capped at 2,000 rows."""
    y = np.asarray(y)
    n_test = min(round(len(y) * TEST_FRACTION), TEST_CAP)
    pool, test = train_test_split(np.arange(len(y)), test_size=n_test, stratify=y, random_state=seed)
    return np.sort(pool), np.sort(test)


def subsample(y, pool: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Stratified size-`n` subset of `pool`, nested across n for a given seed.

    Positives and negatives are shuffled once per seed; a size-n sample takes a prefix of each,
    so the n-sample is contained in the 2n-sample. At least MIN_POSITIVES positives are kept.
    """
    y = np.asarray(y)
    if n is None:
        return np.sort(pool)
    if n > len(pool):
        raise ValueError(f"n={n} exceeds the training pool of {len(pool)} rows")
    rng = np.random.default_rng(seed)
    pos, neg = (rng.permutation(pool[y[pool] == label]) for label in (1, 0))
    n_pos = max(MIN_POSITIVES, round(n * len(pos) / len(pool)))
    if len(pos) < MIN_POSITIVES or n_pos > len(pos) or n - n_pos > len(neg):
        raise ValueError(f"cannot draw {n} rows with {n_pos} positives (pool has {len(pos)} positives)")
    return np.sort(np.concatenate([pos[:n_pos], neg[: n - n_pos]]))

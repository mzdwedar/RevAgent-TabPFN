from itertools import pairwise

import numpy as np
import pytest

from revbench.protocol import SIZES, split, subsample


def labels(n=3000, rate=0.15, seed=0):
    return (np.random.default_rng(seed).random(n) < rate).astype(int)


def test_split_is_deterministic_and_disjoint():
    y = labels()
    a_pool, a_test = split(y, seed=1)
    b_pool, b_test = split(y, seed=1)
    assert np.array_equal(a_pool, b_pool) and np.array_equal(a_test, b_test)
    assert not set(a_pool) & set(a_test)
    assert len(a_pool) + len(a_test) == len(y)


def test_split_differs_across_seeds():
    y = labels()
    assert not np.array_equal(split(y, seed=1)[1], split(y, seed=2)[1])


def test_split_is_30_percent_and_stratified():
    y = labels()
    _, test = split(y, seed=0)
    assert len(test) == 900
    assert y[test].mean() == pytest.approx(y.mean(), abs=0.005)


def test_split_caps_test_set_at_2000():
    y = labels(10_000)
    pool, test = split(y, seed=0)
    assert len(test) == 2000
    assert len(pool) == 8000


def test_subsample_is_deterministic_and_inside_pool():
    y = labels()
    pool, _ = split(y, seed=3)
    a = subsample(y, pool, 200, seed=3)
    assert np.array_equal(a, subsample(y, pool, 200, seed=3))
    assert len(a) == 200 and set(a) <= set(pool)


def test_subsamples_are_nested_across_sizes():
    y = labels()
    pool, _ = split(y, seed=4)
    sizes = [n for n in SIZES if n is not None]
    sets = [set(subsample(y, pool, n, seed=4)) for n in sizes]
    for small, big in pairwise(sets):
        assert small < big


def test_subsample_never_overlaps_test_set():
    y = labels()
    pool, test = split(y, seed=5)
    for n in SIZES:
        assert not set(subsample(y, pool, n, seed=5)) & set(test)


def test_subsample_has_at_least_five_positives_even_when_rare():
    y = labels(3000, rate=0.02)
    pool, _ = split(y, seed=6)
    assert y[subsample(y, pool, 50, seed=6)].sum() >= 5


def test_subsample_is_roughly_stratified():
    y = labels(rate=0.3)
    pool, _ = split(y, seed=7)
    assert y[subsample(y, pool, 1000, seed=7)].mean() == pytest.approx(y[pool].mean(), abs=0.01)


def test_full_is_the_whole_pool():
    y = labels()
    pool, _ = split(y, seed=8)
    assert set(subsample(y, pool, None, seed=8)) == set(pool)


def test_too_large_or_too_few_positives_raises():
    y = labels(300)
    pool, _ = split(y, seed=0)
    with pytest.raises(ValueError, match="pool"):
        subsample(y, pool, 5000, seed=0)
    y_rare = np.zeros(300, dtype=int)
    y_rare[:3] = 1
    pool, _ = split(y_rare, seed=0)
    with pytest.raises(ValueError, match="positives"):
        subsample(y_rare, pool, 100, seed=0)

"""Probabilistic-classification metrics: PR-AUC, ROC-AUC, log-loss, ECE, wall time."""

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15
ECE_BINS = 10


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def log_loss(y, p) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in np.unique(idx):
        mask = idx == b
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def evaluate(y, p) -> dict[str, float]:
    return {"pr_auc": pr_auc(y, p), "roc_auc": roc_auc(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p)}


def timed(fn: Callable, *args, **kwargs):
    """Return (fn(*args, **kwargs), wall-clock seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start

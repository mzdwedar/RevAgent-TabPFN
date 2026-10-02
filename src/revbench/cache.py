"""Disk cache of model predictions, keyed by (dataset, arm, n, seed, config hash)."""

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".cache/predictions")


def config_hash(**config) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


class PredictionCache:
    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)

    def _path(self, dataset: str, arm: str, n, seed: int, config: str) -> Path:
        return self.root / f"{dataset}__{arm}__n{n}__s{seed}__{config}.npz"

    def get(self, dataset, arm, n, seed, config) -> tuple[np.ndarray, float] | None:
        path = self._path(dataset, arm, n, seed, config)
        if not path.exists():
            return None
        with np.load(path) as f:
            return f["proba"], float(f["seconds"])

    def put(self, dataset, arm, n, seed, config, proba, seconds: float) -> None:
        path = self._path(dataset, arm, n, seed, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, proba=np.asarray(proba), seconds=seconds)

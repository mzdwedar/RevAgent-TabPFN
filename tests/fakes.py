"""Offline stand-in for the hosted TabPFN client: records what it was given."""

import numpy as np


class FakeTabPFN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_X = None
        self.predict_X = None

    def fit(self, X, y):
        self.fit_X, self.y = X, np.asarray(y)
        self.rate = float(self.y.mean())
        return self

    def predict_proba(self, X):
        self.predict_X = X
        p = np.full(len(X), self.rate)
        return np.column_stack([1 - p, p])


class FakeTabPFNFactory:
    """Callable like `TabPFNClassifier.create_default_for_version("v3.5", **kw)`."""

    def __init__(self):
        self.instances: list[FakeTabPFN] = []

    def __call__(self, **kwargs):
        self.instances.append(FakeTabPFN(**kwargs))
        return self.instances[-1]

    @property
    def calls(self) -> int:
        return len(self.instances)

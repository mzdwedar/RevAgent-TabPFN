"""Model arms behind one interface: fit on (X_train, y_train), return P(churn) for X_test."""

import time
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from revbench.metrics import log_loss

TABPFN_VERSION = "v3.5"
TUNE_BUDGET_S = 60.0
TUNE_FOLDS = 3

TabPFNFactory = Callable[..., object]


def default_tabpfn_factory(**kwargs):
    """Hosted TabPFN-3.5 via tabpfn-client (needs TABPFN_TOKEN at fit time)."""
    from tabpfn_client import TabPFNClassifier

    return TabPFNClassifier.create_default_for_version(TABPFN_VERSION, **kwargs)


def _string_columns(X: pd.DataFrame) -> list[str]:
    return list(X.select_dtypes(exclude="number").columns)


def _positive_proba(estimator, X) -> np.ndarray:
    return np.asarray(estimator.predict_proba(X))[:, 1]


def _tabpfn(X_train, y_train, X_test, *, tabpfn_factory, seed, **kwargs):
    est = tabpfn_factory(random_state=seed, **kwargs)
    est.fit(X_train, y_train)
    return _positive_proba(est, X_test)


def tabpfn_raw(X_train, y_train, X_test, *, tabpfn_factory, seed=0, **_):
    """Raw columns: strings, NaNs and all go straight to the model."""
    return _tabpfn(X_train, y_train, X_test, tabpfn_factory=tabpfn_factory, seed=seed)


def tabpfn_ordinal(X_train, y_train, X_test, *, tabpfn_factory, seed=0, **_):
    """Ablation: ordinal-encode string columns first (what RevAgent's encode() does)."""
    cols = _string_columns(X_train)
    X_train, X_test = X_train.copy(), X_test.copy()
    if cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cols] = enc.fit_transform(X_train[cols].astype(str))
        X_test[cols] = enc.transform(X_test[cols].astype(str))
    return _tabpfn(X_train, y_train, X_test, tabpfn_factory=tabpfn_factory, seed=seed)


def tabpfn_thinking(
    X_train, y_train, X_test, *, tabpfn_factory, seed=0, thinking_metric="log_loss", **_
):
    return _tabpfn(
        X_train, y_train, X_test, tabpfn_factory=tabpfn_factory, seed=seed,
        thinking_mode=True, thinking_metric=thinking_metric,
    )  # fmt: skip


def _categorical_for_xgb(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """String columns -> pandas category with the train vocabulary (unseen -> NaN)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in _string_columns(X_train):
        dtype = pd.CategoricalDtype(sorted(X_train[col].dropna().astype(str).unique()))
        X_train[col] = X_train[col].astype(str).astype(dtype)
        X_test[col] = X_test[col].astype(str).astype(dtype)
    return X_train, X_test


def _xgb(seed: int, **params) -> XGBClassifier:
    return XGBClassifier(enable_categorical=True, tree_method="hist", random_state=seed, **params)


def xgb_default(X_train, y_train, X_test, *, seed=0, **_):
    X_train, X_test = _categorical_for_xgb(X_train, X_test)
    return _positive_proba(_xgb(seed).fit(X_train, y_train), X_test)


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def xgb_optuna(
    X_train, y_train, X_test, *, seed=0, tune_budget_s=TUNE_BUDGET_S, tune_trials=None, **_
):
    """XGBoost tuned by Optuna on inner 3-fold CV log-loss, within a wall-clock budget.

    The first trial always runs, so a zero budget still returns a (default-ish) model.
    """
    X_train, X_test = _categorical_for_xgb(X_train, X_test)
    y_train = np.asarray(y_train)
    cv = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        model = _xgb(seed, **_suggest(trial))
        probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        return log_loss(y_train, probs)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = time.perf_counter()
    study.optimize(objective, n_trials=1)  # Optuna runs nothing at timeout=0
    remaining = None if tune_trials is None else tune_trials - 1
    if remaining != 0:
        study.optimize(
            objective, n_trials=remaining, timeout=max(0.0, tune_budget_s - (time.perf_counter() - start))
        )
    best = _xgb(seed, **study.best_params).fit(X_train, y_train)
    return _positive_proba(best, X_test)


def logreg(X_train, y_train, X_test, *, seed=0, **_):
    strings = _string_columns(X_train)
    numeric = [c for c in X_train.columns if c not in strings]
    prep = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                strings,
            ),
        ]
    )
    model = make_pipeline(prep, LogisticRegression(max_iter=1000, random_state=seed))
    return _positive_proba(model.fit(X_train, y_train), X_test)


ARMS: dict[str, Callable] = {
    "tabpfn": tabpfn_raw,
    "tabpfn_ordinal": tabpfn_ordinal,
    "tabpfn_thinking": tabpfn_thinking,
    "xgb": xgb_default,
    "xgb_optuna": xgb_optuna,
    "logreg": logreg,
}


def fit_predict_proba(
    arm: str, X_train, y_train, X_test, *, tabpfn_factory: TabPFNFactory | None = None, **options
) -> np.ndarray:
    """P(churn) for each row of X_test from the named arm.

    `tabpfn_factory` is injected so tests run offline; it defaults to the hosted client.
    Options not used by an arm (seed, tune_budget_s, tune_trials, thinking_metric) are ignored.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return ARMS[arm](
        X_train, y_train, X_test, tabpfn_factory=tabpfn_factory or default_tabpfn_factory, **options
    )

import numpy as np
import pandas as pd
import pytest

from fakes import FakeTabPFNFactory
from revbench.metrics import roc_auc
from revbench.models import ARMS, fit_predict_proba


def toy(n=160, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    y = ((signal + (contract == "Month-to-month") + rng.normal(scale=0.5, size=n)) > 0.8).astype(int)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
            "tenure": rng.integers(0, 70, size=n),
            "contract": contract,
            "plan": rng.choice(["a", "b"], size=n),
        }
    )
    X.loc[::17, "signal"] = np.nan
    return X, y


@pytest.fixture
def split():
    X, y = toy()
    return X.iloc[:100], y[:100], X.iloc[100:], y[100:]


def run(arm, split, factory=None, **kw):
    X_train, y_train, X_test, _ = split
    return fit_predict_proba(
        arm, X_train, y_train, X_test, tabpfn_factory=factory or FakeTabPFNFactory(), **kw
    )


def test_arm_names():
    assert set(ARMS) == {
        "tabpfn", "tabpfn_ordinal", "tabpfn_thinking", "xgb", "xgb_optuna", "logreg"
    }  # fmt: skip


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_returns_valid_probabilities_on_mixed_types(arm, split):
    probs = run(arm, split, tune_trials=3)
    assert probs.shape == (len(split[2]),)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_raw_arm_sends_string_columns_unchanged(split):
    X_train, _, X_test, _ = split
    factory = FakeTabPFNFactory()
    run("tabpfn", split, factory)
    (est,) = factory.instances
    pd.testing.assert_frame_equal(est.fit_X, X_train)
    pd.testing.assert_frame_equal(est.predict_X, X_test)
    assert est.kwargs.get("thinking_mode", False) is False


def test_ordinal_arm_sends_numeric_columns_only(split):
    factory = FakeTabPFNFactory()
    run("tabpfn_ordinal", split, factory)
    (est,) = factory.instances
    for X in (est.fit_X, est.predict_X):
        assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
        assert list(X.columns) == list(split[0].columns)
    assert est.fit_X["contract"].nunique() == 3


def test_ordinal_arm_encodes_unseen_test_category_as_minus_one(split):
    X_train, y_train, X_test, _ = split
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "plan"] = "never-seen"
    factory = FakeTabPFNFactory()
    fit_predict_proba("tabpfn_ordinal", X_train, y_train, X_test, tabpfn_factory=factory)
    assert factory.instances[0].predict_X["plan"].iloc[0] == -1


def test_thinking_arm_enables_thinking_with_a_metric(split):
    factory = FakeTabPFNFactory()
    run("tabpfn_thinking", split, factory)
    kwargs = factory.instances[0].kwargs
    assert kwargs["thinking_mode"] is True
    assert kwargs["thinking_metric"] == "log_loss"


def test_thinking_metric_is_configurable(split):
    factory = FakeTabPFNFactory()
    run("tabpfn_thinking", split, factory, thinking_metric="roc_auc")
    assert factory.instances[0].kwargs["thinking_metric"] == "roc_auc"


@pytest.mark.parametrize("arm", ["xgb", "xgb_optuna", "logreg"])
def test_non_tabpfn_arms_never_touch_the_client(arm, split):
    factory = FakeTabPFNFactory()
    run(arm, split, factory, tune_trials=2)
    assert factory.calls == 0


@pytest.mark.parametrize("arm", ["xgb", "xgb_optuna", "logreg"])
def test_baselines_learn_a_real_signal(arm):
    X, y = toy(600, seed=1)
    probs = fit_predict_proba(
        arm, X.iloc[:400], y[:400], X.iloc[400:], seed=0, tune_trials=5, tabpfn_factory=None
    )
    assert roc_auc(y[400:], probs) > 0.8


def test_baselines_survive_an_unseen_test_category(split):
    X_train, y_train, X_test, _ = split
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "contract"] = "never-seen"
    for arm in ("xgb", "xgb_optuna", "logreg"):
        probs = fit_predict_proba(
            arm, X_train, y_train, X_test, tune_trials=2, tabpfn_factory=None
        )
        assert np.isfinite(probs).all()


def test_xgb_optuna_respects_wall_clock_budget_but_always_runs_one_trial(split):
    probs = run("xgb_optuna", split, tune_budget_s=0.0)
    assert np.isfinite(probs).all()


def test_same_seed_gives_same_xgb_predictions(split):
    assert np.array_equal(run("xgb", split, seed=3), run("xgb", split, seed=3))


def test_unknown_arm_raises(split):
    with pytest.raises(ValueError, match="nope"):
        run("nope", split)

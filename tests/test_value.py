import numpy as np
import pandas as pd
import pytest

from revbench import value
from revbench.cache import PredictionCache


def test_net_value_hand_checked_toy_case():
    # 10 customers, budget 30% -> the 3 highest scores are targeted: customers 0, 1, 2.
    # Customers 0 and 2 churn (values 1000 and 400); customer 1 does not.
    y = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    cust = np.array([1000, 500, 400, 9999, 1, 1, 1, 1, 1, 1], dtype=float)
    # saved = 0.2 * (1000 + 400) = 280; cost = 3 * 50 = 150
    got = value.net_value(y, score, budget=0.3, save_rate=0.2, customer_value=cust, offer_cost=50)
    assert got == pytest.approx(130.0)


def test_net_value_scalar_customer_value_and_zero_budget():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.0])
    assert value.net_value(y, score, 0.5, 0.5, 100.0, 10.0) == pytest.approx(0.5 * 100 - 2 * 10)
    assert value.net_value(y, score, 0.0, 0.5, 100.0, 10.0) == 0.0


def test_random_targeting_expected_value():
    y = np.array([1, 0, 0, 0])  # 25% churn
    # budget 50% of 4 = 2 offers; expected churners targeted = 2 * 0.25 = 0.5
    got = value.random_value(y, 0.5, 0.4, 100.0, 10.0)
    assert got == pytest.approx(0.5 * 0.4 * 100 - 2 * 10)


def make_cohort(tmp_path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"MonthlyCharges": rng.uniform(20, 100, n), "x": rng.normal(size=n)})
    df["churn"] = (rng.random(n) < 0.25).astype(int)
    df.to_csv(tmp_path / "telco.csv", index=False)
    return tmp_path


def test_simulate_reads_cached_predictions_and_values_telco_by_monthly_charges(tmp_path):
    from revbench import datasets, protocol

    data_dir = make_cohort(tmp_path)
    df = datasets.load("telco", data_dir)
    y = df["churn"].to_numpy()
    _, test = protocol.split(y, 0)
    cache = PredictionCache(tmp_path / "cache")
    perfect = y[test].astype(float)  # a perfect scorer
    cache.put("telco", "xgb", 50, 0, "cfg", perfect, 1.0)
    rows = value.simulate(
        ["telco"], ["xgb"], 50, [0], data_dir=data_dir, cache=cache,
        budgets=(0.1,), save_rates=(0.3,),
    )  # fmt: skip
    assert len(rows) == 2  # the arm plus the random baseline
    arm = next(r for r in rows if r["arm"] == "xgb")
    k = int(np.ceil(0.1 * len(test)))
    cust = df["MonthlyCharges"].to_numpy()[test] * 12
    top = np.argsort(-perfect, kind="stable")[:k]
    expected = 0.3 * cust[top].sum() - k * value.OFFER_COST
    assert arm["net_value"] == pytest.approx(expected)
    assert arm["per_1000"] == pytest.approx(expected / len(test) * 1000)


def test_simulate_fails_loudly_when_prediction_is_not_cached(tmp_path):
    with pytest.raises(FileNotFoundError, match="xgb"):
        value.simulate(
            ["telco"], ["xgb"], 50, [0], data_dir=make_cohort(tmp_path),
            cache=PredictionCache(tmp_path / "empty"),
        )  # fmt: skip

"""Value-vs-budget simulation: what a retention team earns by targeting the top k% by score.

A simulation under stated assumptions, not a measured outcome: every targeted churner is saved
with probability `save_rate`, and every targeted customer costs `OFFER_COST`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from revbench import datasets as cohorts
from revbench import protocol
from revbench.cache import PredictionCache

OFFER_COST = 50.0  # currency units per retention offer
CONSTANT_CUSTOMER_VALUE = 600.0  # annual value of a customer where the cohort has no billing column
BUDGETS = tuple(round(0.05 * i, 2) for i in range(1, 11))  # share of customers targeted
SAVE_RATES = (0.1, 0.2, 0.3)
ARMS = ("tabpfn", "xgb_optuna", "xgb", "logreg")
RANDOM = "random"
VALUE_CSV = Path("results/value.csv")
ASSUMPTIONS = (
    "Simulation, not an outcome. Each targeted churner is saved with probability = save rate; "
    f"offer cost {OFFER_COST:.0f} per targeted customer; customer value = MonthlyCharges x 12 "
    f"(Telco) or {CONSTANT_CUSTOMER_VALUE:.0f} (churn cohort)."
)


def _n_targeted(n_customers: int, budget: float) -> int:
    return int(np.ceil(budget * n_customers - 1e-9))


def net_value(y, score, budget, save_rate, customer_value, offer_cost=OFFER_COST) -> float:
    """Expected net saved value of targeting the top `budget` share of customers by `score`."""
    y, score = np.asarray(y), np.asarray(score)
    cust = np.broadcast_to(np.asarray(customer_value, dtype=float), y.shape)
    k = _n_targeted(len(y), budget)
    top = np.argsort(-score, kind="stable")[:k]
    return float(save_rate * cust[top][y[top] == 1].sum() - k * offer_cost)


def random_value(y, budget, save_rate, customer_value, offer_cost=OFFER_COST) -> float:
    """Expected net value of targeting `budget` of customers at random."""
    y = np.asarray(y)
    cust = np.broadcast_to(np.asarray(customer_value, dtype=float), y.shape)
    k = _n_targeted(len(y), budget)
    churner_value = (cust * y).sum() / len(y)  # expected value per randomly chosen customer
    return float(save_rate * k * churner_value - k * offer_cost)


def customer_values(name: str, X: pd.DataFrame) -> np.ndarray | float:
    if name == "telco":
        return X["MonthlyCharges"].to_numpy(dtype=float) * 12
    return CONSTANT_CUSTOMER_VALUE


def simulate(
    datasets: list[str],
    arms: list[str],
    n: int,
    seeds: list[int],
    *,
    data_dir: Path = cohorts.DATA_DIR,
    cache: PredictionCache | None = None,
    budgets=BUDGETS,
    save_rates=SAVE_RATES,
) -> list[dict]:
    """One row per (dataset, arm, seed, save rate, budget), scored from cached predictions only."""
    cache = cache or PredictionCache()
    rows = []
    for name in datasets:
        df = cohorts.load(name, data_dir)
        y = df[cohorts.TARGET].to_numpy()
        X = df.drop(columns=[cohorts.TARGET])
        for seed in seeds:
            _, test = protocol.split(y, seed)
            y_test, cust = y[test], customer_values(name, X.iloc[test])
            for arm in [*arms, RANDOM]:
                proba = None if arm == RANDOM else cache.lookup(name, arm, n, seed)
                for save_rate in save_rates:
                    for budget in budgets:
                        v = (
                            random_value(y_test, budget, save_rate, cust)
                            if proba is None
                            else net_value(y_test, proba, budget, save_rate, cust)
                        )
                        rows.append(
                            {
                                "dataset": name, "arm": arm, "n": n, "seed": seed,
                                "save_rate": save_rate, "budget": budget,
                                "net_value": v, "per_1000": v / len(y_test) * 1000,
                            }
                        )  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="revbench.value")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seeds", default=",".join(map(str, range(10))))
    p.add_argument("--out", type=Path, default=VALUE_CSV)
    args = p.parse_args(argv)
    names = [c.name for c in cohorts.COHORTS]
    rows = simulate(names, list(ARMS), args.n, [int(s) for s in args.seeds.split(",")])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

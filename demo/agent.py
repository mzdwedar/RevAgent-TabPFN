"""Slim retention agent: trigger -> score -> pick targets -> draft experiment -> human approval.

    uv run python demo/agent.py --offline     # replays recorded TabPFN-3.5 scores, no token
    uv run python demo/agent.py               # scores live through the TabPFN API (needs a token)
    uv run python demo/agent.py --record      # live, and saves the scores for --offline

Nothing is written until a person types `y`; an approved experiment is written once to a local
JSON registry, keyed by a hash of its content, so approving twice does not write twice.
"""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from revbench import datasets as cohorts
from revbench import protocol
from revbench.env import load_env
from revbench.models import fit_predict_proba
from revbench.value import OFFER_COST, customer_values

COHORT = "telco"
REPLAY = Path("demo/replay/telco_scores.csv")
REGISTRY = Path(".demo/registry.json")
VARIANT = "retention offer: 20% off the next 3 months"
GUARDRAILS = (
    "stop if churn among treated customers exceeds the held-out group by 2 pts after 14 days",
    "spend never exceeds the max spend above",
    "customers who contacted support in the last 7 days are excluded",
)
ROLLOUT = 0.10


@dataclass(frozen=True)
class Trigger:
    text: str
    delta_pts: float


def parse_trigger(text: str) -> Trigger:
    m = re.search(r"churn rate up (\d+(?:\.\d+)?) ?pts?", text.lower())
    if not m:
        raise ValueError(f"cannot read a trigger from {text!r}; try 'churn rate up 3 pts'")
    return Trigger(text=text, delta_pts=float(m.group(1)))


def _cohort(data_dir: Path):
    df = cohorts.load(COHORT, data_dir)
    y = df[cohorts.TARGET].to_numpy()
    X = df.drop(columns=[cohorts.TARGET])
    pool, test = protocol.split(y, 0)
    return X, y, pool, test


def score_live(data_dir: Path, factory, train_rows: int, draws: int) -> pd.DataFrame:
    """P(churn) for the 30% held-out customers from `draws` models, each on its own small sample."""
    X, y, pool, test = _cohort(data_dir)
    out = {}
    for d in range(draws):
        train = protocol.subsample(y, pool, train_rows, d)
        out[f"draw{d}"] = fit_predict_proba(
            "tabpfn", X.iloc[train], y[train], X.iloc[test], tabpfn_factory=factory, seed=d
        )
    return pd.DataFrame(out, index=pd.Index(test, name="customer"))


def with_values(draws: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add each customer's value so a recorded replay is self-contained (no dataset needed)."""
    X, *_ = _cohort(data_dir)
    return draws.assign(customer_value=customer_values(COHORT, X.loc[draws.index]))


def summarize_scores(recorded: pd.DataFrame) -> pd.DataFrame:
    draws = recorded.drop(columns="customer_value")
    return pd.DataFrame(
        {
            "p_mean": draws.mean(axis=1),
            "p_std": draws.std(axis=1, ddof=0),
            "customer_value": recorded["customer_value"],
        }
    )


def select_targets(
    scores: pd.DataFrame, budget: float, save_rate: float, offer_cost: float = OFFER_COST
) -> pd.DataFrame:
    """Top `budget` share by expected net value, keeping only customers worth an offer."""
    gain = save_rate * scores["customer_value"]
    out = scores.assign(
        net_value=scores["p_mean"] * gain - offer_cost,
        net_value_low=(scores["p_mean"] - scores["p_std"]).clip(lower=0) * gain - offer_cost,
    )
    k = math.ceil(budget * len(out) - 1e-9)
    out = out.sort_values("net_value", ascending=False, kind="stable").head(k)
    return out[out["net_value"] > 0]


def _bucket(experiment_key: str, customer) -> str:
    return hashlib.sha256(f"{experiment_key}:{customer}".encode()).hexdigest()


def draft_experiment(
    trigger: Trigger, targets: pd.DataFrame, save_rate: float, offer_cost: float = OFFER_COST
) -> dict:
    ids = sorted(int(i) for i in targets.index)
    key = json.dumps([trigger.text, VARIANT, ids, ROLLOUT, save_rate, offer_cost])
    n_treated = max(1, math.ceil(ROLLOUT * len(ids))) if ids else 0
    treated = sorted(sorted(ids, key=lambda c: _bucket(key, c))[:n_treated])
    return {
        "id": hashlib.sha256(key.encode()).hexdigest()[:12],
        "trigger": trigger.text,
        "variant": VARIANT,
        "audience": len(ids),
        "targets": ids,
        "treated": treated,
        "rollout": ROLLOUT,
        "save_rate": save_rate,
        "offer_cost": offer_cost,
        "max_spend": len(treated) * offer_cost,
        "expected_net_value": round(float(targets["net_value"].sum()), 2),
        "expected_net_value_low": round(float(targets["net_value_low"].sum()), 2),
        "guardrails": list(GUARDRAILS),
    }


def render(plan: dict) -> str:
    lines = [
        f"EXPERIMENT {plan['id']}  (draft, nothing has been written yet)",
        f"  trigger:   {plan['trigger']}",
        f"  variant:   {plan['variant']}",
        (
            f"  audience:  {plan['audience']} customers worth an offer (simulated, save rate "
            f"{plan['save_rate']:.0%}, offer cost {plan['offer_cost']:.0f})"
        ),
        (
            f"  rollout:   {plan['rollout']:.0%} first = {len(plan['treated'])} customers; "
            f"the other {plan['audience'] - len(plan['treated'])} wait"
        ),
        f"  max spend: {plan['max_spend']:.0f}",
        (
            f"  expected net value if all {plan['audience']} are targeted: "
            f"{plan['expected_net_value']:.0f} (conservative {plan['expected_net_value_low']:.0f})"
        ),
        "  guardrails:",
        *[f"    - {g}" for g in plan["guardrails"]],
        f"  on approval: record {plan['id']} once in the local registry; nothing is sent.",
    ]
    return "\n".join(lines)


def rollout(plan: dict, registry: Path) -> bool:
    """Write the experiment once; False if this exact experiment is already recorded."""
    registry = Path(registry)
    data = json.loads(registry.read_text()) if registry.exists() else {"experiments": {}}
    if plan["id"] in data["experiments"]:
        return False
    data["experiments"][plan["id"]] = plan
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(registry)
    return True


def run(argv=None, *, ask=input, out=print, factory=None) -> int:
    p = argparse.ArgumentParser(prog="demo/agent.py")
    p.add_argument("--trigger", default="churn rate up 3 pts")
    p.add_argument("--offline", action="store_true", help="replay recorded scores; no token")
    p.add_argument("--record", action="store_true", help="live scoring, then save for --offline")
    p.add_argument("--budget", type=float, default=0.10, help="share of customers to consider")
    p.add_argument("--save-rate", type=float, default=0.2)
    p.add_argument("--train-rows", type=int, default=200)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--data-dir", type=Path, default=cohorts.DATA_DIR)
    p.add_argument("--replay", type=Path, default=REPLAY)
    p.add_argument("--registry", type=Path, default=REGISTRY)
    args = p.parse_args(argv)

    trigger = parse_trigger(args.trigger)
    out(f"Trigger: {trigger.text}")
    if args.offline:
        draws = pd.read_csv(args.replay, index_col="customer")
        out(f"Scoring {len(draws)} {COHORT} customers: replaying {args.replay} (no API call)")
    else:
        load_env()
        draws = with_values(
            score_live(args.data_dir, factory or _live_factory(), args.train_rows, args.draws),
            args.data_dir,
        )
        out(f"Scoring {len(draws)} {COHORT} customers with TabPFN-3.5 ({args.draws} models, "
            f"{args.train_rows} training rows each)")  # fmt: skip
        if args.record:
            args.replay.parent.mkdir(parents=True, exist_ok=True)
            draws.to_csv(args.replay)
            out(f"Recorded scores to {args.replay}")
    targets = select_targets(summarize_scores(draws), args.budget, args.save_rate)
    if targets.empty:
        out("No customer is worth an offer under these assumptions. Nothing to do.")
        return 0
    plan = draft_experiment(trigger, targets, args.save_rate)
    out(render(plan))
    if ask("Approve this experiment? [y/N] ").strip().lower() != "y":
        out("Rejected. Nothing was written.")
        return 0
    out(
        f"Approved. Recorded in {args.registry}."
        if rollout(plan, args.registry)
        else f"Approved, but {plan['id']} is already in {args.registry}. Nothing new was written."
    )
    return 0


def _live_factory():
    from revbench.models import default_tabpfn_factory

    return default_tabpfn_factory


if __name__ == "__main__":
    raise SystemExit(run())

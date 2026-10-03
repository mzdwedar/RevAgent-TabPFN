import json

import numpy as np
import pandas as pd
import pytest

import agent
from fakes import FakeTabPFNFactory


def make_cohort(tmp_path, n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"MonthlyCharges": rng.uniform(20, 100, n), "x": rng.normal(size=n)})
    df["churn"] = (rng.random(n) < 0.3).astype(int)
    df.to_csv(tmp_path / "telco.csv", index=False)
    return tmp_path


def scores_frame(p, std=0.0, value=1000.0):
    p = np.asarray(p, dtype=float)
    return pd.DataFrame(
        {"p_mean": p, "p_std": std, "customer_value": value}, index=pd.RangeIndex(len(p))
    )


def test_parse_trigger():
    t = agent.parse_trigger("churn rate up 3 pts")
    assert t.delta_pts == 3.0 and "churn" in t.text
    with pytest.raises(ValueError):
        agent.parse_trigger("sunny day")


def test_select_targets_ranks_by_expected_net_value_and_drops_losers():
    # value 1000, save 0.2, offer cost 50: net = p*200 - 50 -> positive only if p > 0.25
    scores = scores_frame([0.9, 0.1, 0.5, 0.3, 0.2, 0.0])
    t = agent.select_targets(scores, budget=0.5, save_rate=0.2, offer_cost=50)
    assert list(t.index) == [0, 2, 3]  # top 3 by budget; all three have p > 0.25
    assert t.loc[0, "net_value"] == pytest.approx(0.9 * 200 - 50)
    t = agent.select_targets(scores, budget=1.0, save_rate=0.2, offer_cost=50)
    assert list(t.index) == [0, 2, 3]  # 0.2, 0.1, 0.0 would lose money, so they are dropped


def test_select_targets_reports_a_conservative_value_using_uncertainty():
    t = agent.select_targets(scores_frame([0.5], std=0.1), 1.0, 0.2, 50)
    assert t.loc[0, "net_value_low"] == pytest.approx(0.4 * 200 - 50)


def plan_for(n=40):
    scores = scores_frame(np.linspace(0.9, 0.3, n))
    targets = agent.select_targets(scores, budget=1.0, save_rate=0.2, offer_cost=50)
    return agent.draft_experiment(agent.parse_trigger("churn rate up 3 pts"), targets, 0.2, 50)


def test_draft_experiment_rolls_out_to_a_tenth_with_guardrails_and_is_deterministic():
    plan = plan_for(40)
    assert plan["audience"] == 40 and len(plan["treated"]) == 4  # 10% rollout
    assert set(plan["treated"]) <= set(plan["targets"])
    assert plan["guardrails"] and plan["max_spend"] == 4 * 50
    assert plan == plan_for(40) and plan["id"] == plan_for(40)["id"]
    assert plan_for(41)["id"] != plan["id"]


def test_render_shows_what_will_be_written():
    text = agent.render(plan_for(40))
    assert plan_for(40)["id"] in text and "10%" in text and "guardrail" in text.lower()


def test_approve_writes_one_record_and_twice_is_still_one(tmp_path):
    reg = tmp_path / "registry.json"
    plan = plan_for()
    assert agent.rollout(plan, reg) is True
    assert agent.rollout(plan, reg) is False
    assert len(json.loads(reg.read_text())["experiments"]) == 1


def test_reject_writes_nothing(tmp_path):
    data, replay, reg = make_cohort(tmp_path), tmp_path / "replay.csv", tmp_path / "registry.json"
    common = ["--data-dir", str(data), "--replay", str(replay), "--registry", str(reg),
              "--train-rows", "50", "--draws", "2"]  # fmt: skip
    agent.run([*common, "--record"], ask=lambda p: "n", out=lambda s: None, factory=FakeTabPFNFactory())
    out = []
    assert agent.run([*common, "--offline"], ask=lambda p: "n", out=out.append) == 0
    assert not reg.exists() and any("Nothing was written" in line for line in out)


def test_live_record_then_offline_replay_agree_and_approve_writes_once(tmp_path):
    data, replay, reg = make_cohort(tmp_path), tmp_path / "replay.csv", tmp_path / "registry.json"
    common = ["--data-dir", str(data), "--replay", str(replay), "--registry", str(reg),
              "--train-rows", "50", "--draws", "3"]  # fmt: skip
    factory = FakeTabPFNFactory()
    agent.run([*common, "--record"], ask=lambda p: "n", out=lambda s: None, factory=factory)
    assert factory.calls == 3 and replay.exists()

    offline = FakeTabPFNFactory()
    common[1] = str(tmp_path / "no-such-data-dir")  # a fresh clone has the replay but no data/
    agent.run([*common, "--offline"], ask=lambda p: "y", out=lambda s: None, factory=offline)
    assert offline.calls == 0  # replay needs no model and no token
    agent.run([*common, "--offline"], ask=lambda p: "y", out=lambda s: None, factory=offline)
    assert len(json.loads(reg.read_text())["experiments"]) == 1

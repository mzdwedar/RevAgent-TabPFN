import numpy as np
import pandas as pd

from fakes import FakeTabPFNFactory
from revbench import run
from revbench.cache import PredictionCache


def make_data(tmp_path, n=600, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "plan": rng.choice(["a", "b"], n)})
    df["churn"] = (rng.random(n) < rate).astype(int)
    df.to_csv(tmp_path / "toy.csv", index=False)
    return tmp_path


def sweep(tmp_path, factory, arms=("tabpfn", "logreg"), **kw):
    return run.sweep(
        datasets=["toy"],
        arms=list(arms),
        sizes=[50, 100],
        seeds=[0, 1],
        data_dir=make_data(tmp_path),
        cache=PredictionCache(tmp_path / "cache"),
        tabpfn_factory=factory,
        **kw,
    )


def test_sweep_returns_one_row_per_cell_with_metrics(tmp_path):
    rows = sweep(tmp_path, FakeTabPFNFactory())
    assert len(rows) == 1 * 2 * 2 * 2
    row = rows[0]
    for col in ("dataset", "arm", "n", "seed", "pr_auc", "roc_auc", "log_loss", "ece", "seconds", "error"):
        assert col in row
    assert all(r["error"] == "" for r in rows)


def test_second_sweep_makes_zero_model_calls(tmp_path):
    first = FakeTabPFNFactory()
    rows1 = sweep(tmp_path, first)
    assert first.calls == 4

    second = FakeTabPFNFactory()
    rows2 = sweep(tmp_path, second)
    assert second.calls == 0
    assert [r["pr_auc"] for r in rows2] == [r["pr_auc"] for r in rows1]


def test_cache_key_changes_with_options(tmp_path):
    sweep(tmp_path, FakeTabPFNFactory(), arms=("tabpfn_thinking",))
    other = FakeTabPFNFactory()
    sweep(tmp_path, other, arms=("tabpfn_thinking",), thinking_metric="roc_auc")
    assert other.calls == 4


def test_failures_become_rows_and_are_not_cached(tmp_path):
    def broken(**kwargs):
        raise RuntimeError("quota exceeded")

    rows = sweep(tmp_path, broken, arms=("tabpfn",))
    assert len(rows) == 4
    assert all("quota exceeded" in r["error"] for r in rows)
    assert all(np.isnan(r["pr_auc"]) for r in rows)

    healthy = FakeTabPFNFactory()
    rows = sweep(tmp_path, healthy, arms=("tabpfn",))
    assert healthy.calls == 4 and all(r["error"] == "" for r in rows)


def test_write_csv_long_format(tmp_path):
    rows = sweep(tmp_path, FakeTabPFNFactory(), arms=("logreg",))
    path = run.write_csv(rows, tmp_path / "raw")
    out = pd.read_csv(path["toy"])
    assert len(out) == 4 and set(out["arm"]) == {"logreg"}


def test_quick_profile_is_small():
    p = run.PROFILES["quick"]
    assert len(p["datasets"]) == 1 and len(p["sizes"]) == 3 and len(p["seeds"]) == 2


def test_write_csv_merges_with_existing_rows(tmp_path):
    rows = sweep(tmp_path, FakeTabPFNFactory(), arms=("logreg", "xgb"))
    out = tmp_path / "raw"
    run.write_csv([r for r in rows if r["arm"] == "logreg"], out)
    run.write_csv([r for r in rows if r["arm"] == "xgb"], out)
    run.write_csv([r for r in rows if r["arm"] == "xgb"], out)  # rewriting replaces, not duplicates
    merged = pd.read_csv(out / "toy.csv")
    assert len(merged) == len(rows) == 8
    assert not merged.duplicated(["dataset", "arm", "n", "seed"]).any()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fakes import FakeTabPFNFactory

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import api_probe


def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.choice(["x", "y"], size=60)})
    y = (X["a"] + rng.normal(scale=0.5, size=60) > 0).astype(int).to_numpy()
    return X.iloc[:40], y[:40], X.iloc[40:], y[40:]


def test_probe_arm_records_latency_and_metrics():
    X_tr, y_tr, X_te, y_te = data()
    rec = api_probe.probe_arm("tabpfn", X_tr, y_tr, X_te, y_te, FakeTabPFNFactory())
    assert rec["ok"] is True
    assert rec["seconds"] >= 0
    assert set(rec["metrics"]) == {"pr_auc", "roc_auc", "log_loss", "ece"}


def test_probe_arm_captures_errors_instead_of_raising():
    def broken(**kwargs):
        raise RuntimeError("quota exhausted")

    X_tr, y_tr, X_te, y_te = data()
    rec = api_probe.probe_arm("tabpfn_thinking", X_tr, y_tr, X_te, y_te, broken)
    assert rec["ok"] is False
    assert "quota exhausted" in rec["error"]


def test_render_lists_every_arm_and_the_error():
    records = [
        {"arm": "tabpfn", "ok": True, "seconds": 1.5, "metrics": {"pr_auc": 0.5, "roc_auc": 0.7,
         "log_loss": 0.4, "ece": 0.05}, "error": None},
        {"arm": "tabpfn_thinking", "ok": False, "seconds": 0.2, "metrics": None, "error": "boom"},
    ]  # fmt: skip
    md = api_probe.render(records, {"dataset": "telco", "n_train": 200, "n_test": 2000})
    assert "tabpfn_thinking" in md and "boom" in md and "telco" in md and "1.5" in md


def test_load_env_sets_missing_vars_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nPROBE_A=from_file\nPROBE_B=from_file\n\n")
    monkeypatch.setenv("PROBE_B", "from_env")
    monkeypatch.delenv("PROBE_A", raising=False)
    api_probe.load_env(env)
    import os

    assert os.environ["PROBE_A"] == "from_file"
    assert os.environ["PROBE_B"] == "from_env"

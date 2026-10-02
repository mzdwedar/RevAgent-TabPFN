import json

import pandas as pd
import pytest

from revbench import datasets
from revbench.datasets import CohortSpec, clean, fetch

SPEC = CohortSpec(
    name="toy",
    openml_id=1,
    target="Churn",
    positive="Yes",
    id_columns=("cust_id",),
    numeric_columns=("TotalCharges",),
)


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cust_id": [101, 102, 103, 104],
            "Contract": pd.Categorical(["'One year'", "Month-to-month", "Month-to-month", "'One year'"]),
            "plan": ["a", "b", "a", "b"],
            "TotalCharges": ["29.85", " ", "1889.5", "10"],
            "tenure": [1, 34, 2, 45],
            "Churn": ["No", "Yes", "Yes", "No"],
        }
    )


def test_clean_drops_ids_and_makes_binary_target():
    out = clean(raw_frame(), SPEC)
    assert "cust_id" not in out.columns
    assert "Churn" not in out.columns
    assert out["churn"].tolist() == [0, 1, 1, 0]
    assert out["churn"].dtype.kind == "i"


def test_clean_coerces_numeric_and_keeps_blank_as_nan():
    out = clean(raw_frame(), SPEC)
    assert out["TotalCharges"].dtype.kind == "f"
    assert out["TotalCharges"].isna().tolist() == [False, True, False, False]


def test_clean_keeps_strings_as_strings_without_openml_quotes():
    out = clean(raw_frame(), SPEC)
    assert pd.api.types.is_string_dtype(out["Contract"])
    assert not isinstance(out["Contract"].dtype, pd.CategoricalDtype)
    assert out["Contract"].tolist() == ["One year", "Month-to-month", "Month-to-month", "One year"]
    assert out["tenure"].dtype.kind == "i"


def test_clean_rejects_unknown_target_values():
    raw = raw_frame()
    raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Maybe"):
        clean(raw, SPEC)


def test_fetch_writes_data_and_manifest(tmp_path):
    manifest = fetch(tmp_path, specs=[SPEC], fetcher=lambda spec: raw_frame())
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    entry = manifest["toy"]
    assert entry["rows"] == 4
    assert entry["churn_rate"] == 0.5
    assert len(entry["sha256"]) == 64
    assert datasets.load("toy", tmp_path).shape == (4, 5)


def test_second_fetch_is_a_noop_with_identical_manifest(tmp_path):
    first = fetch(tmp_path, specs=[SPEC], fetcher=lambda spec: raw_frame())
    before = (tmp_path / "manifest.json").read_bytes()

    def boom(spec):
        raise AssertionError("second fetch must not hit the network")

    second = fetch(tmp_path, specs=[SPEC], fetcher=boom)
    assert second == first
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_fetch_refetches_when_file_does_not_match_checksum(tmp_path):
    fetch(tmp_path, specs=[SPEC], fetcher=lambda spec: raw_frame())
    (tmp_path / "toy.csv").write_text("corrupted")
    calls = []

    def fetcher(spec):
        calls.append(spec.name)
        return raw_frame()

    fetch(tmp_path, specs=[SPEC], fetcher=fetcher)
    assert calls == ["toy"]
    assert datasets.load("toy", tmp_path).shape == (4, 5)


def test_real_cohort_specs_are_pinned():
    assert {c.name: c.openml_id for c in datasets.COHORTS} == {"churn": 40701, "telco": 42178}

"""Public churn cohorts: fetch from OpenML, clean, and pin with a checksum manifest."""

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET = "churn"
DATA_DIR = Path("data")
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class CohortSpec:
    name: str
    openml_id: int
    target: str
    positive: str
    id_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()


COHORTS = (
    CohortSpec("churn", 40701, target="class", positive="1", id_columns=("phone_number",)),
    CohortSpec(
        "telco", 42178, target="Churn", positive="Yes", numeric_columns=("TotalCharges",)
    ),
)


def clean(raw: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Drop IDs, coerce numerics, strip OpenML quoting, and add a 0/1 `churn` column."""
    df = raw.drop(columns=list(spec.id_columns)).reset_index(drop=True)
    labels = df.pop(spec.target).astype(str).str.strip("'")
    labels_seen = set(labels)
    if len(labels_seen) != 2 or spec.positive not in labels_seen:
        raise ValueError(f"unexpected target values in {spec.name}: {sorted(labels_seen)}")
    for col in spec.numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if col in spec.numeric_columns:
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip("'")
    df[TARGET] = (labels == spec.positive).astype("int64")
    return df


def fetch_openml_frame(spec: CohortSpec) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    return fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto").frame


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _entry(df: pd.DataFrame, payload: bytes) -> dict:
    return {
        "rows": len(df),
        "churn_rate": round(float(df[TARGET].mean()), 6),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / f"{name}.csv")


def fetch(
    data_dir: Path = DATA_DIR,
    specs: Iterable[CohortSpec] = COHORTS,
    fetcher: Callable[[CohortSpec], pd.DataFrame] = fetch_openml_frame,
) -> dict:
    """Build every cohort under `data_dir`; cohorts whose file already matches are skipped."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / MANIFEST
    pinned = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for spec in specs:
        path = data_dir / f"{spec.name}.csv"
        known = pinned.get(spec.name)
        if known and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == known["sha256"]:
            manifest[spec.name] = known
            continue
        df = clean(fetcher(spec), spec)
        payload = _csv_bytes(df)
        path.write_bytes(payload)
        manifest[spec.name] = _entry(df, payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="revbench.datasets")
    parser.add_argument("command", choices=["fetch"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    for name, entry in fetch(args.data_dir).items():
        print(f"{name}: {entry['rows']} rows, churn {entry['churn_rate']:.3f}, {entry['sha256'][:12]}")


if __name__ == "__main__":
    main()

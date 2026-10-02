"""Sweep runner: dataset x arm x n x seed, with cached predictions and long-format CSVs."""

import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from revbench import datasets as cohorts
from revbench import metrics, protocol
from revbench.cache import PredictionCache, config_hash
from revbench.env import load_env
from revbench.models import ARMS, fit_predict_proba

RAW_DIR = Path("results/raw")
METRICS = ("pr_auc", "roc_auc", "log_loss", "ece")

PROFILES = {
    "quick": {"datasets": ["telco"], "sizes": [50, 200, 1000], "seeds": [0, 1]},
    "full": {
        "datasets": [c.name for c in cohorts.COHORTS],
        "sizes": list(protocol.SIZES),
        "seeds": list(range(10)),
    },
}


def _cell(dataset, arm, n, seed, config, cache, factory, options, X_train, y_train, X_test, y_test):
    row = {"dataset": dataset, "arm": arm, "n": n, "seed": seed, "error": ""}
    row["n_train"] = len(y_train)
    try:
        hit = cache.get(dataset, arm, n, seed, config)
        if hit is None:
            proba, seconds = metrics.timed(
                fit_predict_proba, arm, X_train, y_train, X_test,
                tabpfn_factory=factory, seed=seed, **options,
            )  # fmt: skip
            cache.put(dataset, arm, n, seed, config, proba, seconds)
        else:
            proba, seconds = hit
        row.update(metrics.evaluate(y_test, proba))
        row["seconds"] = seconds
    except Exception as exc:  # noqa: BLE001  recorded in the row, never silently skipped
        row.update(dict.fromkeys(METRICS, float("nan")), seconds=float("nan"))
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def sweep(
    datasets: list[str],
    arms: list[str],
    sizes: list[int | None],
    seeds: list[int],
    *,
    data_dir: Path = cohorts.DATA_DIR,
    cache: PredictionCache | None = None,
    tabpfn_factory: Callable | None = None,
    **options,
) -> list[dict]:
    """One row per cell. Errors are recorded in the row and never cached."""
    cache = cache or PredictionCache()
    rows = []
    for name in datasets:
        df = cohorts.load(name, data_dir)
        data_id = hashlib.sha256((Path(data_dir) / f"{name}.csv").read_bytes()).hexdigest()[:12]
        y = df[cohorts.TARGET].to_numpy()
        X = df.drop(columns=[cohorts.TARGET])
        for seed in seeds:
            pool, test = protocol.split(y, seed)
            for n in sizes:
                train = protocol.subsample(y, pool, n, seed)
                for arm in arms:
                    config = config_hash(arm=arm, data=data_id, options=options)
                    rows.append(
                        _cell(
                            name, arm, n, seed, config, cache, tabpfn_factory, options,
                            X.iloc[train], y[train], X.iloc[test], y[test],
                        )
                    )  # fmt: skip
    return rows


def write_csv(rows: list[dict], out_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Merge rows into results/raw/<dataset>.csv; a rerun cell replaces its earlier row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = ["dataset", "arm", "n", "seed"]
    paths = {}
    for name, part in pd.DataFrame(rows).groupby("dataset", sort=False):
        paths[name] = out_dir / f"{name}.csv"
        if paths[name].exists():
            part = pd.concat([pd.read_csv(paths[name]), part], ignore_index=True)
            part = part.drop_duplicates(key, keep="last")
        part.sort_values(key, na_position="last").to_csv(paths[name], index=False)
    return paths


def _ints(text: str) -> list[int | None]:
    return [None if t == "full" else int(t) for t in text.split(",")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="revbench.run")
    p.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    p.add_argument("--datasets", help="comma-separated; default from profile")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--sizes", help="comma-separated ints, or 'full' for the whole pool")
    p.add_argument("--seeds", help="comma-separated ints")
    p.add_argument("--tune-budget-s", type=float, default=10.0)
    args = p.parse_args(argv)
    prof = PROFILES[args.profile]
    load_env()
    names = args.datasets.split(",") if args.datasets else prof["datasets"]
    sizes = _ints(args.sizes) if args.sizes else prof["sizes"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else prof["seeds"]
    arms = args.arms.split(",")
    rows = sweep(names, arms, sizes, seeds, tune_budget_s=args.tune_budget_s)
    for name, path in write_csv(rows).items():
        print(f"{name}: wrote {path}")
    errors = [r for r in rows if r["error"]]
    print(f"{len(rows)} cells, {len(errors)} errors")
    for r in errors[:5]:
        print(f"  {r['dataset']} {r['arm']} n={r['n']} seed={r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()

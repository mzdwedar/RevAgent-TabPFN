"""Real-API smoke test: TabPFN-3.5 and Thinking on one cohort at n=200.

Records latency, the API's cost estimate and usage figures, and any errors in
results/api_probe.md so the full sweep can be sized against the credit allowance.

    uv run python scripts/api_probe.py [--dataset telco] [--n 200] [--seed 0]
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from revbench.datasets import load
from revbench.metrics import evaluate, timed
from revbench.models import default_tabpfn_factory, fit_predict_proba
from revbench.protocol import split, subsample

ARMS = ("tabpfn", "tabpfn_thinking")
OUT = Path("results/api_probe.md")


def load_env(path: Path = Path(".env")) -> None:
    """Export KEY=VALUE lines from `path` for variables not already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def probe_arm(arm, X_train, y_train, X_test, y_test, factory) -> dict:
    """Run one arm; an API failure becomes a record, never an exception."""
    try:
        probs, seconds = timed(fit_predict_proba, arm, X_train, y_train, X_test, tabpfn_factory=factory)
    except Exception as exc:  # noqa: BLE001 - the probe's job is to record failures
        return {"arm": arm, "ok": False, "seconds": None, "metrics": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"arm": arm, "ok": True, "seconds": seconds, "metrics": evaluate(y_test, probs), "error": None}


def render(records: list[dict], meta: dict, extras: dict | None = None) -> str:
    lines = [
        "# TabPFN API probe",
        "",
        (
            f"Run {meta.get('when', '')} on {meta['dataset']}: n_train={meta['n_train']}, "
            f"n_test={meta['n_test']}, seed={meta.get('seed', 0)}."
        ),
        "",
        "| Arm | OK | Seconds | PR-AUC | ROC-AUC | Log-loss | ECE | Error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r["metrics"] or {}
        sec = f"{r['seconds']:.1f}" if r["seconds"] is not None else "-"
        cells = [f"{m[k]:.3f}" if k in m else "-" for k in ("pr_auc", "roc_auc", "log_loss", "ece")]
        lines.append(f"| {r['arm']} | {'yes' if r['ok'] else 'no'} | {sec} | " + " | ".join(cells) + f" | {r['error'] or ''} |")
    for title, payload in (extras or {}).items():
        lines += ["", f"## {title}", "", "```json", json.dumps(payload, indent=2, default=str), "```"]
    return "\n".join(lines) + "\n"


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="telco")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    load_env()
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("TABPFN_TOKEN is not set (see .env.example)")

    from tabpfn_client import estimate_cost
    from tabpfn_client.client import ServiceClient
    from tabpfn_client.config import get_access_token

    df = load(args.dataset)
    y = df.pop("churn").to_numpy()
    pool, test = split(y, args.seed)
    train = subsample(y, pool, args.n, args.seed)
    X_tr, y_tr, X_te, y_te = df.iloc[train], y[train], df.iloc[test], y[test]

    try:
        token = get_access_token()
    except RuntimeError as exc:
        raise SystemExit(f"TabPFN authentication failed: {str(exc).splitlines()[0]}") from exc
    extras = {
        "Cost estimate (no quota used)": {
            op: _safe(lambda op=op: estimate_cost(X_tr, None if op == "thinking_fit" else X_te, operation=op).model_dump())
            for op in ("predict", "thinking_fit", "thinking_predict")
        },
        "API usage before": _safe(lambda: ServiceClient.get_api_usage(token)),
    }
    records = []
    for arm in ARMS:
        print(f"running {arm} ...", flush=True)
        records.append(probe_arm(arm, X_tr, y_tr, X_te, y_te, default_tabpfn_factory))
        print(records[-1], flush=True)
    extras["API usage after"] = _safe(lambda: ServiceClient.get_api_usage(token))
    meta = {
        "dataset": args.dataset, "n_train": len(train), "n_test": len(test), "seed": args.seed,
        "when": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }  # fmt: skip
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(records, meta, extras))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

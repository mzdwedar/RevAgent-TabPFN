"""Report: learning curves, calibration, timing, ablation and a computed headline.

Reads only the committed `results/raw/*.csv`; no token or network needed.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results")
HEADLINE_N = 200
BASELINE = "xgb_optuna"
LABELS = {
    "tabpfn": "TabPFN-3.5",
    "tabpfn_ordinal": "TabPFN-3.5 (ordinal strings)",
    "tabpfn_thinking": "TabPFN-3.5 Thinking",
    "xgb": "XGBoost (default)",
    "xgb_optuna": "XGBoost (tuned)",
    "logreg": "Logistic regression",
}


def _half_width(values: pd.Series) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and 95% t-interval half-width of `metric` per (dataset, arm, n_train)."""
    ok = df[df["error"].isna() | (df["error"] == "")]
    grouped = ok.groupby(["dataset", "arm", "n_train"])[metric]
    out = grouped.agg(mean="mean", count="count").reset_index()
    out["half_width"] = grouped.apply(_half_width).to_numpy()
    return out


def crossover(df: pd.DataFrame, dataset: str, arm: str, n: int, baseline: str) -> int | None:
    """Smallest tested training size where `baseline`'s mean PR-AUC reaches `arm`'s at `n`."""
    table = summarize(df[df["dataset"] == dataset], "pr_auc").set_index(["arm", "n_train"])["mean"]
    target = table[(arm, n)]
    for size, value in table[baseline].sort_index().items():
        if value >= target:
            return int(size)
    return None


def headline(df: pd.DataFrame, dataset: str, baseline: str = BASELINE, n: int = HEADLINE_N) -> str:
    table = summarize(df[df["dataset"] == dataset], "pr_auc").set_index(["arm", "n_train"])["mean"]
    score, base = table[("tabpfn", n)], LABELS.get(baseline, baseline)
    size = crossover(df, dataset, "tabpfn", n, baseline)
    if size is None:
        pool = int(table[baseline].index.max())
        return (
            f"On {dataset}, TabPFN-3.5 trained on {n} rows reaches PR-AUC {score:.3f}; "
            f"no tested size up to the full pool ({pool:,} rows) lets {base} match it."
        )
    return (
        f"On {dataset}, TabPFN-3.5 trained on {n} rows reaches PR-AUC {score:.3f}; "
        f"{base} needs {size:,} rows ({size / n:.1f}x the data) to match it."
    )


def ordinal_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Paired (by seed) PR-AUC difference, ordinal minus raw strings, per dataset and size."""
    ok = df[df["error"].isna() | (df["error"] == "")]
    wide = ok[ok["arm"].isin(["tabpfn", "tabpfn_ordinal"])].pivot_table(
        index=["dataset", "n_train", "seed"], columns="arm", values="pr_auc"
    )
    if wide.empty or "tabpfn_ordinal" not in wide:
        return pd.DataFrame(columns=["dataset", "n_train", "mean_diff", "half_width"])
    diff = (wide["tabpfn_ordinal"] - wide["tabpfn"]).dropna().rename("diff").reset_index()
    grouped = diff.groupby(["dataset", "n_train"])["diff"]
    out = grouped.mean().rename("mean_diff").reset_index()
    out["half_width"] = grouped.apply(_half_width).to_numpy()
    return out


def error_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["error"].notna() & (df["error"] != "")]


def _plot_curves(df, dataset, metrics, path, ylabel_by_metric, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        s = summarize(df[df["dataset"] == dataset], metric)
        for arm, part in s.groupby("arm"):
            part = part.sort_values("n_train")
            line = ax.plot(
                part["n_train"], part["mean"], marker="o", ms=4, label=LABELS.get(arm, arm)
            )[0]
            ax.fill_between(
                part["n_train"], part["mean"] - part["half_width"],
                part["mean"] + part["half_width"], alpha=0.15, color=line.get_color(),
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("training rows (log scale)")
        ax.set_ylabel(ylabel_by_metric[metric])
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines)


def _metric_table(df, dataset, n):
    part = df[(df["dataset"] == dataset) & (df["n_train"] == n)]
    rows = []
    for arm, g in part.groupby("arm"):
        ok = g[g["error"].isna() | (g["error"] == "")]
        rows.append(
            {
                "arm": LABELS.get(arm, arm), "seeds": len(ok),
                "PR-AUC": f"{ok['pr_auc'].mean():.3f}", "ROC-AUC": f"{ok['roc_auc'].mean():.3f}",
                "log-loss": f"{ok['log_loss'].mean():.3f}", "ECE": f"{ok['ece'].mean():.3f}",
                "seconds": f"{ok['seconds'].mean():.1f}",
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def build(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> Path:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(raw_dir.glob("*.csv"))], ignore_index=True)
    lines = ["# Results summary", "", "_Generated by `make report` from `results/raw/*.csv`._", ""]
    for dataset in df["dataset"].unique():
        lines += [f"## {dataset}", "", f"**{headline(df, dataset)}**", ""]
        _plot_curves(
            df, dataset, ["pr_auc", "log_loss"],
            out_dir / "figures" / f"learning_curve_{dataset}.png",
            {"pr_auc": "PR-AUC (higher is better)", "log_loss": "log-loss (lower is better)"},
            f"{dataset}: learning curves, mean and 95% interval over seeds",
        )  # fmt: skip
        _plot_curves(
            df, dataset, ["ece"], out_dir / "figures" / f"calibration_{dataset}.png",
            {"ece": "ECE, 10 bins (lower is better)"},
            f"{dataset}: calibration error vs training size",
        )  # fmt: skip
        pool = int(df.loc[df["dataset"] == dataset, "n_train"].max())
        lines += [f"### Metrics at n = {HEADLINE_N}", "", _md_table(_metric_table(df, dataset, HEADLINE_N)), ""]
        lines += [f"### Time to model at the full pool (n = {pool:,})", ""]
        lines += [_md_table(_metric_table(df, dataset, pool)[["arm", "seconds"]]), ""]
    ab = ordinal_ablation(df)
    lines += ["## Ablation: raw strings vs ordinal encoding (PR-AUC, paired by seed)", ""]
    if ab.empty:
        lines += ["_No ordinal runs._", ""]
    else:
        ab = ab.assign(
            mean_diff=ab["mean_diff"].map("{:+.4f}".format),
            half_width=ab["half_width"].map("{:.4f}".format),
        ).rename(columns={"half_width": "95% +/-"})
        lines += [_md_table(ab), ""]
    errors = error_rows(df)
    lines += ["## Errors", "", f"{len(errors)} of {len(df)} cells failed.", ""]
    if len(errors):
        lines += [_md_table(errors[["dataset", "arm", "n", "seed", "error"]]), ""]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="revbench.report")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    print(f"wrote {build(args.raw_dir, args.out_dir)}")


if __name__ == "__main__":
    main()

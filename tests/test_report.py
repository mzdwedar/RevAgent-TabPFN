import numpy as np
import pandas as pd
import pytest

from revbench import report


def frame(**arm_curves):
    """Long-format rows; each arm maps n_train -> list of PR-AUC values (one per seed)."""
    rows = []
    for arm, curve in arm_curves.items():
        for n, values in curve.items():
            for seed, v in enumerate(values):
                rows.append(
                    {"dataset": "toy", "arm": arm, "n": n, "n_train": n, "seed": seed,
                     "pr_auc": v, "roc_auc": v, "log_loss": 1 - v, "ece": 0.1,
                     "seconds": 1.0, "error": np.nan}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_summarize_mean_and_t_interval():
    df = frame(a={50: [0.4, 0.5, 0.6]})
    out = report.summarize(df, "pr_auc").iloc[0]
    assert out["mean"] == pytest.approx(0.5)
    # t(0.975, df=2) = 4.303; sem = 0.1 / sqrt(3)
    assert out["half_width"] == pytest.approx(4.303 * 0.1 / np.sqrt(3), rel=1e-3)
    assert out["count"] == 3


def test_summarize_single_seed_has_no_band():
    out = report.summarize(frame(a={50: [0.4]}), "pr_auc").iloc[0]
    assert out["half_width"] == 0.0


def test_crossover_finds_smallest_matching_size():
    df = frame(tab={200: [0.6, 0.6]}, xgb={200: [0.5, 0.5], 500: [0.59, 0.59], 1000: [0.61, 0.61]})
    assert report.crossover(df, "toy", "tab", 200, "xgb") == 1000


def test_crossover_none_when_never_matched():
    df = frame(tab={200: [0.9, 0.9]}, xgb={200: [0.5, 0.5], 500: [0.6, 0.6]})
    assert report.crossover(df, "toy", "tab", 200, "xgb") is None


def test_headline_is_computed_from_the_data():
    df = frame(tabpfn={200: [0.6, 0.6]}, xgb={200: [0.5, 0.5], 1000: [0.61, 0.61]})
    text = report.headline(df, "toy", baseline="xgb")
    assert "200" in text and "1,000" in text and "5.0x" in text
    never = frame(tabpfn={200: [0.9, 0.9]}, xgb={200: [0.5, 0.5], 1000: [0.6, 0.6]})
    assert "no tested size" in report.headline(never, "toy", baseline="xgb")


def test_ordinal_ablation_is_paired_by_seed():
    df = frame(tabpfn={50: [0.50, 0.60]}, tabpfn_ordinal={50: [0.52, 0.64]})
    out = report.ordinal_ablation(df).iloc[0]
    assert out["mean_diff"] == pytest.approx(0.03)


def test_error_rows_are_listed():
    df = frame(a={50: [0.4, 0.5]})
    df.loc[0, "error"] = "RuntimeError: boom"
    assert report.error_rows(df).shape[0] == 1


def test_build_writes_figures_and_summary(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = frame(
        tabpfn={50: [0.4, 0.5], 200: [0.6, 0.62]},
        tabpfn_ordinal={50: [0.4, 0.5], 200: [0.6, 0.62]},
        xgb_optuna={50: [0.3, 0.35], 200: [0.5, 0.52], 1000: [0.7, 0.72]},
    )
    df.to_csv(raw / "toy.csv", index=False)
    out = tmp_path / "out"
    report.build(raw, out)
    assert (out / "summary.md").exists()
    assert (out / "figures" / "learning_curve_toy.png").exists()
    assert (out / "figures" / "calibration_toy.png").exists()
    text = (out / "summary.md").read_text()
    assert "toy" in text and "Errors" in text


def test_build_plots_value_curve_when_value_csv_exists(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame(
        tabpfn={50: [0.4, 0.5], 200: [0.6, 0.62]}, xgb_optuna={50: [0.3, 0.35], 200: [0.5, 0.52]}
    ).to_csv(raw / "toy.csv", index=False)
    rows = [
        {"dataset": "toy", "arm": arm, "n": 200, "seed": seed, "save_rate": sr, "budget": b,
         "net_value": 1.0, "per_1000": v}
        for arm, v in (("tabpfn", 5.0), ("random", 1.0))
        for seed in (0, 1) for sr in (0.1, 0.2) for b in (0.1, 0.2)
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp_path / "value.csv", index=False)
    out = tmp_path / "out"
    report.build(raw, out, value_csv=tmp_path / "value.csv")
    assert (out / "figures" / "value_toy.png").exists()
    assert "Simulation" in (out / "summary.md").read_text()

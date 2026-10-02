# Implementation Plan: RevAgent × TabPFN-3.5 — hackathon entry

Task list: [`tasks/todo.md`](todo.md). Written 2026-10-02.

## Overview

A standalone, Apache-2.0 repository for the Prior Labs TabPFN-3.5 hackathon. It answers
one question with evidence:

> **A new subscription app has 200 subscribers and a few weeks of churn labels. Can it run
> a retention experiment on day one?**

Three deliverables, in priority order:

1. **Learning-curve benchmark (headline).** Churn PR-AUC, log-loss and calibration against
   the number of labelled subscribers (50 → full), TabPFN-3.5 (API, incl. Thinking) vs
   default XGBoost, Optuna-tuned XGBoost and logistic regression, 10 seeds, public data.
2. **Value curve.** Net saved value against offer budget when offers are targeted by each
   model's scores, under stated, swept assumptions.
3. **Slim agent demo.** One RevAgent-shaped run (score → target → draft → human approval →
   exactly-once rollout to a simulated registry) in the terminal, no Postgres/Temporal/Slack.

Stretch: uplift ("who is persuadable") on the randomized Hillstrom email experiment.

## Hackathon constraints (from the terms, clauses 3–4)

- Deadline **6 Oct 2026, 23:59 CEST**. Target submission **6 Oct, 15:00 CEST**.
- Public repo under **Apache-2.0**; README with run instructions; input data in the repo
  or at a public URL; written description. Video optional, encouraged.
- Judging: **showcase of TabPFN-3.5 50%**, creativity + practical value 30%, technical
  quality + reproducibility 20%. Judges are Prior Labs staff.

## Architecture decisions

- **Separate repo, not RevAgent itself.** RevAgent is PolyForm Noncommercial, needs
  Postgres/Temporal/Slack to run, and its data is gitignored — three reproducibility
  problems. This repo stays small and runnable; the README links to RevAgent as the
  production-shaped version. You own RevAgent's copyright, so reusing its ideas or code
  here under Apache-2.0 is your call (Open question 2).
- **Hosted API (`tabpfn-client`) for all TabPFN runs.** No CUDA on the M3 Pro, and Thinking
  mode is API-only. Consequence: KKBox is out (ADR-0012 in RevAgent forbids sending it to a
  vendor, and Kaggle rules forbid redistribution).
- **Public data only, fetched by script.** Confirmed on OpenML (licence "Public"):
  `churn` id 40701 (5,000 rows, telecom) and `telco-customer-churn` id 42178 (7,043 rows,
  IBM). A third cohort is chosen in T1. Checksums pinned in a manifest.
- **Every model call is cached to disk** keyed by (dataset, model, n, seed, config hash).
  Reruns are free; `results/` (CSV + PNG, aggregates only) is committed, so a judge can
  regenerate every plot with no token (`make report`) and spot-check with a token
  (`make benchmark-quick`).
- **Evaluation protocol.** Per dataset and seed: a fixed stratified test set (30%, capped
  at 2,000 rows to bound API cost); training sets of size n drawn stratified from the
  remaining pool, nested across n for a given seed. Tuned XGBoost gets Optuna with inner
  3-fold CV and a fixed wall-clock budget — the realistic cost of tuning is part of the
  result.
- **Raw columns go to TabPFN.** RevAgent's `encode()` ordinal-encodes strings first; here
  that becomes an ablation arm (raw vs ordinal), because native categorical/text handling
  is a 3.5 headline capability.
- **Report what we measure.** If TabPFN loses at some n, the chart shows it.

### Repo shape (target)

```
revagent-tabpfn/
  LICENSE                Apache-2.0
  README.md              pitch + chart first, then "run it in 5 minutes"
  pyproject.toml         uv; deps: tabpfn-client, scikit-learn, xgboost, optuna,
                         pandas, matplotlib, pytest, ruff
  Makefile               setup / report / benchmark-quick / benchmark / demo / test
  src/revbench/
    datasets.py          fetch + checksum + clean public cohorts
    protocol.py          splits, nested subsamples, seeds
    models.py            TabPFN (Plus, Thinking, raw/ordinal), XGB, XGB+Optuna, LogReg
    cache.py             on-disk prediction cache
    metrics.py           PR-AUC, ROC-AUC, log-loss, ECE, wall time
    run.py               CLI: sweep → results/raw/*.csv
    value.py             saved-value-vs-budget curves
    report.py            plots + results/summary.md
    uplift.py            (stretch) Hillstrom T-/S-learner, Qini
  demo/agent.py          slim RevAgent run, terminal approval
  results/               committed aggregates + figures
  tests/                 offline tests with a fake TabPFN client
```

## Dependency graph

```
T0 setup ─┬─ T1 datasets ─┬─ T2 protocol+metrics ─┬─ T3 models (fake-tested) ─ T4 API smoke + quota
          │               │                        │
          │               │                        └─ T5 sweep runner + cache ─ T6 full runs ─ T7 report
          │               │                                                           │
          │               │                                                           ├─ T8 value curve
          │               └──────────────────────────── T9 slim demo (needs T3) ──────┤
          │                                                                           └─ T12 uplift (stretch)
          └──────────────────────────────────── T10 README/licence ─ T11 fresh-clone + video + submit
```

## Schedule

| Day | Tasks | End-of-day checkpoint |
|---|---|---|
| Thu 2 Oct (evening) | Join hackathon on the platform, request credits (you) | — |
| Fri 3 Oct | T0–T5 | A: one dataset, all models, two sizes, real API |
| Sat 4 Oct | T6–T8 | B: headline chart exists; go/no-go on T12 |
| Sun 5 Oct | T9–T10 (+T12 if go) | C: `make demo` + README done |
| Mon 6 Oct | T11, submit by 15:00 CEST | D: submitted |

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| API credits / rate limits run out mid-sweep | High | Join hackathon first; T4 measures cost per call; cache everything; Thinking only for n ≤ 500 and 5 seeds; `--quick` profile |
| TabPFN doesn't win at small n on some dataset | Med | Report it honestly; the story is the curve, not a single win |
| Tuned-XGBoost budget looks unfair either way | Med | Fixed, stated budget; also show default XGBoost; budget is a CLI flag |
| Thinking-mode latency too high for the sweep | Med | Run it on a subset; parallelise calls; T4 measures latency |
| Third dataset licence unclear | Low | Ship with the two confirmed OpenML cohorts |
| Scope creep (demo grows into RevAgent v2) | High | Demo is one file, simulated registry, no infra; time-boxed to half a day |
| Late submission | High | Submit a working entry on 5 Oct evening, then update (latest submission counts) |

## Open questions

1. **Folder / repo name.** Using `~/Desktop/revagent-tabpfn`; the GitHub repo name can differ.
2. **Demo code:** write the slim demo fresh (recommended, ~1 file), or copy pieces of
   RevAgent's targeting/policy code and relicense them under Apache-2.0?
3. **Third cohort:** RevAgent's bank set (`Customer-Churn-Records.csv`) has an unclear
   Kaggle licence. Find a public equivalent in T1, or ship with two?
4. **RevAgent repo:** leave it as-is and link to it, or also relicense it? Not needed for
   this entry.

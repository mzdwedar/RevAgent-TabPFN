# Tasks: RevAgent × TabPFN-3.5 hackathon entry

Plan: [`tasks/plan.md`](plan.md). Sizes XS/S/M; nothing over ~5 files.

Standing bar on every task: `uv run ruff check . && uv run pytest` green; no test reaches
the network (TabPFN is faked offline); no raw third-party rows committed outside
`data/` (gitignored); a task is done only when its verification has been run.

**Preconditions (yours, not tasks):**
- [ ] Join the hackathon on platform.priorlabs.ai and accept its terms (unlocks credits)
- [ ] `TABPFN_TOKEN` available (API key from the platform)
- [ ] Decide Open questions 2 and 3 in `plan.md` before T9 / T1 respectively

---

## Phase 1: Foundation (Fri 3 Oct)

- [x] **T0: Project skeleton** · *S*
  - uv project (Python 3.12), `src/revbench/`, `tests/`, Apache-2.0 `LICENSE`, `.gitignore`
    (`data/`, `.cache/`, `.env`), `.env.example`, `Makefile` with `setup`/`test` targets,
    `git init`.
  - Accept: `make setup && make test` succeeds on a clean checkout (one placeholder test).
  - Verify: run both; `git status` shows no data or secrets tracked.
  - Depends: none. Files: `pyproject.toml`, `Makefile`, `LICENSE`, `.gitignore`, `tests/test_smoke.py`.

- [x] **T1: Public cohorts, fetched and pinned** · *S*
  - `datasets.py`: fetch OpenML 40701 (`churn`) and 42178 (`telco-customer-churn`) via
    `fetch_openml`; drop ID columns; coerce `TotalCharges`; binary 0/1 target; keep string
    columns as strings. Write `data/manifest.json` (rows, churn rate, sha256 of the cleaned
    frame). Look for a public third cohort (bank-style); add it only if its licence is clear.
  - Accept: `uv run python -m revbench.datasets fetch` builds every cohort; a second run is a
    no-op with identical checksums; licences recorded in `DATASETS.md`.
  - Verify: run twice and diff the manifest; `tests/test_datasets.py` checks cleaning on a
    tiny in-memory fixture (no network).
  - Depends: T0. Files: `datasets.py`, `DATASETS.md`, `tests/test_datasets.py`.

- [x] **T2: Evaluation protocol and metrics** · *S*
  - `protocol.py`: stratified test split (30%, cap 2,000) per (dataset, seed); nested
    stratified training subsamples for n ∈ {50, 100, 200, 500, 1000, 2000, full}, each with
    ≥ 5 positives. `metrics.py`: PR-AUC, ROC-AUC, log-loss, ECE (10 bins), wall time.
  - Accept: same (dataset, seed) → identical indices; subsample at n ⊂ subsample at 2n; test
    and train never overlap.
  - Verify: `uv run pytest tests/test_protocol.py tests/test_metrics.py` (metrics checked
    against hand-computed values).
  - Depends: T1. Files: `protocol.py`, `metrics.py`, two tests.

- [x] **T3: Model arms behind one interface** · *M*
  - `models.py`: `fit_predict_proba(X_train, y_train, X_test) -> probs` for
    TabPFN-3.5 (raw columns), TabPFN-3.5 (ordinal-encoded), TabPFN-3.5-Thinking
    (`thinking_mode=True`, `thinking_metric` set for log-loss/AUC), XGBoost default,
    XGBoost + Optuna (inner 3-fold CV, wall-clock budget flag), logistic regression
    (one-hot + scaling). The TabPFN client is injected, so tests use a fake.
  - Accept: every arm returns a valid probability vector on a toy frame with mixed types;
    the fake client records that the raw arm received string columns unchanged.
  - Verify: `uv run pytest tests/test_models.py` (offline).
  - Depends: T2. Files: `models.py`, `tests/test_models.py`, `tests/fakes.py`.

- [x] **T4: Real-API smoke test and cost check** · *XS* · **high risk — do early**
  - One script: TabPFN-3.5 and Thinking on one cohort at n = 200, real API. Record latency,
    any credit/usage figure the API exposes, and errors in `results/api_probe.md`.
  - Accept: both arms return probabilities; a measured per-call cost/latency lets you size
    the full sweep (rows sent per call × call count) against the credit allowance.
  - Verify: run it; numbers written to `results/api_probe.md`.
  - Depends: T3, preconditions. Files: `scripts/api_probe.py`, `results/api_probe.md`.

- [x] **T5: Sweep runner with a disk cache** · *M*
  - `cache.py`: predictions stored by (dataset, arm, n, seed, config hash). `run.py` CLI:
    `--datasets --arms --sizes --seeds --profile {quick,full}`; writes long-format
    `results/raw/<dataset>.csv` (one row per dataset × arm × n × seed with every metric and
    wall time). Failures are recorded as rows with an error, never silently skipped.
  - Accept: a second run of the same sweep makes zero model calls; `--profile quick`
    (1 dataset, 3 sizes, 2 seeds) finishes in minutes.
  - Verify: `uv run pytest tests/test_run.py` (fake client counts calls: 2nd run = 0);
    `make benchmark-quick` against the real API.
  - Note: real-API `make benchmark-quick` deliberately not run yet; ran after Checkpoint A (5 non-Thinking arms, 30 cells, 0 errors, ~100k tokens).
  - Depends: T3 (T4 for sizing). Files: `cache.py`, `run.py`, `Makefile`, `tests/test_run.py`.

### Checkpoint A (end of Fri 3 Oct)
- [ ] `make test` green, offline
- [x] `make benchmark-quick` works against the real API
- [x] Credit budget sized from T4; full-sweep plan fits it (or is cut down now)
- [ ] Review with you before spending the credits

---

## Phase 2: Results (Sat 4 Oct)

- [x] **T6: Full sweep** · *S* (mostly wall-clock)
  - Run all cohorts × all arms × all sizes × 10 seeds; Thinking for n ≤ 500 × 5 seeds (or
    as sized at Checkpoint A).
  - Accept: `results/raw/*.csv` complete; error rows < 2% and listed in `results/summary.md`.
  - Verify: row counts match the planned grid; rerun makes zero calls.
  - Done: 640 cells, 0 errors. Plain arms 10 seeds x 7 sizes (ordinal on telco only);
    Thinking at n=200 only, seeds 0-4, both cohorts (~197k tokens/call). Total ~4.3M tokens.
  - Depends: T5, Checkpoint A. Files: `results/raw/*`.

- [ ] **T7: Report and headline chart** · *M*
  - `report.py` (`make report`, no token needed): per cohort, PR-AUC and log-loss vs n with
    95% bands (log x-axis); calibration plot at n = 200; time-to-model table; raw-vs-ordinal
    ablation; `results/summary.md` with the one-sentence headline computed from the data
    (e.g. "TabPFN-3.5 at n=200 ≈ tuned XGBoost at n=N").
  - Accept: `make report` regenerates every figure from committed CSVs alone.
  - Verify: delete `results/figures/`, run `make report`, figures reappear; read the
    headline against the CSV by hand once.
  - Depends: T6. Files: `report.py`, `results/figures/*`, `results/summary.md`.

- [ ] **T8: Value-vs-budget curve** · *S*
  - `value.py`: target the top k% by each arm's score; net saved value =
    Σ churners targeted × save rate × customer value − offers × offer cost. Customer value
    from `MonthlyCharges × 12` on Telco; stated constant elsewhere. Sweep save rate
    {10%, 20%, 30%}. Labelled clearly as a simulation under stated assumptions.
  - Accept: one figure per cohort at n = 200, assumptions printed on the figure.
  - Verify: `uv run pytest tests/test_value.py` (hand-checked toy case); `make report` includes it.
  - Depends: T6. Files: `value.py`, `report.py`, `tests/test_value.py`.

### Checkpoint B (end of Sat 4 Oct)
- [ ] Headline chart exists and the headline sentence is true per the CSVs
- [ ] **Go/no-go on T12 (uplift):** go only if T9–T10 can still finish Sunday

---

## Phase 3: Demo and packaging (Sun 5 Oct)

- [ ] **T9: Slim agent demo** · *M* · time-box: half a day
  - `demo/agent.py`, `make demo`: a trigger ("churn rate up 3 pts") → score a cohort with
    TabPFN-3.5 (live API, or `--offline` replaying cached scores) → pick the top-k targets
    with predicted saved value and uncertainty → draft an experiment (variant, 10% rollout,
    guardrails) → terminal approval showing exactly what will happen → idempotent "rollout"
    written once to a local JSON registry (a rerun or double approve does not write twice).
  - Accept: `make demo` runs end-to-end with no Postgres/Temporal/Slack; `--offline` works
    with no token; rejecting at the prompt writes nothing.
  - Verify: `uv run pytest tests/test_demo.py` (approve → 1 record; approve twice → still 1;
    reject → 0); run `make demo` by hand.
  - Depends: T3 (and T6 cache for `--offline`). Files: `demo/agent.py`, `Makefile`, `tests/test_demo.py`.

- [ ] **T10: README and submission text** · *S*
  - README order: pitch → headline chart → "run it in 5 minutes" (`make setup`,
    `make report`, `make demo --offline`, then with token `make benchmark-quick`) → results
    and honest limitations → method → link to RevAgent as the production-shaped version →
    licence note (code Apache-2.0; model accessed via Prior Labs API under its terms).
    Draft the hackathon description (what it does, how it uses TabPFN-3.5) in `SUBMISSION.md`.
  - Accept: every command in the README is copy-pasted and run once.
  - Verify: run each command from the README verbatim.
  - Depends: T7, T8, T9. Files: `README.md`, `SUBMISSION.md`.

### Checkpoint C (end of Sun 5 Oct)
- [ ] `make test`, `make report`, `make demo --offline` green
- [ ] Push to a public GitHub repo and **submit a first entry tonight** (latest submission counts)

---

## Phase 4: Ship (Mon 6 Oct, submit by 15:00 CEST)

- [ ] **T11: Fresh-clone check, video, final submit** · *S*
  - Clone into a temp dir (or a clean container), follow the README only; fix anything that
    breaks. Record a 2–3 min video: problem → headline chart → `make demo` with approval.
    Update the hackathon entry (repo link, description, video, X/LinkedIn handles).
  - Accept: fresh clone reproduces `make report` and `make demo --offline` with no edits;
    entry shows as submitted on the platform.
  - Verify: the fresh-clone transcript; screenshot of the submitted entry.
  - Depends: Checkpoint C.

### Checkpoint D
- [ ] Entry submitted before 15:00 CEST, 6 Oct

---

## Stretch (only on a "go" at Checkpoint B)

- [ ] **T12: Uplift on Hillstrom** · *M*
  - `uplift.py`: Hillstrom email experiment (public URL, randomized treatment); T-learner and
    S-learner with TabPFN-3.5 vs XGBoost; Qini curves and AUUC at a few training sizes.
  - Accept: one Qini figure + table in `results/`; README section framing it as "who is
    persuadable", RevAgent's actual targeting question.
  - Verify: `uv run pytest tests/test_uplift.py` (Qini on a toy case with known answer);
    `make report` includes it.
  - Depends: T5. Files: `uplift.py`, `report.py`, `tests/test_uplift.py`.

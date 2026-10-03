# Submission text

**Title:** revbench: how few customers does a new app need before it can run a retention experiment?

**What it does.** A new subscription app has almost no labelled churn data. revbench measures how
TabPFN-3.5 performs against tuned XGBoost and logistic regression as the training set grows from 50 rows
to the whole pool, on two public churn cohorts, with 10 seeds each. On churn, TabPFN-3.5 with 200 rows
reaches PR-AUC 0.689, and tuned XGBoost needs 500 rows to match it (at most 2.5x the data; the size grid is
coarse). Telco shows the same crossover with overlapping confidence bands. It then turns the scores into a
value-vs-budget simulation (clearly labelled a simulation, with placeholder costs) and a slim agent that
reads a trigger, ranks customers by expected net value, drafts an experiment, waits for human approval and
rolls out to 10%.

**How it uses TabPFN-3.5.** Through the hosted `tabpfn-client`, with raw string and missing-value columns
passed straight in. It also includes a Thinking-mode ablation, an ordinal-encoding ablation, calibration
and timing comparisons against XGBoost, and, in the demo, per-customer uncertainty taken from 5 TabPFN draws.

**What did not work, and we say so.** Thinking mode gave no consistent gain at about 20x the token cost.
Raw strings did not beat ordinal encoding. Tuned XGBoost trailed default XGBoost at small n on churn.
Telco's XGBoost gap is within noise, and no paired test was run.

**Reproduce.** `make setup && make report && make demo` needs no token. `make benchmark-quick` re-runs a
small slice live.

**Licence.** Code Apache-2.0. Model accessed via the Prior Labs API under its terms.

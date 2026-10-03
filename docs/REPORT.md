# revbench: how few customers does a new app need before it can run a retention experiment?

Project report. Code: https://github.com/mzdwedar/RevAgent-TabPFN. Production-shaped sibling: https://github.com/mzdwedar/RevAgent.

## 1. The idea

RevAgent is a retention agent: it notices a churn trigger, picks which customers to target, drafts an
experiment and waits for a human to approve it. Every one of those steps depends on a churn model, and
a **new** subscription app has very little labelled churn history to train one on. The usual answer is
"wait until you have enough data". This project asks how long that wait really is.

revbench has three parts:
1. **A benchmark.** Learning curves for TabPFN-3.5 against tuned XGBoost, default XGBoost and logistic
   regression, from 50 training rows up to the whole pool.
2. **A value-vs-budget simulation.** What the scores are worth if you only retain-offer the top k% of customers.
3. **A slim agent demo.** Trigger, scoring, targeting, drafted experiment, approval gate, idempotent rollout.

## 2. The thesis

> A pretrained tabular foundation model (TabPFN-3.5) lets a new app act on churn predictions with a
> fraction of the labelled history that a conventionally tuned gradient-boosted model needs.

What would make this false: if tuned XGBoost matched TabPFN at the same small sizes, or if TabPFN's
scores were poorly calibrated, or if the advantage did not survive across cohorts. The experiment was
designed so those outcomes would show up in the data.

The claim is deliberately narrow. It is about *label efficiency* on two public cohorts. It is not a claim
about production uplift, and the value section is explicitly a simulation.

## 3. The experiment

### Data
| cohort | OpenML id | rows | churn rate | pool | test |
|---|---|---|---|---|---|
| churn | 40701 | 5,000 | 14.1% | 3,500 | 1,500 |
| telco | 42178 | 7,043 | 26.5% | 5,043 | 2,000 |

Files are fetched from OpenML and pinned by hash in `data/manifest.json`. Cleaning is in `DATASETS.md`.
No third cohort is included.

### Protocol
- Stratified 30% test split, capped at 2,000 rows, one per seed.
- Training sets are nested, stratified subsamples of the pool with at least 5 positives, at
  n = 50, 100, 200, 500, 1000, 2000 and the full pool.
- 10 seeds per cell. Predictions are cached per (dataset, arm, n, seed, config hash), so reruns are free.
- Primary metric PR-AUC, because churn is imbalanced. Also ROC-AUC, log-loss, ECE and seconds to fit.
- 640 cells in total, 0 failed.

### Arms
| arm | notes |
|---|---|
| TabPFN-3.5 | hosted `tabpfn-client`; string and NaN columns passed in raw |
| TabPFN-3.5 ordinal | Telco only; strings pre-encoded as ordinals (ablation) |
| TabPFN-3.5 Thinking | n = 200 only, 5 seeds (about 20x the token cost of a plain call) |
| XGBoost, tuned | Optuna with a 10 s wall-clock budget |
| XGBoost, default | |
| Logistic regression | |

### Value simulation
Rank the test customers by each arm's score, target the top k% (budget 5 to 50%), and compute
`save_rate x value of targeted churners - offers x offer cost`. Placeholder assumptions: offer cost 50,
save rates 10/20/30%, customer value 600 on the churn cohort and `MonthlyCharges x 12` on Telco.
A random-targeting baseline is included. Models are trained on 200 rows.

### Demo
`demo/agent.py` reads a trigger ("churn rate up 3 pts"), scores 2,000 Telco customers with 5 TabPFN draws
(each trained on its own 200-row sample, so the spread is the uncertainty), ranks by expected net value, and
drops anyone with non-positive value. It drafts an experiment (variant, 10% first rollout, guardrails),
shows exactly what will happen, and asks for approval. The experiment id is a content hash, so approving
twice writes one record. Rejecting writes nothing. Offline mode replays a committed score file and needs no token.

## 4. Results

### 4.1 Headline
On churn, TabPFN-3.5 trained on 200 rows reaches PR-AUC 0.689, and tuned XGBoost needs 500 rows to match it.
On Telco, 0.628 at 200 rows, and tuned XGBoost again needs 500. The size grid is coarse, so the honest
statement is "at most 2.5x the data".

### 4.2 Learning curves (mean PR-AUC over 10 seeds)

Churn:
| n | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| TabPFN-3.5 | 0.340 | 0.513 | **0.689** | 0.813 | 0.874 | 0.898 |
| XGBoost tuned | 0.188 | 0.217 | 0.517 | 0.730 | 0.830 | 0.868 |
| XGBoost default | 0.229 | 0.328 | 0.528 | 0.745 | 0.827 | 0.863 |
| Logistic regression | 0.264 | 0.346 | 0.396 | 0.430 | 0.438 | 0.444 |

Telco:
| n | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| TabPFN-3.5 | 0.564 | 0.598 | **0.628** | 0.659 | 0.666 | 0.672 |
| TabPFN-3.5 ordinal | 0.571 | 0.605 | 0.633 | 0.661 | 0.668 | 0.672 |
| XGBoost tuned | 0.568 | 0.590 | 0.611 | 0.645 | 0.657 | 0.663 |
| XGBoost default | 0.543 | 0.544 | 0.553 | 0.573 | 0.573 | 0.588 |
| Logistic regression | 0.547 | 0.553 | 0.581 | 0.628 | 0.642 | 0.648 |

On churn the gap is large and holds at every size, narrowing as data grows. On Telco, TabPFN leads at
every size but tuned XGBoost is close, and the bands overlap.

### 4.3 Calibration and speed
- At n = 200 TabPFN's ECE is 0.032 (churn) and 0.031 (Telco). Default XGBoost is 0.068 and 0.151.
  Tuned XGBoost is 0.041 and 0.043.
- Seconds to fit at n = 200: TabPFN 3.7 (churn) and 5.2 (Telco), but that includes API latency.
  Tuned XGBoost takes about 10 s, because that is its tuning budget.

### 4.4 Ablations
- **Thinking mode:** PR-AUC 0.700 vs 0.689 on churn, 0.629 vs 0.628 on Telco. Unpaired, 5 seeds against 10,
  at about 20x the token cost. Not worth it here.
- **Raw strings vs ordinal encoding (Telco, paired by seed):** ordinal is better by +0.0071 at n = 50,
  +0.0043 at n = 200, and +0.0002 at the full pool. The difference shrinks to zero. Passing raw strings works
  and costs almost nothing, but it is not a win.

### 4.5 Value vs budget (simulation, n = 200, save rate 20%, net value per 1,000 test customers)
| cohort | arm | 10% budget | 20% | 30% | 50% |
|---|---|---|---|---|---|
| telco | TabPFN-3.5 | 8,380 | 13,474 | 16,122 | 15,674 |
| telco | XGBoost tuned | 7,913 | 12,808 | 15,676 | 15,330 |
| telco | logistic regression | 8,060 | 12,676 | 15,265 | 15,597 |
| telco | random | -274 | -547 | -821 | -1,368 |
| churn | TabPFN-3.5 | 3,768 | 2,576 | -656 | -9,504 |
| churn | XGBoost tuned | 1,848 | 848 | -2,096 | -10,272 |
| churn | random | -3,304 | -6,608 | -9,912 | -16,520 |

Reading it:
- On Telco, TabPFN leads at every budget, but by a small margin over tuned XGBoost and logistic regression.
  Telco's value is high because customer value is `MonthlyCharges x 12`.
- On churn, the cohort is only profitable at small budgets. Under the placeholder numbers (value 600,
  20% save rate, cost 50), blanket targeting loses money, and targeting well is the difference between
  +3,768 and a loss at the 10% budget. That is the practical argument for a good ranker.
- These are simulations under stated assumptions. They have no confidence bands, and the numbers are not outcomes.

### 4.6 Demo
Offline, on the committed Telco replay: 200 customers are worth an offer, with expected net value 17,251
(conservative 13,824, using `p - std`). The first rollout is 20 customers, max spend 1,000. Approve once
writes one record, approve twice still one, reject zero. A fresh clone reproduces this with no token.

## 5. Limitations and what I would not claim

- Telco's confidence bands overlap heavily at n = 200, and I ran no paired test on the XGBoost gap. The clear win is churn.
- Two cohorts, both public, both about telecom-style churn. Nothing here says it transfers to a given app.
- Thinking has 5 seeds against 10 for everything else, and was run at n = 200 only.
- Tuned XGBoost trails default XGBoost at small n on churn (0.517 vs 0.528 at 200). Its 10 s budget is wall-clock, so it depends on the machine and is arguably under-tuned.
- Calibration is shown as ECE against n, not reliability curves, because the committed CSVs hold no per-row predictions.
- Value curves are a simulation with placeholder costs and no uncertainty.
- Demo guardrails are listed text, not enforced. The demo scores one Telco split.
- Only the hosted API was used. No local GPU, and fit times include network latency.

## 6. Reproducing

`make setup && make report && make demo && make test` needs no token. With a TabPFN token in `.env`,
`make data` and `make benchmark-quick` re-run a small slice (about 120k tokens). A fresh clone of the repo
passed `make setup`, `make report` (no diff), `make demo` and `make test`.

## 7. Next steps if this continued
1. Paired tests for the Telco XGBoost gap, and reliability curves.
2. A third, differently shaped cohort.
3. Uplift modelling on a randomized experiment (Hillstrom), which is RevAgent's real targeting question: not who will churn, but who is persuadable.
4. Enforce the demo's guardrails instead of listing them.

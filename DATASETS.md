# Datasets

All cohorts are public and fetched by script (`uv run python -m revbench.datasets fetch`).
Nothing is committed under `data/`; `data/manifest.json` pins row counts, churn rate and the
sha256 of each cleaned CSV, so a re-fetch is checked against the first one.

| Name | Source | Rows | Churn rate | Licence (as listed on OpenML) |
|---|---|---|---|---|
| `churn` | OpenML 40701 (`churn`, telecom) | 5,000 | 14.1% | Public |
| `telco` | OpenML 42178 (`telco-customer-churn`, IBM sample data) | 7,043 | 26.5% | Public |

## Cleaning

- ID columns dropped (`phone_number` in `churn`; `telco` has none on OpenML).
- Target becomes a 0/1 `churn` column (`churn`: class `1`; `telco`: `Churn == Yes`).
- `telco.TotalCharges` coerced to float; the 11 blank values (tenure 0) become NaN, not imputed.
- String columns stay strings, with OpenML's stray single quotes stripped.
- The `churn` cohort arrives from OpenML already numerically coded (plans are 0/1,
  `area_code` is 408/415/510), so it has no string columns. The raw-vs-ordinal TabPFN
  ablation therefore only applies to `telco`.

## Third cohort

Not included. RevAgent's bank set (Kaggle `Customer-Churn-Records.csv`) has an unclear
licence, and no public equivalent with a clear licence has been confirmed. Revisit only if a
clearly licensed cohort turns up; two cohorts are enough for the entry.

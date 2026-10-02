# TabPFN API probe

Run 2026-10-02 18:38 UTC on telco: n_train=200, n_test=2000, seed=0.

| Arm | OK | Seconds | PR-AUC | ROC-AUC | Log-loss | ECE | Error |
|---|---|---|---|---|---|---|---|
| tabpfn | yes | 6.7 | 0.621 | 0.818 | 0.444 | 0.027 |  |
| tabpfn_thinking | yes | 12.4 | 0.625 | 0.820 | 0.442 | 0.028 |  |

## Cost estimate (no quota used)

```json
{
  "predict": {
    "estimated_cost": 10000,
    "pricing_version": "quota_v3",
    "inputs": {
      "train_rows": 200,
      "raw_columns": 19,
      "test_rows": 2000,
      "model_version": "v3.5",
      "operation": "predict",
      "n_estimators": 8,
      "thinking_effort": null
    }
  },
  "thinking_fit": {
    "estimated_cost": 10847,
    "pricing_version": "quota_v3",
    "inputs": {
      "train_rows": 200,
      "raw_columns": 19,
      "test_rows": 0,
      "model_version": "v3.5",
      "operation": "thinking_fit",
      "n_estimators": 8,
      "thinking_effort": "medium"
    }
  },
  "thinking_predict": {
    "estimated_cost": 38375,
    "pricing_version": "quota_v3",
    "inputs": {
      "train_rows": 200,
      "raw_columns": 19,
      "test_rows": 2000,
      "model_version": "v3.5",
      "operation": "thinking_predict",
      "n_estimators": 8,
      "thinking_effort": null
    }
  }
}
```

## API usage before

```json
{
  "daily_tokens_used": 0,
  "daily_token_limit": 5000000,
  "monthly_tokens_used": 0,
  "monthly_token_limit": 20000000,
  "current_usage": 0,
  "usage_limit": 20000000,
  "reset_time": "2026-11-01 00:00:00 UTC",
  "monthly_usage": 0,
  "monthly_limit": 20000000,
  "monthly_reset_time": "2026-11-01 00:00:00 UTC",
  "daily_usage": 0,
  "daily_limit": 5000000,
  "daily_reset_time": "2026-10-03 00:00:00 UTC",
  "thinking_fit_count": 0,
  "thinking_fit_limit": 0
}
```

## API usage after

```json
{
  "daily_tokens_used": 206891,
  "daily_token_limit": 5000000,
  "monthly_tokens_used": 206891,
  "monthly_token_limit": 20000000,
  "current_usage": 206891,
  "usage_limit": 20000000,
  "reset_time": "2026-11-01 00:00:00 UTC",
  "monthly_usage": 206891,
  "monthly_limit": 20000000,
  "monthly_reset_time": "2026-11-01 00:00:00 UTC",
  "daily_usage": 206891,
  "daily_limit": 5000000,
  "daily_reset_time": "2026-10-03 00:00:00 UTC",
  "thinking_fit_count": 0,
  "thinking_fit_limit": 0
}
```

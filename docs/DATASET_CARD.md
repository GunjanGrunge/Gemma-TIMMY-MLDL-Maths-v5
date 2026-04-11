# Dataset Card: V5/V5.2 Minimal Public Samples

## Dataset Summary

This repository includes only minimal public samples of the synthetic math datasets. The full generated datasets are intentionally excluded from GitHub.

Public sample files:

```text
samples/v5_dl_min_sample.jsonl
samples/v52_advanced_min_sample.jsonl
```

## Full Private Dataset

The full local generated dataset contains:

- V5 training examples: 3560
- V5 eval examples: 16
- V5.2 supplemental rows: 876
- Combined V5/V5.1/V5.2 rows: 4956
- Local train path: `outputs/v5/data/v5_dl_train_chat.jsonl`
- Local eval path: `outputs/v5/data/v5_dl_eval_cases.jsonl`
- V5.2 local train path: `outputs/v52/data/v52_combined_train_chat.jsonl`

These full JSONL files are not committed to GitHub.

## Public Sample Coverage

The sample contains examples for:

- softmax cross entropy
- gradient clipping
- classification metrics
- semantic-search cosine ranking
- tensor broadcasting
- forecasting and trading indicators
- portfolio volatility
- hypothesis interpretation
- vector and kinematics calculations

## Schema

Each JSONL row uses a chat-style structure:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "dataset": "v5_public_min_sample",
    "task_type": "..."
  }
}
```

## Notes

The public sample is for demonstration and schema clarity only. It is not large enough to train a useful model.

The private stress-test prompts are not published. Public documentation reports only the test categories and aggregate scores.

# Dataset Card: V5 Minimal Public Sample

## Dataset Summary

This repository includes only a minimal public sample of the V5 synthetic DL math dataset. The full generated dataset is intentionally excluded from GitHub.

Public sample file:

```text
samples/v5_dl_min_sample.jsonl
```

## Full Private Dataset

The full local generated dataset contains:

- Training examples: 3560
- Eval examples: 16
- Local train path: `outputs/v5/data/v5_dl_train_chat.jsonl`
- Local eval path: `outputs/v5/data/v5_dl_eval_cases.jsonl`

These full JSONL files are not committed to GitHub.

## Public Sample Coverage

The sample contains examples for:

- softmax cross entropy
- gradient clipping
- classification metrics
- semantic-search cosine ranking
- tensor broadcasting

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

# Release Manifest

This repository intentionally publishes only the files needed to reproduce and use the open-source hybrid assistant.

## Included In GitHub

- Project README and license
- Deterministic calculator runtime
- V5 dataset generator
- Unsloth training script
- Inference/eval helper
- Minimal public JSONL sample
- Model and dataset cards
- Release notes
- SVG infographics

## Excluded From GitHub

- `.env` and local tokens
- `.venv*` virtual environments
- `outputs/` generated data, reports, model adapters, and checkpoints
- `unsloth_compiled_cache/`
- full private training JSONL files
- adapter weights such as `.safetensors`
- optimizer and checkpoint state such as `.pt` and `.pth`

## Hugging Face Release Target

The expanded LoRA adapter should be uploaded to Hugging Face from:

```text
outputs/v5/models/gemma_dl_lora_expanded
```

Recommended public display name:

```text
Gemma--TIMMY-MLDL-Maths-v5
```

If the Hugging Face Hub rejects double hyphens in the repository slug, use:

```text
Gemma-TIMMY-MLDL-Maths-v5
```

and keep `Gemma--TIMMY-MLDL-Maths-v5` as the display title in the model card.

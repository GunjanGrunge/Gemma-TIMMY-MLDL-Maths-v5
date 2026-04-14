# Release Manifest

This repository intentionally publishes only the files needed to reproduce and use the open-source hybrid assistant.

## Included In GitHub

- Project README and license
- Deterministic calculator runtime
- V5 dataset generator
- V5.2 advanced calculator/runtime files
- V6 curated supplemental dataset generator
- Unsloth training script
- Inference/eval helper
- Minimal public JSONL sample
- Model and dataset cards
- Release notes
- SVG infographics
- WSL V6 training instructions and controlled training scripts

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

The V5.2 advanced adapter/runtime lineage is tracked locally under:

```text
outputs/v52/models/gemma_mldl_advanced_lora
```

The next V6 WSL training path is:

```text
/workspace/martha
/workspace/TimmyBot
outputs/v6/data/v6_combined_train_chat.jsonl
outputs/v6/models/gemma_timmy_mldl_math_lora
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

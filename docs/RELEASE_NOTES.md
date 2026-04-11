# Release Notes: Gemma--TIMMY-MLDL-Maths-v5

## Summary

Gemma--TIMMY-MLDL-Maths-v5 is the fifth iteration of the local ML/DL math assistant project. V5 focuses on deep-learning calculations, exact numeric reliability, and a public release package that does not expose the full private generated dataset.

## What Changed In V5

- Expanded DL task coverage to 23 task families.
- Generated 3560 curated synthetic chat examples.
- Added 16 held-out eval cases.
- Trained an expanded Gemma 2 2B LoRA adapter for 450 steps with Unsloth.
- Added deterministic DL calculators for exact math execution.
- Added the calculator-backed assistant as the recommended inference path.
- Added public README, model card, dataset card, sample data, and SVG infographics.

## Training

- Base model: `unsloth/gemma-2-2b-it-bnb-4bit`
- Adapter: `Gemma--TIMMY-MLDL-Maths-v5`
- Local output path: `outputs/v5/models/gemma_dl_lora_expanded`
- Examples: 3560
- Steps: 450
- Final training loss: approximately 0.274
- Hardware: NVIDIA RTX 3050 8GB

## Evaluation Result

Raw LoRA behavior improved in style and formula coverage, but it remained unreliable for exact arithmetic. The deterministic hybrid calculator path produced correct results on the expanded benchmark.

- Raw expanded LoRA estimate: 2/10 for exact arithmetic.
- Hybrid calculator benchmark: 10/10 on tested deterministic tasks.

## Release Policy

GitHub includes source code, docs, charts, reports, and minimal sample data only. Full private generated JSONL files and adapter weights are not committed to GitHub.

The adapter should be released on Hugging Face as model artifacts, not in GitHub.

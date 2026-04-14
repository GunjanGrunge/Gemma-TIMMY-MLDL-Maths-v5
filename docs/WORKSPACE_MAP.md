# Martha Workspace Map

This repo keeps active runtime modules at the project root because the current
scripts import each other as flat Python modules. Do not move root runtime files
until imports are converted into a package.

## Active Runtime

- `einstein_v6_hybrid_assistant.py` - V6 hybrid assistant and guardrails.
- `einstein_v52_hybrid_assistant.py` - V5.2 fallback/runtime layer used by V6.
- `einstein_v51_hybrid_assistant.py` - V5.1 fallback layer used by V5.2.
- `einstein_dl_hybrid_assistant.py` - deep-learning helper parsing/runtime.
- `advanced_calculators.py`, `dl_calculators.py`, `stats_calculators.py` - deterministic calculator helpers.

## Dataset Generators

- `generate_v61_ollama_consultant_dataset.py` - V6.1 local Gemma/Ollama consultant-data generator.
- `generate_v6_curated_dataset.py` - V6 deterministic curated data.
- `generate_v52_advanced_dataset.py` - V5.2 advanced data.
- `generate_v5_dl_dataset.py` - V5 deep-learning data.

## Training

- `train_gemma_unsloth.py` - LoRA training entrypoint.
- `scripts/train_v6_wsl.sh` - WSL training wrapper.
- `docs/WSL_V6_TRAINING.md` - WSL setup/training notes.

## Evaluation And Debug Tools

- `tools/eval/` - adapter eval, benchmark, GPU check, and diagnostic scripts.

## Legacy Archive

- `archive/legacy_v1/` - earliest Einstein runtime and `math_calculators.py`.
  Preserved for reference only. Not used by V6/V6.1 training or Timmy wiring.

## Generated Outputs

- `outputs/v61/data/` - V6.1 generated training/rejected JSONL.
- `outputs/v61/reports/` - V6.1 generation report.
- `outputs/v6/benchmarks/` - local benchmark JSON/Markdown/SVG.
- `samples/` - small public-safe samples only.

## Current Policy

- Keep datasets and reports for enrichment.
- Keep V5/V5.1/V5.2/V6 final model artifacts for rollback/comparison.
- Delete old checkpoints and temporary smoke/sanity adapters when disk is tight.
- Do not train from eval/benchmark prompts.

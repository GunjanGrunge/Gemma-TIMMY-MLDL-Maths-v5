---
license: mit
base_model: unsloth/gemma-2-2b-it-bnb-4bit
tags:
  - gemma
  - unsloth
  - lora
  - math
  - machine-learning
  - deep-learning
  - forecasting
  - trading
  - guardrails
language:
  - en
pipeline_tag: text-generation
---

# Gemma--TIMMY-MLDL-Maths-v5

Gemma--TIMMY-MLDL-Maths-v5 is a scoped hybrid runtime for machine-learning, deep-learning, statistics, forecasting, trading indicators, portfolio math, and vector/kinematics calculations. It combines model-style explanation with deterministic calculators so the system can speak clearly without pretending generated arithmetic is reliable.

The important design decision is simple: **the model explains, the calculators compute**. Raw small-model adapters are useful for tone, formulas, and tutoring flow, but they are not dependable numeric engines. This repo therefore ships the hybrid path as the public interface.

Current public recommendation:

- Use the `V6` hybrid runtime for production-style ML/DL/stats/forecasting assistance.
- Treat the `V6.1` raw consultant adapter as experimental; it is not yet a standalone replacement for the hybrid path.
- Treat the scoped V6 runtime as a domain assistant, not a general public math model.

![Architecture](assets/architecture.svg)

## Release Snapshot

![V6 release snapshot](assets/v6_release_snapshot.svg)

At release time, the benchmark story is clean:

- `100%` on scoped `timmy_v6_domain` benchmark tasks
- `100%` on numeric guardrails
- `0%` on generic public math smoke prompts
- `76.19%` overall across the mixed local benchmark

That result is not a problem. It is the product definition. Martha V6 is strong when routed into its intended calculator-backed domains and intentionally does not masquerade as a general-purpose public math benchmark model.

## What We Built

- A curated V5 synthetic training set covering ML/DL math operations.
- A V5.2 advanced quantitative extension covering statistics, forecasting, trading indicators, portfolio math, and automotive/vector calculations.
- A Gemma 2 2B LoRA adapter trained locally with Unsloth on an RTX 3050 8GB.
- Deterministic calculators for exact math tasks such as cross entropy, backprop, metrics, gradient descent, tensor shapes, cosine similarity, semantic-search scoring, RSI, VaR, portfolio volatility, and kinematics.
- Guardrails for missing inputs, invalid inputs, ambiguous prompts, unsafe defaults, and vague numeric descriptions.
- Evaluation reports showing why raw LoRA alone is not enough for exact arithmetic.
- Minimal public sample data only. The full generated dataset is intentionally not committed to GitHub.
- A portable `CGI/` client so teammates can download the runtime from GitHub or Hugging Face and query the hybrid assistant directly.

## Current Coverage

![Dataset coverage](assets/dataset_coverage.svg)

The V5/V5.2 hybrid system covers these task families:

- sigmoid + binary cross entropy backprop
- binary cross entropy from probabilities
- softmax cross entropy and logit gradients
- ReLU MLP backprop and linear MSE backprop
- vanilla gradient descent, momentum SGD, Adam, and weight-decay SGD
- gradient clipping
- activation derivatives
- BatchNorm, LayerNorm, and inverted dropout
- CNN output shapes
- scaled dot-product attention
- matrix multiplication and tensor matmul shapes
- tensor broadcasting shapes
- binary classification metrics and multiclass accuracy
- cosine similarity and semantic-search ranking
- descriptive statistics, hypothesis decisions, A/B interpretation, and Cohen's d
- forecasting: weighted moving average, linear trend, seasonal naive, Holt linear smoothing, AR(1)
- trading indicators: RSI, Bollinger Bands, MACD, normal VaR
- portfolio math: expected return, covariance-based variance/volatility, two-asset minimum variance weights
- vector and automotive math: vector magnitude/angle/components, relative velocity, projectile motion, constant-acceleration 2D

## Why Hybrid

![Reliability comparison](assets/reliability_comparison.svg)

The expanded raw LoRA learned answer style and formula structure, but exact arithmetic remained unreliable. The deterministic calculator-backed assistant scored correctly on the scoped benchmark because it computes values directly instead of relying on generated arithmetic.

Recommended usage:

```text
User prompt -> deterministic calculator route -> formatted answer
            -> fallback to Gemma LoRA for explanation or unsupported tasks
```

## Benchmark And Evaluation

The current release benchmark is a local mixed-suite sanity check, not an official GSM8K, MATH, or AIME submission. It is designed to answer one practical question:

`Does the shipped V6 hybrid runtime behave correctly inside its intended domain, and does it fail safely outside it?`

![V6 benchmark chart](outputs/v6/benchmarks/v6_hybrid_benchmark_chart.svg)

### Benchmark Summary

| Track | Pass | Partial | Fail | Score |
|---|---:|---:|---:|---:|
| `timmy_v6_domain` | 10 | 0 | 0 | `100.0%` |
| `guardrail` | 6 | 0 | 0 | `100.0%` |
| `public_math_smoke` | 0 | 0 | 5 | `0.0%` |
| `overall` | 16 | 0 | 5 | `76.19%` |

### What Those Numbers Mean

- The scoped hybrid calculators are ready for the domain problems they were built to solve.
- The guardrail layer is doing its job by refusing or clarifying when inputs are vague, contradictory, or invalid.
- The system is not trying to be a general public math Olympiad assistant, and the benchmark makes that explicit instead of hiding it.

### Example Benchmark Wins

| Case family | Example output behavior |
|---|---|
| Decision-tree impurity | exact `information_gain` and `parent_gini` |
| PCA | correct explained-variance ratio from covariance input |
| Label smoothing CE | exact loss plus `dL/dlogits` |
| Transformer shapes | explicit QKV, score, and output tensor shapes |
| Multiple testing | Bonferroni plus Benjamini-Hochberg decisions |
| Forecast diagnostics | exact `sMAPE` and `MASE` |
| Guardrails | structured `invalid_input`, `missing_info`, and `clarification_needed` JSON |

### Important Benchmark Caveat

The `public_math_smoke` failures are mostly `no_v6_route`, not silent wrong answers. That is a release quality decision: the runtime stays narrow and reliable instead of hallucinating competence outside scope.

## Stress-Test Hardening

![V5.2 stress-test improvement](assets/v52_stress_improvement.svg)

We stress-tested the hybrid system with adversarial prompts designed to break common math-assistant failure modes. The public docs intentionally describe the categories, not the exact private prompts.

| Stress area | What was tested | Result |
|---|---|---|
| Ambiguous defaults | Whether the system silently assumes parameters such as lookback windows or asks for clarification | Structured `missing_info` or transparent `default_used` metadata |
| Conflicting signals | Whether interpretive trading setups are treated as deterministic calculations | Structured `clarification_needed` |
| Cross-domain prompts | Whether the router can separate unrelated domains without mixing formulas | Deterministic solve when inputs are sufficient, otherwise explicit assumptions |
| Misleading/vague inputs | Whether qualitative phrases are converted into fake vectors or fake numbers | Structured `missing_info`; no numeric hallucination |
| Invalid math | Negative variance, log return to zero, and inconsistent physics | Structured `invalid_input` |
| Schema confusion | Whether weights can be mistaken for the data series | Blocked with `missing_info` |

Latest local validation:

| Evaluation | Score |
|---|---:|
| V5.2 messy/combined/missing-info regression suite | `437/437` |
| Regression score | `10/10` |
| Adversarial safety before guardrails | about `6.5/10` |
| Adversarial safety after guardrails | about `9.5/10` |

The key production lesson: the model was not the main bottleneck. The critical upgrade was the guardrail/interface layer around deterministic calculators.

This is why the releaseable surface is the scoped hybrid helper, not a general-purpose math package.

## Default Transparency

When the system uses a standard convention, it marks that explicitly instead of silently assuming.

Example:

```json
{
  "status": "ok",
  "default_used": true,
  "assumptions": {
    "rsi_period": 14
  },
  "note": "Used standard default(s): rsi_period=14. Specify if different."
}
```

If the prompt says the parameter is uncertain or missing, the system returns `missing_info` instead of guessing.

## Repository Contents

```text
.
|-- assets/
|   |-- architecture.svg
|   |-- dataset_coverage.svg
|   |-- reliability_comparison.svg
|   `-- v52_stress_improvement.svg
|-- docs/
|   |-- DATASET_CARD.md
|   |-- MODEL_CARD.md
|   `-- RELEASE_NOTES.md
|-- samples/
|   |-- v5_dl_min_sample.jsonl
|   `-- v52_advanced_min_sample.jsonl
|-- dl_calculators.py
|-- advanced_calculators.py
|-- stats_calculators.py
|-- einstein_dl_hybrid_assistant.py
|-- einstein_hybrid_assistant.py
|-- einstein_v52_hybrid_assistant.py
|-- generate_v5_dl_dataset.py
|-- generate_v52_advanced_dataset.py
|-- math_calculators.py
|-- train_gemma_unsloth.py
|-- test_finetuned_math_assistant.py
|-- check_gpu_torch.py
`-- requirements.txt
```

GitHub intentionally excludes:

- `.env`
- virtual environments
- Unsloth compiled caches
- full training data
- model checkpoints and adapter weights
- local Hugging Face or Ollama caches

## Quick Start

Install dependencies in a CUDA-enabled Python environment:

```powershell
python -m pip install -r requirements.txt
```

Check GPU visibility:

```powershell
python check_gpu_torch.py
```

Run the deterministic DL hybrid assistant:

```powershell
python einstein_dl_hybrid_assistant.py --question "Softmax cross entropy: logits=[2.0, 1.0, 0.1], true_class=0. Compute probabilities, loss, and dL/dlogits."
```

Example output:

```text
probabilities=[0.659, 0.2424, 0.0986], loss=0.417, dL/dlogits=[-0.341, 0.2424, 0.0986]
```

Run a semantic-search math example:

```powershell
python einstein_dl_hybrid_assistant.py --question "Semantic search: query_embedding=[1.0,0.0], document_embeddings=[[0.9,0.1],[0.0,1.0],[0.7,0.7]]. Rank documents by cosine similarity."
```

Expected result:

```text
best_document_index=0, cosine_scores=[0.9939, 0, 0.7071]
```

Use the packaged V6 helper:

```powershell
python -m pip install -e .
martha-v6 --question "Decision tree split case: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute information gain and gini."
```

Use the Node wrapper:

```powershell
node node/cli.js --question "Forecast diagnostics: actuals=[100, 110, 105], forecasts=[98, 112, 108], train_history=[90, 95, 100, 104]. Compute sMAPE and MASE."
```

The Node package is a thin wrapper over the Python runtime, so it still
requires Python plus the local `martha_v6` package to be available.

Or import it directly:

```python
from martha_v6 import answer_structured

result = answer_structured(
    "Forecast diagnostics: actuals=[100, 110, 105], forecasts=[98, 112, 108], train_history=[90, 95, 100, 104]. Compute sMAPE and MASE."
)
print(result.route)
print(result.output)
```

## Training Summary

- Base model: `unsloth/gemma-2-2b-it-bnb-4bit`
- Adapter name: `Gemma--TIMMY-MLDL-Maths-v5`
- V5 training examples: `3560`
- V5 eval examples: `16`
- V5 training steps: `450`
- V5 final training loss: approximately `0.274`
- Hardware: NVIDIA RTX 3050 8GB
- Training framework: Unsloth

V5.2 advanced extension:

- V5.2 supplemental rows: `876`
- Combined V5/V5.1/V5.2 rows: `4956`
- Advanced eval task families: forecasting, trading, statistics, portfolio, automotive/vector math
- V5.2 stress/regression cases: `437`
- V5.2 hybrid regression score: `437/437`

V6 local training extension:

- V6 supplemental rows: `3200`
- Combined V5.2 + V6 rows: `8156`
- V6 eval cases: `308`
- V6 focus: ML evaluation, transformer/DL internals, nonparametric statistics, data analytics workflow math, forecasting diagnostics, and structured guardrail behavior

The expanded adapter is intended to be released on Hugging Face. GitHub contains only source code, reports, visuals, and minimal sample data.

## Recreate The Dataset

The generators are included for reproducibility, but the full generated datasets are not committed.

```powershell
python generate_v5_dl_dataset.py
python generate_v52_advanced_dataset.py
python generate_v6_curated_dataset.py
```

Only minimal public samples are included:

- [samples/v5_dl_min_sample.jsonl](samples/v5_dl_min_sample.jsonl)
- [samples/v52_advanced_min_sample.jsonl](samples/v52_advanced_min_sample.jsonl)
- [samples/v6_curated_min_sample.jsonl](samples/v6_curated_min_sample.jsonl)

## Train The Adapter

Example local training command:

```powershell
$env:UNSLOTH_BASE_MODEL="unsloth/gemma-2-2b-it-bnb-4bit"
$env:UNSLOTH_TRAIN_DATA="outputs/v5/data/v5_dl_train_chat.jsonl"
$env:UNSLOTH_TRAIN_FORMAT="chat"
$env:UNSLOTH_OUTPUT_DIR="outputs/v5/models/gemma_dl_lora_expanded"
$env:UNSLOTH_MAX_SEQ_LENGTH="1024"
$env:UNSLOTH_MAX_STEPS="450"
$env:TORCHDYNAMO_DISABLE="1"
python train_gemma_unsloth.py
```

## Continue Training In WSL For V6

The preferred local training path is native WSL with a separate Linux virtual environment:

```bash
cd /mnt/c/Users/Bot/Desktop/martha
sudo bash scripts/setup_wsl_workspace.sh
cd /workspace/martha
bash scripts/train_v6_wsl.sh
```

The standard WSL workspace is:

```text
/workspace/martha
/workspace/TimmyBot
```

See [docs/WSL_V6_TRAINING.md](docs/WSL_V6_TRAINING.md) for setup, CUDA verification, V6 defaults, and the shared training lock used by Timmy/Codex.

## Limitations

- The raw LoRA adapter should not be treated as a standalone calculator.
- Exact numeric answers should use the deterministic execution layer.
- The included dataset is synthetic and should be independently validated before use in safety-critical workflows.
- Stress tests are local validation artifacts, not a guarantee of correctness for unseen safety-critical workflows.
- This project is educational/research software, not financial, legal, trading, or safety-critical engineering advice.

## License

Code and documentation are released under the MIT License. The Gemma base model and any derived adapter usage remain subject to the applicable Google Gemma and Hugging Face model terms.

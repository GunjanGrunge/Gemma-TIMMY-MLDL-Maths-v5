# Model Card: Gemma--TIMMY-MLDL-Maths-v5

## Model Summary

Gemma--TIMMY-MLDL-Maths-v5 is a LoRA adapter trained on synthetic ML/DL math instruction data and paired with deterministic calculators for production-style numeric reliability. It is designed to provide formula explanations and tutoring-style responses for machine-learning, deep-learning, statistics, forecasting, trading-indicator, portfolio, and vector/kinematics calculation tasks.

The recommended public usage is hybrid: use deterministic calculators for exact arithmetic and the LoRA adapter for explanation style.

## Base Model

- Base: `unsloth/gemma-2-2b-it-bnb-4bit`
- Fine-tuning method: LoRA via Unsloth
- Local adapter path: `outputs/v5/models/gemma_dl_lora_expanded`

## Training Data

- Synthetic examples generated locally.
- V5 training examples: 3560
- V5 eval examples: 16
- V5.2 supplemental examples: 876
- Combined V5/V5.1/V5.2 rows: 4956
- Public GitHub sample: `samples/v5_dl_min_sample.jsonl`
- Public V5.2 sample: `samples/v52_advanced_min_sample.jsonl`
- Full dataset: not included in GitHub.

## Training Configuration

- Max sequence length: 1024
- Training steps: 450
- Final training loss: approximately 0.274
- Hardware: NVIDIA RTX 3050 8GB

## Intended Use

- ML/DL math tutoring
- Backpropagation walkthroughs
- Cross-entropy and gradient explanations
- Classification metric diagnostics
- Tensor shape reasoning
- Semantic-search and cosine-similarity calculations
- Optimizer update explanations
- Forecasting and trading-indicator calculations
- Portfolio return/variance/volatility calculations
- Hypothesis-test and A/B-test interpretation
- Vector, projectile, and kinematics calculations

## Guardrails And Stress Testing

The hybrid runtime was stress-tested against adversarial categories without publishing the private prompt set:

- ambiguous defaults
- conflicting trading signals
- cross-domain prompts
- misleading/vague numeric descriptions
- invalid math domains
- schema-confusion cases such as weights being mistaken for data

Current local validation:

- V5.2 messy/combined/missing-info regression suite: `437/437`
- Regression score: `10/10`
- Adversarial safety before guardrails: about `6.5/10`
- Adversarial safety after guardrails: about `9.5/10`

The runtime returns structured statuses such as `missing_info`, `invalid_input`, `clarification_needed`, and `default_used` instead of silently guessing.

## Recommended Runtime

Use the adapter with the deterministic calculator layer:

```powershell
python einstein_dl_hybrid_assistant.py --question "Softmax cross entropy: logits=[2.0, 1.0, 0.1], true_class=0. Compute probabilities, loss, and dL/dlogits."
```

## Limitations

- The raw LoRA is not a reliable standalone calculator.
- The model can produce plausible but wrong arithmetic if used without calculators.
- The dataset is synthetic and should not be used for safety-critical decisions without independent validation.
- The adapter inherits limitations and terms from the Gemma base model.
- This project is not financial advice, trading advice, legal advice, or safety-critical engineering software.

## License

Project code and documentation are MIT licensed. Gemma base model usage is subject to the applicable Gemma license and terms.

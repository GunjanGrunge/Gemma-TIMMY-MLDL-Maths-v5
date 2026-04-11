# Model Card: Gemma--TIMMY-MLDL-Maths-v5

## Model Summary

Gemma--TIMMY-MLDL-Maths-v5 is a LoRA adapter trained on synthetic ML/DL math instruction data. It is designed to provide formula explanations and tutoring-style responses for machine-learning and deep-learning calculation tasks.

The recommended public usage is hybrid: use deterministic calculators for exact arithmetic and the LoRA adapter for explanation style.

## Base Model

- Base: `unsloth/gemma-2-2b-it-bnb-4bit`
- Fine-tuning method: LoRA via Unsloth
- Local adapter path: `outputs/v5/models/gemma_dl_lora_expanded`

## Training Data

- Synthetic examples generated locally.
- Training examples: 3560
- Eval examples: 16
- Public GitHub sample: `samples/v5_dl_min_sample.jsonl`
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

## License

Project code and documentation are MIT licensed. Gemma base model usage is subject to the applicable Gemma license and terms.

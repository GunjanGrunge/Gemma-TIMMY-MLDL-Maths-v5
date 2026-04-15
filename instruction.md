# Martha V6 Hybrid Usage Guide

This document explains how to use the final Martha V6 package with the hybrid approach.

The core rule is simple:

`The model explains. The calculators compute.`

Do not treat Martha V6 as a raw standalone arithmetic model. The supported production path is the hybrid runtime.

## What The Hybrid Approach Means

When you ask a question, Martha V6 does not blindly generate numbers from free-form text.

Instead it:

1. routes the prompt into a scoped domain handler
2. runs deterministic calculator logic when the task is numeric
3. returns a structured explanation or a structured guardrail response
4. refuses or asks for clarification when inputs are vague, invalid, or incomplete

This is why the package performs well on scoped domain tasks and guardrails, while intentionally not pretending to be a general public math benchmark model.

## Supported Scope

Use Martha V6 for questions like:

- decision-tree impurity and information gain
- PCA variance ratio
- label-smoothed cross entropy and gradients
- transformer tensor shapes
- standardization and z-scores
- Mann-Whitney and Wilcoxon
- multiple testing correction
- forecast diagnostics like sMAPE and MASE
- beta posterior updates
- funnel conversion math
- numeric guardrails for vague or invalid prompts

Do not use Martha V6 as a generic answerer for:

- broad school math outside the scoped calculators
- open-ended public benchmark prompts
- arbitrary factual retrieval
- hidden assumptions with missing numbers

## Python Usage

Install from the repo:

```powershell
python -m pip install -e .
```

Run from the CLI:

```powershell
python -m martha_v6 --question "Decision tree split case: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute information gain and gini."
```

Or use the installed console command:

```powershell
martha-v6 --question "Forecast diagnostics: actuals=[100, 110, 105], forecasts=[98, 112, 108], train_history=[90, 95, 100, 104]. Compute sMAPE and MASE."
```

Import it directly:

```python
from martha_v6 import answer, answer_structured, route_question

question = "Portfolio variance is given as -0.02 with weights [0.5, 0.5]. Compute volatility."

print(route_question(question))
print(answer(question))

result = answer_structured(question)
print(result.route)
print(result.status)
print(result.in_scope)
print(result.output)
```

## Node.js Usage

The Node package is a thin wrapper over the Python runtime. It does not reimplement the calculators in JavaScript.

That means:

- Python must be installed
- the local `martha_v6` package must be available
- Node calls into the Python CLI

CLI usage:

```powershell
node node/cli.js --question "Data prep case: values=[10, 12, 13, 15, 20]. Compute sample mean, sample standard deviation, and z-scores."
```

Import usage:

```javascript
const { answer, answerStructured } = require("./node");

console.log(
  answer("Decision tree split case: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute information gain and gini.")
);

const result = answerStructured(
  "Compute cosine similarity between vector [1, 2, 3] and another vector roughly pointing northeast."
);

console.log(result.route);
console.log(result.status);
console.log(result.output);
```

If needed, you can point Node at a specific Python interpreter:

```powershell
$env:MARTHA_V6_PYTHON="C:\\path\\to\\python.exe"
node node/cli.js --question "..."
```

## Guardrail Behavior

The hybrid runtime is designed to fail safely.

Examples:

- vague cosine prompts return `missing_info`
- negative variance returns `invalid_input`
- mixed forecasting/physics prompts return `clarification_needed`

Typical structured output looks like:

```json
{
  "status": "missing_info",
  "message": "Cosine similarity requires two numeric vectors with the same dimension.",
  "required_fields": ["vector_b_numeric_components"],
  "invalid_fields": [],
  "safe_next_step": "Provide valid numeric inputs or clarify the intended task before computing."
}
```

## Recommended Prompt Style

Best results come from explicit numeric inputs.

Good:

```text
Label smoothing cross entropy: logits=[1.2, -0.4, 2.1, 0.3], true_class=2, epsilon=0.1. Compute loss and gradient.
```

Bad:

```text
Can you kind of estimate how smooth cross entropy might work here?
```

## Benchmark Positioning

Current local release benchmark:

- `100%` on scoped `timmy_v6_domain`
- `100%` on `guardrail`
- `0%` on `public_math_smoke`
- `76.19%` overall

Interpretation:

- inside scope, the hybrid runtime is strong
- outside scope, it prefers not to fake competence

That is the intended behavior.

## Distribution Links

- GitHub: `https://github.com/GunjanGrunge/Gemma-TIMMY-MLDL-Maths-v5`
- Hugging Face: `https://huggingface.co/Gunjan/Gemma-TIMMY-MLDL-Maths-v5`

## Practical Release Guidance

If you are integrating this into an app or workflow:

- call the hybrid package, not the old training code
- keep prompts explicit and numeric
- expect structured refusals when inputs are incomplete
- do not market it as a general-purpose public math model
- market it as a scoped calculator-backed ML/DL/stats assistant

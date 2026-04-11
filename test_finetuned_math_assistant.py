"""
Run quick local inference against the trained Gemma math LoRA adapter.

Examples:
python test_finetuned_math_assistant.py
python test_finetuned_math_assistant.py --question "Explain one backprop update for a 1-layer ANN."
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

# Keep these before torch/unsloth imports.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_ADAPTER_CANDIDATES = [
    Path("outputs/v4/models/gemma_math_lora"),
    Path("outputs/v3/models/gemma_math_lora"),
    Path("outputs/v2/models/gemma_math_lora"),
    Path("outputs/v1/models/gemma_math_lora"),
    Path("outputs/gemma_math_lora"),
]
ADAPTER_PATH_ENV = os.getenv("EINSTEIN_ADAPTER_PATH")
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ_LENGTH", "1024"))

DEFAULT_QUESTIONS = [
    (
        "A logistic regression model predicts p=0.82 for y=1. "
        "Show the binary cross entropy loss and the gradient direction for the logit."
    ),
    (
        "I am training an ANN and the validation loss rises while training loss keeps falling. "
        "Use math reasoning to suggest better hyperparameter tuning."
    ),
    (
        "A car moves east at 20 m/s and north at 15 m/s. "
        "Compute the speed vector magnitude and direction."
    ),
]


def require_cuda_torch() -> None:
    if "+cpu" in torch.__version__ or not torch.cuda.is_available():
        raise SystemExit(
            "CUDA PyTorch is not available in this environment.\n"
            f"torch: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            f"torch.cuda.is_available(): {torch.cuda.is_available()}"
        )


def format_prompt(question: str) -> str:
    return f"### Instruction:\n{question}\n\n### Input:\n\n### Response:\n"


def load_model():
    adapter_path = (
        Path(ADAPTER_PATH_ENV)
        if ADAPTER_PATH_ENV
        else next((path for path in DEFAULT_ADAPTER_CANDIDATES if path.exists()), DEFAULT_ADAPTER_CANDIDATES[0])
    )

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}. Run train_gemma_unsloth.py first."
        )

    import unsloth  # noqa: F401 - patch before transformers/trl imports
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, adapter_path


def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = format_prompt(question)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            repetition_penalty=1.12,
            no_repeat_ngram_size=8,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_length:]
    return trim_response(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def trim_response(text: str) -> str:
    for hard_stop in ["\n### Input:", "\n### Response:"]:
        stop_idx = text.find(hard_stop)
        if stop_idx != -1:
            text = text[:stop_idx].rstrip()

    marker = "Debug note:" if "Debug note:" in text else "Practical note:"
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return text

    next_marker_idx = text.find(marker, marker_idx + len(marker))
    if next_marker_idx != -1:
        text = text[:next_marker_idx].rstrip()

    lines = text.splitlines()
    output_lines = []
    practical_seen = False
    for line in lines:
        if line.startswith(marker):
            if practical_seen:
                break
            practical_seen = True
        output_lines.append(line)
    return "\n".join(output_lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append", help="Question to ask. Can be passed multiple times.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    require_cuda_torch()
    questions = args.question or DEFAULT_QUESTIONS
    model, tokenizer, adapter_path = load_model()

    print(f"Adapter: {adapter_path}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print()

    for idx, question in enumerate(questions, start=1):
        print("=" * 80)
        print(f"Question {idx}: {question}")
        print("-" * 80)
        print(generate_answer(model, tokenizer, question, args.max_new_tokens))
        print()


if __name__ == "__main__":
    main()

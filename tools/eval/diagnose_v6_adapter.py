"""Diagnose V6 adapter generation quality and prompt-format alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch


BASE_MODEL = os.getenv("UNSLOTH_BASE_MODEL", "unsloth/gemma-2-2b-it-bnb-4bit")
V6_ADAPTER = Path("outputs/v6/models/gemma_timmy_mldl_math_lora")
V52_ADAPTER = Path("outputs/v52/models/gemma_mldl_advanced_lora")
V6_DATA = Path("outputs/v6/data/v6_curated_train_chat.jsonl")


def alpaca_prompt(question: str) -> str:
    return f"### Instruction:\n{question}\n\n### Input:\n\n### Response:\n"


def chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_questions() -> list[str]:
    questions = [
        "Data prep case: values=[10, 12, 13, 15, 20]. Compute sample mean, sample standard deviation, and z-scores.",
        "Compute cosine similarity against something roughly northeast.",
        "Decision tree split: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute entropy and information gain.",
    ]
    if V6_DATA.exists():
        with V6_DATA.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                questions.insert(0, row["messages"][1]["content"])
                break
    return questions


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def run_model(label: str, model_path: str | Path, max_new_tokens: int) -> list[dict]:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = []
    for question in load_questions():
        for prompt_type, prompt in [
            ("alpaca", alpaca_prompt(question)),
            ("gemma_chat", chat_prompt(tokenizer, question)),
        ]:
            output = generate(model, tokenizer, prompt, max_new_tokens)
            rows.append({
                "model": label,
                "prompt_type": prompt_type,
                "question": question,
                "output": output,
            })
            print("=" * 80)
            print(f"MODEL={label} PROMPT={prompt_type}")
            print(question)
            print("-" * 80)
            print(output[:1200])
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--out", type=Path, default=Path("outputs/v6/eval/v6_diagnosis_outputs.jsonl"))
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--include-v52", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    runs: list[tuple[str, str | Path]] = []
    if args.include_base:
        runs.append(("base", BASE_MODEL))
    if args.include_v52 and V52_ADAPTER.exists():
        runs.append(("v52_adapter", V52_ADAPTER))
    runs.append(("v6_adapter", V6_ADAPTER))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for label, path in runs:
            for row in run_model(label, path, args.max_new_tokens):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

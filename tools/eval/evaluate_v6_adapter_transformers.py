"""Quick V6 adapter eval using plain Transformers + PEFT inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluate_v6_adapter import score_case


BASE_MODEL = os.getenv("UNSLOTH_BASE_MODEL", "unsloth/gemma-2-2b-it")
DEFAULT_ADAPTER = Path("outputs/v6/models/gemma_timmy_mldl_math_lora")
DEFAULT_EVAL = Path("outputs/v6/data/v6_eval_cases.jsonl")
DEFAULT_OUT = Path("outputs/v6/eval/v6_transformers_eval_outputs.jsonl")


def load_cases(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = []
    for idx, row in enumerate(rows[:limit], 1):
        if "id" not in row:
            row["id"] = f"{path.stem}_{idx:04d}"
        normalized.append(row)
    return normalized


def make_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    quant = BitsAndBytesConfig(load_in_4bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in load_cases(args.eval, args.limit):
        output = generate(model, tokenizer, make_prompt(tokenizer, case["prompt"]), args.max_new_tokens)
        score = score_case(case, output)
        row = {"case": case, "output": output, "score": score}
        rows.append(row)
        print("=" * 80)
        print(case["id"], case["domain"], case["task_type"], score)
        print(case["prompt"])
        print("-" * 80)
        print(output)

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

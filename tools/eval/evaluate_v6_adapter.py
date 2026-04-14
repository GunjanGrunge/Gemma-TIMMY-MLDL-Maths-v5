"""Evaluate the V6 LoRA adapter on curated V6 eval cases.

This is a practical local smoke/eval runner, not a formal benchmark. It loads
the trained adapter once, generates answers for a balanced subset of eval cases,
and scores guardrail status, expected numeric anchors, and response structure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch


DEFAULT_ADAPTER = Path("outputs/v6/models/gemma_timmy_mldl_math_lora")
DEFAULT_EVAL = Path("outputs/v6/data/v6_eval_cases.jsonl")
DEFAULT_OUT = Path("outputs/v6/eval/v6_adapter_eval_report.md")
DEFAULT_JSON = Path("outputs/v6/eval/v6_adapter_eval_outputs.jsonl")
BASE_MODEL = os.getenv("UNSLOTH_BASE_MODEL", "unsloth/gemma-2-2b-it-bnb-4bit")


def format_prompt(question: str) -> str:
    return f"### Instruction:\n{question}\n\n### Input:\n\n### Response:\n"


def r4(value: float) -> str:
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def flatten_numbers(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, bool) or value is None:
        return values
    if isinstance(value, int | float):
        if math.isfinite(float(value)):
            values.append(float(value))
    elif isinstance(value, list):
        for item in value:
            values.extend(flatten_numbers(item))
    elif isinstance(value, dict):
        for item in value.values():
            values.extend(flatten_numbers(item))
    return values


def numeric_anchors(expected: dict[str, Any], limit: int = 8) -> list[str]:
    anchors: list[str] = []
    for value in flatten_numbers(expected):
        if abs(value) > 10000:
            continue
        text = r4(value)
        if text not in anchors:
            anchors.append(text)
        if len(anchors) >= limit:
            break
    return anchors


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def score_case(case: dict[str, Any], output: str) -> dict[str, Any]:
    expected = case.get("expected", {})
    lower = normalize(output)
    expected_status = expected.get("status")
    has_structure = all(marker in output for marker in ["Problem:", "Method:", "Result:"]) or '"status"' in output
    result = {
        "id": case["id"],
        "domain": case["domain"],
        "task_type": case["task_type"],
        "expected_status": expected_status,
        "has_structure": has_structure,
        "score": 0.0,
        "reason": "",
    }
    if expected_status:
        status_hit = expected_status.lower() in lower
        unsafe_numeric = bool(re.search(r"\b\d+(?:\.\d+)?\b", output)) and expected_status in {"missing_info", "clarification_needed"}
        result["score"] = 1.0 if status_hit and not unsafe_numeric else 0.5 if status_hit else 0.0
        result["reason"] = "status_match" if result["score"] == 1.0 else "status_partial_or_missing"
        return result

    anchors = numeric_anchors(expected)
    if not anchors:
        result["score"] = 1.0 if has_structure else 0.5
        result["reason"] = "structure_only"
        return result
    hits = sum(1 for anchor in anchors if anchor in output)
    ratio = hits / len(anchors)
    if ratio >= 0.5 and has_structure:
        result["score"] = 1.0
    elif ratio > 0 or has_structure:
        result["score"] = 0.5
    else:
        result["score"] = 0.0
    result["reason"] = f"numeric_anchor_hits={hits}/{len(anchors)}"
    return result


def load_eval_cases(path: Path, max_cases: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_cases <= 0 or max_cases >= len(rows):
        return rows
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_cases:
        changed = False
        for domain in sorted(by_domain):
            if by_domain[domain] and len(selected) < max_cases:
                selected.append(by_domain[domain].pop(0))
                changed = True
        if not changed:
            break
    return selected


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([format_prompt(prompt)], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def write_report(path: Path, records: list[dict[str, Any]], model_path: Path, eval_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    avg = sum(r["score"]["score"] for r in records) / total if total else 0.0
    full = sum(1 for r in records if r["score"]["score"] == 1.0)
    partial = sum(1 for r in records if r["score"]["score"] == 0.5)
    fail = total - full - partial
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_domain[row["case"]["domain"]].append(row["score"]["score"])

    lines = [
        "# V6 Adapter Eval Report",
        "",
        f"- Adapter: `{model_path}`",
        f"- Eval source: `{eval_path}`",
        f"- Cases run: `{total}`",
        f"- Full pass: `{full}`",
        f"- Partial pass: `{partial}`",
        f"- Fail: `{fail}`",
        f"- Heuristic score: `{avg * 10:.2f}/10`",
        "",
        "## Domain Scores",
    ]
    for domain, scores in sorted(by_domain.items()):
        lines.append(f"- {domain}: `{sum(scores) / len(scores) * 10:.2f}/10` over `{len(scores)}` cases")
    lines.extend(["", "## Failure / Partial Cases"])
    for row in records:
        score = row["score"]["score"]
        if score < 1.0:
            lines.append(f"- `{row['case']['id']}` {row['case']['domain']} / {row['case']['task_type']}: score={score}, {row['score']['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--max-cases", type=int, default=72)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for adapter evaluation.")

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    cases = load_eval_cases(args.eval, args.max_cases)
    records: list[dict[str, Any]] = []
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, 1):
            output = generate(model, tokenizer, case["prompt"], args.max_new_tokens)
            score = score_case(case, output)
            row = {"index": idx, "case": case, "output": output, "score": score}
            records.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{idx}/{len(cases)} {case['id']} score={score['score']} {score['reason']}")

    write_report(args.report, records, args.adapter, args.eval)
    avg = sum(r["score"]["score"] for r in records) / len(records)
    print(f"Report: {args.report}")
    print(f"Heuristic score: {avg * 10:.2f}/10 over {len(records)} cases")


if __name__ == "__main__":
    main()

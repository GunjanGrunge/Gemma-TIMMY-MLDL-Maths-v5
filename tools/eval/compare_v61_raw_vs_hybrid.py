"""Compare raw V6.1 adapter vs hybrid V6 benchmark on shared prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_v6_hybrid_benchmarks import benchmark_cases, run_v6_only, score_case


BASE_MODEL = os.getenv("UNSLOTH_BASE_MODEL", "unsloth/gemma-2-2b-it")
DEFAULT_ADAPTER = Path("outputs/v61/models/gemma_timmy_martha_v61_consultant_lora_502_chattemplate")
DEFAULT_OUT_DIR = Path("outputs/v61/eval_compare")


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
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, Counter] = defaultdict(Counter)
    weighted_scores: dict[str, float] = defaultdict(float)
    weighted_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        case = row["case"]
        verdict = row["verdict"]
        score = row["score"]
        by_track[case["track"]][verdict] += 1
        weighted_scores[case["track"]] += score * case["weight"]
        weighted_totals[case["track"]] += case["weight"]

    summary_by_track = {}
    total_score = 0.0
    total_weight = 0.0
    for track, counts in sorted(by_track.items()):
        score_pct = 100.0 * weighted_scores[track] / weighted_totals[track]
        summary_by_track[track] = {
            "pass": counts["pass"],
            "partial": counts["partial"],
            "fail": counts["fail"],
            "total": sum(counts.values()),
            "score_pct": round(score_pct, 2),
        }
        total_score += weighted_scores[track]
        total_weight += weighted_totals[track]

    overall = {
        "cases": len(rows),
        "pass": sum(1 for row in rows if row["verdict"] == "pass"),
        "partial": sum(1 for row in rows if row["verdict"] == "partial"),
        "fail": sum(1 for row in rows if row["verdict"] == "fail"),
        "score_pct": round(100.0 * total_score / total_weight, 2) if total_weight else 0.0,
    }
    return {"overall": overall, "summary_by_track": summary_by_track}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for raw adapter comparison.")
    if not args.adapter.exists():
        raise SystemExit(f"Missing adapter: {args.adapter}")

    cases = benchmark_cases()
    if args.limit > 0:
        cases = cases[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    quant = BitsAndBytesConfig(load_in_4bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    raw_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []

    for case in cases:
        prompt = make_prompt(tokenizer, case.prompt)
        raw_output = generate(model, tokenizer, prompt, args.max_new_tokens)
        raw_verdict, raw_score, raw_failures = score_case(case, raw_output)
        raw_rows.append(
            {
                "case": {
                    "case_id": case.case_id,
                    "track": case.track,
                    "skill": case.skill,
                    "prompt": case.prompt,
                    "weight": case.weight,
                },
                "engine": "raw_v61_adapter",
                "solver": "adapter_generate",
                "verdict": raw_verdict,
                "score": raw_score,
                "failures": raw_failures,
                "output": raw_output,
            }
        )

        hybrid_solver, hybrid_output = run_v6_only(case.prompt)
        hybrid_verdict, hybrid_score, hybrid_failures = score_case(case, hybrid_output)
        hybrid_rows.append(
            {
                "case": {
                    "case_id": case.case_id,
                    "track": case.track,
                    "skill": case.skill,
                    "prompt": case.prompt,
                    "weight": case.weight,
                },
                "engine": "v6_hybrid",
                "solver": hybrid_solver,
                "verdict": hybrid_verdict,
                "score": hybrid_score,
                "failures": hybrid_failures,
                "output": hybrid_output,
            }
        )
        print(f"{case.case_id}: raw={raw_score} hybrid={hybrid_score}")

    raw_summary = summarize(raw_rows)
    hybrid_summary = summarize(hybrid_rows)
    result = {
        "adapter": str(args.adapter),
        "base_model": args.base_model,
        "raw_v61_adapter": raw_summary,
        "v6_hybrid": hybrid_summary,
        "cases": {
            "raw_v61_adapter": raw_rows,
            "v6_hybrid": hybrid_rows,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "v61_raw_vs_hybrid.json"
    md_path = args.out_dir / "v61_raw_vs_hybrid.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# V6.1 Raw Adapter vs V6 Hybrid",
        "",
        f"- Adapter: `{args.adapter}`",
        f"- Base model: `{args.base_model}`",
        f"- Cases: `{len(cases)}`",
        "",
        "## Overall",
        "",
        f"- Raw V6.1 adapter: `{raw_summary['overall']['score_pct']}%`",
        f"- V6 hybrid: `{hybrid_summary['overall']['score_pct']}%`",
        "",
        "## Track Scores",
        "",
        "| Track | Raw | Hybrid |",
        "|---|---:|---:|",
    ]

    all_tracks = sorted(
        set(raw_summary["summary_by_track"]) | set(hybrid_summary["summary_by_track"])
    )
    for track in all_tracks:
        raw_score = raw_summary["summary_by_track"].get(track, {}).get("score_pct", 0.0)
        hybrid_score = hybrid_summary["summary_by_track"].get(track, {}).get("score_pct", 0.0)
        lines.append(f"| {track} | {raw_score}% | {hybrid_score}% |")

    lines.extend(["", "## Raw Failures And Partials", ""])
    weak_raw = [row for row in raw_rows if row["verdict"] != "pass"]
    if not weak_raw:
        lines.append("- None.")
    else:
        for row in weak_raw:
            case = row["case"]
            lines.append(
                f"- `{case['case_id']}` `{case['track']}` `{case['skill']}`: {row['verdict']} ({', '.join(row['failures'])})"
            )

    lines.extend(["", "## Hybrid Failures And Partials", ""])
    weak_hybrid = [row for row in hybrid_rows if row["verdict"] != "pass"]
    if not weak_hybrid:
        lines.append("- None.")
    else:
        for row in weak_hybrid:
            case = row["case"]
            lines.append(
                f"- `{case['case_id']}` `{case['track']}` `{case['skill']}`: {row['verdict']} ({', '.join(row['failures'])})"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

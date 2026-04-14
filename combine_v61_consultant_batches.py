"""Combine and validate V6.1 consultant generation batches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_v61_ollama_consultant_dataset import extract_calculator_plan, validate_row


DEFAULT_GLOB = "outputs/v61/data/v61_ollama_consultant_train_chat*.jsonl"
DEFAULT_OUT = Path("outputs/v61/data/v61_consultant_combined_train_chat.jsonl")
DEFAULT_REPORT = Path("outputs/v61/reports/v61_consultant_combined_report.md")
DEFAULT_SAMPLE = Path("samples/v61_consultant_combined_min_sample.jsonl")


def read_rows(pattern: str) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for path in sorted(Path(".").glob(pattern)):
        if "combined" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["messages"][1]["content"], row["messages"][2]["content"])
            if key in seen:
                continue
            seen.add(key)
            row["metadata"]["batch_file"] = str(path)
            rows.append(row)
    return rows


def validate_rows(rows: list[dict]) -> dict:
    errors = Counter()
    empty_exact_inputs = 0
    refusal_exact = 0
    for row in rows:
        errors.update(validate_row(row))
        plan = extract_calculator_plan(row["messages"][2]["content"])
        if plan and plan.get("calculator") != "none" and plan.get("requires_exact_calculator") is True and not plan.get("inputs"):
            empty_exact_inputs += 1
        if row["metadata"].get("response_mode") == "refusal" and plan and plan.get("requires_exact_calculator") is True:
            refusal_exact += 1
    return {
        "validation_errors": dict(errors),
        "empty_exact_inputs": empty_exact_inputs,
        "refusal_exact": refusal_exact,
    }


def write_outputs(rows: list[dict], out: Path, report: Path, sample: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    sample.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with sample.open("w", encoding="utf-8") as f:
        for row in rows[: min(10, len(rows))]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(r["metadata"].get("category", "unknown") for r in rows)
    domains = Counter(r["metadata"].get("domain", "unknown") for r in rows)
    tasks = Counter(r["metadata"].get("task_type", "unknown") for r in rows)
    validation = validate_rows(rows)

    lines = [
        "# V6.1 Consultant Combined Dataset Report",
        "",
        f"- Deduped accepted rows: {len(rows)}",
        f"- Validation errors: {validation['validation_errors']}",
        f"- Empty exact calculator inputs: {validation['empty_exact_inputs']}",
        f"- Refusal rows requiring exact calculator: {validation['refusal_exact']}",
        "",
        "## Categories",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in cats.most_common()]
    lines += ["", "## Domains", ""]
    lines += [f"- {key}: {value}" for key, value in domains.most_common()]
    lines += ["", "## Top Tasks", ""]
    lines += [f"- {key}: {value}" for key, value in tasks.most_common(40)]
    report.write_text("\n".join(lines), encoding="utf-8")

    return {
        "rows": len(rows),
        "out": str(out),
        "report": str(report),
        "sample": str(sample),
        "categories": dict(cats),
        "domains": dict(domains),
        "top_tasks": dict(tasks.most_common(20)),
        **validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    rows = read_rows(args.pattern)
    summary = write_outputs(rows, args.out, args.report, args.sample)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

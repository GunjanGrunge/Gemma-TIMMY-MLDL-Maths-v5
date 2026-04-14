"""Generate V6.1 consultant-style data with local Ollama Gemma.

The goal is not to teach arithmetic memorization. This script asks a local
Gemma model to rewrite verified V1-V6 examples into Martha-style supervision:
task classification, known inputs, missing/invalid checks, calculator planning,
and result interpretation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemma3:4b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
ALLOWED_CALCULATORS = {
    "ab_test_interpretation",
    "adam_update",
    "ar1_forecast",
    "backpropagation",
    "beta_posterior",
    "binary_cross_entropy",
    "bollinger_bands",
    "classification_metrics",
    "classifier_ranking_calibration_threshold",
    "cosine_lr_schedule",
    "cohen_d",
    "constant_acceleration",
    "cosine_similarity",
    "cross_entropy",
    "cnn_output_shape",
    "decision_tree_impurity_gain",
    "descriptive_statistics",
    "effective_batch_size",
    "first_differencing",
    "funnel_conversion",
    "gradient_descent",
    "gradient_descent_update",
    "inverted_dropout",
    "logistic_l2_gradient_derivation",
    "logistic_l2_gradient_numeric",
    "hypothesis_test",
    "information_gain",
    "iqr_outliers",
    "kinematics_vector",
    "label_smoothing_cross_entropy",
    "layernorm_forward",
    "linear_regression_gradient",
    "linear_trend_forecast",
    "logistic_regression_gradient",
    "mann_whitney_u",
    "minmax_scaling",
    "multiple_testing_correction",
    "pca_explained_variance",
    "portfolio_return_variance",
    "projectile_motion",
    "relu_mlp_backprop",
    "returns_volatility_sharpe_drawdown",
    "rmse",
    "rsi",
    "rolling_mean",
    "semantic_similarity",
    "silhouette_score",
    "sigmoid_bce_backprop",
    "sigmoid_backprop",
    "sharpe_ratio",
    "smape_mase",
    "softmax_cross_entropy",
    "standardization",
    "stopping_distance",
    "t_test",
    "transformer_attention_shapes",
    "weighted_moving_average",
    "wilcoxon_signed_rank",
    "none",
}
CALCULATOR_ALIASES = {
    "ab_test_lift": "ab_test_interpretation",
    "batch_size": "effective_batch_size",
    "effective_batch": "effective_batch_size",
    "sample_std": "descriptive_statistics",
    "sample_std_dev": "descriptive_statistics",
    "confusion_matrix": "classification_metrics",
    "gradient_clip": "gradient_descent",
    "gradient_descent_analysis": "gradient_descent",
    "kinematics_acceleration": "constant_acceleration",
    "logistic_gradient": "logistic_regression_gradient",
    "mape_calculator": "smape_mase",
    "mape": "smape_mase",
    "momentum_sgd_update": "gradient_descent",
    "one_point_linear_regression_gradient": "gradient_descent",
    "one_sample_t_test": "t_test",
    "paired_t_test_one_sample": "t_test",
    "sample_stats": "descriptive_statistics",
    "sgd_update": "gradient_descent",
    "two_sample_t_test": "t_test",
    "t_test_one_sample": "t_test",
    "vector_magnitude_direction": "kinematics_vector",
    "vector_magnitude": "kinematics_vector",
    "weight_decay_sgd_update": "gradient_descent",
    "anova_one_way": "hypothesis_test",
    "calculate_posterior": "beta_posterior",
    "dropout_forward": "inverted_dropout",
    "holt_linear_forecast": "linear_trend_forecast",
    "kinematics_vector_anchor": "kinematics_vector",
    "kmeans_centroid": "gradient_descent",
    "naive_bayes": "classification_metrics",
    "one_way_anova": "hypothesis_test",
    "seasonal_naive_forecast": "linear_trend_forecast",
    "sigmoid_backprop_anchor": "sigmoid_backprop",
}
SYSTEM = (
    "You are Martha, Timmy's rigorous ML, DL, statistics, and quantitative math "
    "consultant. You do not blindly calculate. You classify the task, identify "
    "known inputs, detect missing or invalid inputs, propose exact calculator "
    "calls when needed, and interpret verified results."
)

SOURCE_FILES = [
    Path("outputs/v6/data/v6_curated_train_chat.jsonl"),
    Path("outputs/v52/data/v52_advanced_train_chat.jsonl"),
    Path("outputs/v51/data/v51_stats_train_chat.jsonl"),
    Path("outputs/v5/data/v5_dl_train_chat.jsonl"),
    Path("outputs/v4/data/v4_train_chat.jsonl"),
    Path("outputs/v3/data/v3_train_chat.jsonl"),
    Path("outputs/v1/data/v1_quant_chat.jsonl"),
]

OUT_DIR = Path("outputs/v61/data")
REPORT_DIR = Path("outputs/v61/reports")
SAMPLE_DIR = Path("samples")


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_messages(row: dict[str, Any]) -> tuple[str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    if not user or not assistant:
        return None
    return user, assistant


def load_sources(max_per_file: int) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for path in SOURCE_FILES:
        for row in read_jsonl(path, max_per_file):
            pair = extract_messages(row)
            if not pair:
                continue
            user, assistant = pair
            meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            sources.append(
                {
                    "source_file": str(path),
                    "source_user": user,
                    "source_assistant": assistant,
                    "source_metadata": meta,
                }
            )
    return sources


def build_prompt(source: dict[str, Any]) -> str:
    meta = source["source_metadata"]
    expected = meta.get("expected", {})
    canonical_calculator = canonical_calculator_for(meta.get("task_type", "unknown"))
    return f"""
Create exactly one supervised fine-tuning JSON object for Martha V6.1.

Return ONLY valid JSON, no markdown.

Required output schema:
{{
  "messages": [
    {{"role": "system", "content": "{SYSTEM}"}},
    {{"role": "user", "content": "a realistic user prompt"}},
    {{"role": "assistant", "content": "Martha consultant response"}}
  ],
    "metadata": {{
    "dataset": "v61_ollama_consultant",
    "category": "one of: calculator_plan, explain_method, interpret_result, refusal_or_clarification, model_diagnostic",
    "response_mode": "one of: plan_only, plan_plus_interpretation, refusal, diagnostic",
    "domain": "domain name",
    "task_type": "task name",
    "source_dataset": "source dataset name",
    "quality_contract": ["no invented numbers", "explicit assumptions", "calculator-first for exact arithmetic"]
  }}
}}

Assistant content requirements:
- Start with "Task classification:".
- Include "Known inputs:".
- Include "Missing or invalid inputs:".
- Include "Calculator plan:" followed by compact JSON with exactly these keys:
  {{"calculator": "specific_task_name_or_none", "inputs": {{}}, "expected_outputs": [], "requires_exact_calculator": true_or_false}}
- Use this exact calculator name unless refusing: "{canonical_calculator}".
- "calculator" must be a specific snake_case calculator name, not "calculate".
- "inputs" must contain the numeric/source inputs copied from the Source user prompt. Do not leave it empty when exact computation is needed.
- "expected_outputs" must be names like ["forecast", "loss", "gradient"], never numeric values.
- Include "Interpretation:".
- Use only the verified answer/result below for numeric claims.
- Put verified numeric results in Interpretation, not inside Calculator plan.
- Do not claim "statistically significant" unless p-value/alpha or a critical-value decision is explicitly present.
- If the source lacks required inputs, refuse with missing_info or clarification_needed.
- Do not add citations, external references, or fabricated data.

Source metadata:
domain={meta.get("domain", "Unknown")}
task_type={meta.get("task_type", "unknown")}
difficulty={meta.get("difficulty", "unknown")}
source_dataset={meta.get("dataset", "unknown")}
expected={json.dumps(expected, ensure_ascii=False)}

Source user prompt:
{source["source_user"]}

Verified source answer:
{source["source_assistant"]}
""".strip()


def build_repair_prompt(source: dict[str, Any], bad_raw: str, errors: list[str]) -> str:
    meta = source["source_metadata"]
    canonical_calculator = canonical_calculator_for(meta.get("task_type", "unknown"))
    return f"""
Repair this generated Martha V6.1 training row. Return ONLY one valid JSON object.

Validation errors:
{json.dumps(errors, ensure_ascii=False)}

Hard requirements:
- messages must have exactly system, user, assistant.
- assistant content must include these exact labels:
  Task classification:
  Known inputs:
  Missing or invalid inputs:
  Calculator plan:
  Interpretation:
- Calculator plan must be valid compact JSON with keys:
  calculator, inputs, expected_outputs, requires_exact_calculator
- Use calculator="{canonical_calculator}" unless this is a refusal/diagnostic-only row.
- inputs must copy known numeric/source inputs from the source prompt when requires_exact_calculator=true.
- expected_outputs must be string names from expected metadata, not numeric values.
- Do not claim statistical significance without p-value/alpha/critical decision context.

Source metadata:
domain={meta.get("domain", "Unknown")}
task_type={meta.get("task_type", "unknown")}
source_dataset={meta.get("dataset", "unknown")}
expected={json.dumps(meta.get("expected", {}), ensure_ascii=False)}

Source user prompt:
{source["source_user"]}

Verified source answer:
{source["source_assistant"]}

Bad generated row to repair:
{bad_raw[:3500]}
""".strip()


def canonical_calculator_for(task_type: str) -> str:
    normalized = str(task_type or "unknown").strip()
    if normalized in ALLOWED_CALCULATORS:
        return normalized
    if normalized in CALCULATOR_ALIASES:
        return CALCULATOR_ALIASES[normalized]
    if "weighted_moving_average" in normalized:
        return "weighted_moving_average"
    if "linear_trend" in normalized:
        return "linear_trend_forecast"
    if "classification_metrics" in normalized:
        return "classification_metrics"
    if "logistic" in normalized and "gradient" in normalized:
        return "logistic_regression_gradient"
    if "label_smoothing" in normalized:
        return "label_smoothing_cross_entropy"
    if "transformer" in normalized:
        return "transformer_attention_shapes"
    if "zscore" in normalized or "descriptive" in normalized:
        return "descriptive_statistics"
    if "semantic_search" in normalized:
        return "cosine_similarity"
    if "t_test" in normalized:
        return "t_test"
    if "hyperparameter" in normalized:
        return "none"
    return normalized


def ollama_chat(prompt: str, model: str, timeout: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate strict JSON training rows. Output only valid JSON. "
                    "Do not wrap in markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["message"]["content"]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("messages_must_have_3_items")
    else:
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            errors.append("roles_must_be_system_user_assistant")
        assistant = str(messages[2].get("content", ""))
        for marker in [
            "Task classification:",
            "Known inputs:",
            "Missing or invalid inputs:",
            "Calculator plan:",
        ]:
            if marker not in assistant:
                errors.append(f"missing_marker:{marker}")
        if not re.search(r"\bInterpretation\s*:", assistant, re.IGNORECASE):
            errors.append("missing_marker:Interpretation:")
        plan = extract_calculator_plan(assistant)
        if plan is None:
            errors.append("calculator_plan_not_valid_json")
        else:
            required_plan_keys = {"calculator", "inputs", "expected_outputs", "requires_exact_calculator"}
            missing = required_plan_keys - set(plan)
            if missing:
                errors.append("calculator_plan_missing_keys:" + ",".join(sorted(missing)))
            if not isinstance(plan.get("inputs"), dict):
                errors.append("calculator_plan_inputs_not_object")
            if (
                plan.get("calculator") != "none"
                and plan.get("requires_exact_calculator") is True
                and isinstance(plan.get("inputs"), dict)
                and not plan["inputs"]
            ):
                errors.append("calculator_plan_inputs_empty")
            if plan.get("calculator") == "none" and plan.get("requires_exact_calculator") is True:
                errors.append("calculator_none_requires_exact")
            if not isinstance(plan.get("expected_outputs"), list):
                errors.append("calculator_plan_expected_outputs_not_list")
            if not isinstance(plan.get("requires_exact_calculator"), bool):
                errors.append("calculator_plan_requires_exact_not_bool")
            if plan.get("calculator") == "calculate":
                errors.append("calculator_plan_too_generic")
            calculator = plan.get("calculator")
            if isinstance(calculator, str) and calculator in CALCULATOR_ALIASES:
                calculator = CALCULATOR_ALIASES[calculator]
                plan["calculator"] = calculator
                replace_calculator_plan(row, plan)
            if isinstance(calculator, str) and calculator not in ALLOWED_CALCULATORS:
                errors.append(f"calculator_not_allowed:{calculator}")
            expected_outputs = plan.get("expected_outputs")
            if isinstance(expected_outputs, list) and not all(isinstance(item, str) for item in expected_outputs):
                errors.append("calculator_plan_expected_outputs_must_be_strings")
            if "statistically significant" in assistant.lower():
                has_decision_context = bool(re.search(r"\bp\s*[-_ ]?value\b|\balpha\b|critical", assistant, re.IGNORECASE))
                if not has_decision_context:
                    errors.append("unsupported_significance_claim")
    if not isinstance(metadata, dict):
        errors.append("metadata_missing")
    else:
        if metadata.get("dataset") != "v61_ollama_consultant":
            errors.append("wrong_dataset_name")
        if metadata.get("category") not in {
            "calculator_plan",
            "explain_method",
            "interpret_result",
            "refusal_or_clarification",
            "model_diagnostic",
        }:
            errors.append("bad_category")
        if metadata.get("response_mode") not in {"plan_only", "plan_plus_interpretation", "refusal", "diagnostic"}:
            errors.append("bad_response_mode")
    return errors


def extract_calculator_plan(assistant: str) -> dict[str, Any] | None:
    marker = "Calculator plan:"
    if marker not in assistant:
        return None
    after = assistant.split(marker, 1)[1]
    before_interpretation = after.split("Interpretation:", 1)[0].strip()
    start = before_interpretation.find("{")
    end = before_interpretation.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        plan = json.loads(before_interpretation[start : end + 1])
    except json.JSONDecodeError:
        return None
    return plan if isinstance(plan, dict) else None


def source_expected_outputs(source: dict[str, Any]) -> list[str]:
    expected = source.get("source_metadata", {}).get("expected", {})
    if isinstance(expected, dict) and expected:
        return [str(key) for key in expected.keys()]
    return []


def repair_row(row: dict[str, Any], source: dict[str, Any]) -> None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        metadata["dataset"] = "v61_ollama_consultant"
        if metadata.get("response_mode") not in {"plan_only", "plan_plus_interpretation", "refusal", "diagnostic"}:
            metadata["response_mode"] = "plan_plus_interpretation"
        if metadata.get("category") not in {
            "calculator_plan",
            "explain_method",
            "interpret_result",
            "refusal_or_clarification",
            "model_diagnostic",
        }:
            metadata["category"] = "calculator_plan"

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return
    assistant = str(messages[2].get("content", ""))
    plan = extract_calculator_plan(assistant)
    if plan is None:
        return

    calculator = plan.get("calculator")
    if isinstance(calculator, str):
        plan["calculator"] = CALCULATOR_ALIASES.get(calculator, calculator)
    canonical = canonical_calculator_for(source.get("source_metadata", {}).get("task_type", "unknown"))
    if canonical in ALLOWED_CALCULATORS:
        plan["calculator"] = canonical

    expected_outputs = source_expected_outputs(source)
    if expected_outputs:
        plan["expected_outputs"] = expected_outputs
    metadata = row.get("metadata", {})
    is_refusal = isinstance(metadata, dict) and (
        metadata.get("category") == "refusal_or_clarification" or metadata.get("response_mode") == "refusal"
    )
    if plan.get("calculator") == "none" or is_refusal:
        plan["requires_exact_calculator"] = False

    replace_calculator_plan(row, plan)


def replace_calculator_plan(row: dict[str, Any], plan: dict[str, Any]) -> None:
    assistant = str(row["messages"][2].get("content", ""))
    marker = "Calculator plan:"
    if marker not in assistant or "Interpretation:" not in assistant:
        return
    before, rest = assistant.split(marker, 1)
    _old_plan, after = rest.split("Interpretation:", 1)
    row["messages"][2]["content"] = (
        before
        + marker
        + " "
        + json.dumps(plan, ensure_ascii=False)
        + " Interpretation:"
        + after
    )


def safe_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return cleaned.strip("_")


def tagged_path(directory: Path, stem: str, suffix: str, tag: str) -> Path:
    if not tag:
        return directory / f"{stem}{suffix}"
    return directory / f"{stem}_{tag}{suffix}"


def generate(
    limit: int,
    model: str,
    seed: int,
    timeout: int,
    max_per_file: int,
    repair_attempts: int,
    output_tag: str,
) -> dict[str, Any]:
    random.seed(seed)
    sources = load_sources(max_per_file=max_per_file)
    if not sources:
        raise RuntimeError("No source rows found.")
    random.shuffle(sources)
    selected = sources[:limit]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    repaired_count = 0
    started = time.time()
    for idx, source in enumerate(selected, 1):
        prompt = build_prompt(source)
        try:
            raw = ollama_chat(prompt, model=model, timeout=timeout)
            row = extract_json_object(raw)
            repair_row(row, source)
            errors = validate_row(row)
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            raw = locals().get("raw", "")
            row = {}
            errors = [f"exception:{type(exc).__name__}:{exc}"]

        raw_before_repair = str(raw)
        for _attempt in range(repair_attempts):
            if not errors:
                break
            try:
                raw = ollama_chat(build_repair_prompt(source, raw_before_repair, errors), model=model, timeout=timeout)
                row = extract_json_object(raw)
                repair_row(row, source)
                errors = validate_row(row)
                if not errors:
                    repaired_count += 1
                    break
            except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                errors = [f"repair_exception:{type(exc).__name__}:{exc}"]

        record = {
            "index": idx,
            "source_file": source["source_file"],
            "source_task_type": source["source_metadata"].get("task_type", "unknown"),
            "source_domain": source["source_metadata"].get("domain", "Unknown"),
            "errors": errors,
            "repair_attempts": repair_attempts,
            "raw_preview": str(raw)[:500],
        }
        if errors:
            rejected.append(record)
        else:
            row["metadata"]["source_file"] = source["source_file"]
            row["metadata"]["source_task_type"] = source["source_metadata"].get("task_type", "unknown")
            row["metadata"]["source_domain"] = source["source_metadata"].get("domain", "Unknown")
            accepted.append(row)
        print(f"[{idx}/{len(selected)}] accepted={len(accepted)} rejected={len(rejected)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    tag = safe_tag(output_tag)
    train_path = tagged_path(OUT_DIR, "v61_ollama_consultant_train_chat", ".jsonl", tag)
    reject_path = tagged_path(OUT_DIR, "v61_ollama_consultant_rejected", ".jsonl", tag)
    sample_path = tagged_path(SAMPLE_DIR, "v61_ollama_consultant_min_sample", ".jsonl", tag)
    report_path = tagged_path(REPORT_DIR, "v61_ollama_consultant_report", ".md", tag)
    manifest_path = tagged_path(REPORT_DIR, "v61_ollama_consultant_manifest", ".json", tag)

    write_jsonl(train_path, accepted)
    write_jsonl(reject_path, rejected)
    write_jsonl(sample_path, accepted[: min(5, len(accepted))])

    categories = Counter(row["metadata"].get("category", "unknown") for row in accepted)
    domains = Counter(row["metadata"].get("domain", "Unknown") for row in accepted)
    tasks = Counter(row["metadata"].get("task_type", "unknown") for row in accepted)
    manifest = {
        "model": model,
        "output_tag": tag,
        "limit_requested": limit,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "repaired": repaired_count,
        "acceptance_rate": round(len(accepted) / max(1, len(selected)), 4),
        "runtime_seconds": round(time.time() - started, 2),
        "train_path": str(train_path),
        "reject_path": str(reject_path),
        "sample_path": str(sample_path),
        "categories": dict(categories),
        "domains": dict(domains),
        "top_tasks": dict(tasks.most_common(20)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# V6.1 Ollama Consultant Dataset Report",
        "",
        f"- Local generator model: `{model}`",
        f"- Requested rows: {limit}",
        f"- Accepted rows: {len(accepted)}",
        f"- Rejected rows: {len(rejected)}",
        f"- Repaired rows: {repaired_count}",
        f"- Acceptance rate: {manifest['acceptance_rate']}",
        f"- Runtime seconds: {manifest['runtime_seconds']}",
        "",
        "## Design Goal",
        "",
        "Train standalone Martha behavior around task classification, missing-input detection, calculator planning, result interpretation, and ML/DL/stat diagnostics.",
        "",
        "## Categories",
        "",
    ]
    for category, count in categories.most_common():
        report_lines.append(f"- {category}: {count}")
    report_lines.extend(["", "## Domains", ""])
    for domain, count in domains.most_common():
        report_lines.append(f"- {domain}: {count}")
    report_lines.extend(["", "## Top Tasks", ""])
    for task, count in tasks.most_common(20):
        report_lines.append(f"- {task}: {count}")
    report_lines.extend(["", "## Quality Checks", ""])
    if rejected:
        report_lines.append("- Some rows were rejected. Inspect the rejected JSONL before scaling generation.")
    else:
        report_lines.append("- All generated rows passed schema and marker validation.")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-per-file", type=int, default=600)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    manifest = generate(
        limit=args.limit,
        model=args.model,
        seed=args.seed,
        timeout=args.timeout,
        max_per_file=args.max_per_file,
        repair_attempts=args.repair_attempts,
        output_tag=args.output_tag,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

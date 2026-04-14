"""Run local V6 hybrid benchmarks and write JSON/Markdown reports.

This is a local benchmark harness, not an official GSM8K/MATH/AIME runner.
It keeps public-style smoke tests separate from Timmy-specific domain and
guardrail tests so we can track improvements without leaking training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from einstein_v6_hybrid_assistant import (
    answer_v6_dl,
    answer_v6_guardrail,
    answer_v6_ml,
    answer_v6_stats_da_forecast,
)


BENCHMARK_VERSION = "v6_hybrid_local_benchmark_2026-04-12"
SOLVERS: list[Callable[[str], str | None]] = [
    answer_v6_guardrail,
    answer_v6_ml,
    answer_v6_dl,
    answer_v6_stats_da_forecast,
]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    track: str
    skill: str
    prompt: str
    expected_any: tuple[str, ...]
    expected_all: tuple[str, ...] = ()
    expected_status: str | None = None
    weight: float = 1.0


def run_v6_only(prompt: str) -> tuple[str | None, str | None]:
    for solver in SOLVERS:
        output = solver(prompt)
        if output:
            return solver.__name__, output
    return None, None


def normalize(text: str | None) -> str:
    return (text or "").lower().replace("\u2212", "-")


def score_case(case: BenchmarkCase, output: str | None) -> tuple[str, float, list[str]]:
    text = normalize(output)
    failures: list[str] = []

    if output is None:
        return "fail", 0.0, ["no_v6_route"]

    if case.expected_status:
        try:
            status = json.loads(output).get("status")
        except json.JSONDecodeError:
            failures.append("expected_json_status")
            status = None
        if status != case.expected_status:
            failures.append(f"status_expected_{case.expected_status}_got_{status}")

    for needle in case.expected_all:
        if needle.lower() not in text:
            failures.append(f"missing_all:{needle}")

    if case.expected_any and not any(needle.lower() in text for needle in case.expected_any):
        failures.append("missing_any:" + "|".join(case.expected_any))

    if not failures:
        return "pass", 1.0, []

    # Partial credit means the right route produced an answer but missed an
    # expected marker. This is useful for tracking parser/reporting regressions.
    return "partial", 0.5, failures


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            "public_smoke_math_001",
            "public_math_smoke",
            "competition_math_like",
            "A contest-style problem: if 3x + 7 = 22, solve for x.",
            ("x=5", "x = 5", "5"),
        ),
        BenchmarkCase(
            "public_smoke_math_002",
            "public_math_smoke",
            "arithmetic_word_problem_like",
            "A farmer has 12 boxes with 8 apples each and gives away 19 apples. How many apples remain?",
            ("77",),
        ),
        BenchmarkCase(
            "public_smoke_math_003",
            "public_math_smoke",
            "probability_like",
            "A fair die is rolled twice. What is the probability the sum is 7?",
            ("1/6", "0.1667"),
        ),
        BenchmarkCase(
            "public_smoke_math_004",
            "public_math_smoke",
            "geometry_like",
            "A circle has radius 3. Compute its area in terms of pi.",
            ("9pi", "9*pi", "28.274"),
        ),
        BenchmarkCase(
            "public_smoke_math_005",
            "public_math_smoke",
            "sequence_like",
            "The sequence is 2, 5, 8, 11. What is the 10th term?",
            ("29",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_001",
            "timmy_v6_domain",
            "decision_tree",
            "Decision tree split case: parent class counts=[9, 5], child counts=[[6, 1], [3, 4]]. Compute information gain and gini.",
            ("information_gain=0.1518",),
            ("parent_gini=0.4592",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_002",
            "timmy_v6_domain",
            "pca",
            "PCA covariance matrix=[[4, 1.2], [1.2, 2]]. Compute explained variance ratio.",
            ("76.0342", "0.7603"),
        ),
        BenchmarkCase(
            "timmy_v6_domain_003",
            "timmy_v6_domain",
            "label_smoothing_ce",
            "Label smoothing cross entropy: logits=[1.2, -0.4, 2.1, 0.3], true_class=2, epsilon=0.1. Compute loss and gradient.",
            ("loss=0.6332",),
            ("dL/dlogits",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_004",
            "timmy_v6_domain",
            "transformer_shapes",
            "Transformer attention shape check: batch=2, seq_len=16, d_model=64, heads=8. Compute QKV, score, and output shapes.",
            ("attention_output=[2, 16, 64]",),
            ("head_dim=8",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_005",
            "timmy_v6_domain",
            "standardization",
            "Data prep case: values=[10, 12, 13, 15, 20]. Compute sample mean, sample standard deviation, and z-scores.",
            ("sample_std=3.8079",),
            ("z_scores",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_006",
            "timmy_v6_domain",
            "mann_whitney",
            "Mann-Whitney test: sample_a=[4, 5, 6, 7], sample_b=[1, 2, 3, 8]. Compute U statistic.",
            ("Mann-Whitney U=4", "u=4"),
        ),
        BenchmarkCase(
            "timmy_v6_domain_007",
            "timmy_v6_domain",
            "multiple_testing",
            "Multiple testing correction: p_values=[0.001, 0.02, 0.04, 0.20], alpha=0.05. Compute Bonferroni and Benjamini-Hochberg decisions.",
            ("Bonferroni reject=[True, False, False, False]",),
            ("BH reject=[True, True, False, False]",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_008",
            "timmy_v6_domain",
            "forecast_diagnostics",
            "Forecast diagnostics: actuals=[100, 110, 105], forecasts=[98, 112, 108], train_history=[90, 95, 100, 104]. Compute sMAPE and MASE.",
            ("sMAPE=0.0221",),
            ("MASE=0.5",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_009",
            "timmy_v6_domain",
            "beta_posterior",
            "Beta posterior conversion model: successes=18, failures=82, prior=Beta(2, 2). Compute posterior mean.",
            ("posterior_mean=0.1923",),
        ),
        BenchmarkCase(
            "timmy_v6_domain_010",
            "timmy_v6_domain",
            "funnel_conversion",
            "Funnel analytics: stage_counts=[1000, 420, 210, 84]. Compute step rates and overall conversion.",
            ("overall_conversion=0.084",),
        ),
        BenchmarkCase(
            "guardrail_001",
            "guardrail",
            "negative_variance",
            "Portfolio variance is given as -0.02 with weights [0.5, 0.5]. Compute volatility.",
            ("variance cannot be negative",),
            expected_status="invalid_input",
        ),
        BenchmarkCase(
            "guardrail_002",
            "guardrail",
            "vague_cosine",
            "Compute cosine similarity between vector [1, 2, 3] and another vector roughly pointing northeast.",
            ("numeric vectors",),
            expected_status="missing_info",
        ),
        BenchmarkCase(
            "guardrail_003",
            "guardrail",
            "log_zero",
            "Compute log returns for price going from 100 to 0.",
            ("log(0)",),
            expected_status="invalid_input",
        ),
        BenchmarkCase(
            "guardrail_004",
            "guardrail",
            "stopping_distance_missing_road",
            "Estimate the stopping distance of a car moving at 20 m/s. Road conditions weren't clearly documented.",
            ("reaction time", "friction"),
            expected_status="missing_info",
        ),
        BenchmarkCase(
            "guardrail_005",
            "guardrail",
            "ambiguous_physics_forecast",
            "Velocity increases linearly over time like a trend model. Predict next velocity using moving average logic.",
            ("physics model", "forecasting model"),
            expected_status="clarification_needed",
        ),
        BenchmarkCase(
            "guardrail_006",
            "guardrail",
            "qualitative_sharpe",
            "Compute Sharpe ratio, but risk-free rate was low and volatility was moderate.",
            ("numeric return", "numeric risk-free rate"),
            expected_status="missing_info",
        ),
    ]


def write_svg_chart(summary_by_track: dict[str, dict[str, float]], path: Path) -> None:
    tracks = list(summary_by_track)
    width = 920
    height = 110 + 54 * len(tracks)
    left = 210
    bar_width = 560
    rows = []
    for idx, track in enumerate(tracks):
        y = 80 + idx * 54
        score = summary_by_track[track]["score_pct"]
        fill = "#1f9d55" if score >= 85 else "#d97706" if score >= 60 else "#dc2626"
        rows.append(
            f'<text x="24" y="{y + 16}" class="label">{track}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="22" rx="11" class="bg"/>'
            f'<rect x="{left}" y="{y}" width="{bar_width * score / 100:.1f}" height="22" rx="11" fill="{fill}"/>'
            f'<text x="{left + bar_width + 18}" y="{y + 16}" class="score">{score:.1f}%</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Georgia, serif; fill: #172033; }}
    .subtitle {{ font: 14px Verdana, sans-serif; fill: #526070; }}
    .label {{ font: 14px Verdana, sans-serif; fill: #172033; }}
    .score {{ font: 700 14px Verdana, sans-serif; fill: #172033; }}
    .bg {{ fill: #e8edf3; }}
  </style>
  <rect width="100%" height="100%" rx="24" fill="#f8fafc"/>
  <text x="24" y="36" class="title">Gemma TIMMY V6 Hybrid Benchmark</text>
  <text x="24" y="58" class="subtitle">Local eval-only benchmark: public-style smoke, domain calculators, and safety guardrails.</text>
  {''.join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def run(out_dir: Path) -> dict:
    cases = benchmark_cases()
    out_dir.mkdir(parents=True, exist_ok=True)
    details = []
    by_track: dict[str, Counter] = defaultdict(Counter)
    weighted_scores: dict[str, float] = defaultdict(float)
    weighted_totals: dict[str, float] = defaultdict(float)

    for case in cases:
        solver, output = run_v6_only(case.prompt)
        verdict, score, failures = score_case(case, output)
        by_track[case.track][verdict] += 1
        weighted_scores[case.track] += score * case.weight
        weighted_totals[case.track] += case.weight
        details.append(
            {
                "case_id": case.case_id,
                "track": case.track,
                "skill": case.skill,
                "prompt": case.prompt,
                "solver": solver,
                "verdict": verdict,
                "score": score,
                "failures": failures,
                "output": output,
            }
        )

    summary_by_track = {}
    total_score = 0.0
    total_weight = 0.0
    for track, counts in sorted(by_track.items()):
        score_pct = 100 * weighted_scores[track] / weighted_totals[track]
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
        "cases": len(cases),
        "score_pct": round(100 * total_score / total_weight, 2),
        "pass": sum(1 for d in details if d["verdict"] == "pass"),
        "partial": sum(1 for d in details if d["verdict"] == "partial"),
        "fail": sum(1 for d in details if d["verdict"] == "fail"),
    }
    result = {
        "benchmark_version": BENCHMARK_VERSION,
        "engine": "v6_hybrid_only_no_v52_fallback",
        "official_public_benchmark": False,
        "note": "Public-style cases are local smoke tests, not official GSM8K/MATH/AIME scores.",
        "overall": overall,
        "summary_by_track": summary_by_track,
        "cases": details,
    }

    json_path = out_dir / "v6_hybrid_benchmark_results.json"
    md_path = out_dir / "v6_hybrid_benchmark_report.md"
    svg_path = out_dir / "v6_hybrid_benchmark_chart.svg"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_svg_chart(summary_by_track, svg_path)

    lines = [
        "# Gemma TIMMY V6 Hybrid Benchmark Report",
        "",
        f"- Benchmark version: `{BENCHMARK_VERSION}`",
        "- Engine: `v6_hybrid_only_no_v52_fallback`",
        "- Official public benchmark: `false`",
        "- Note: public-style cases are local smoke tests, not official GSM8K/MATH/AIME scores.",
        "",
        "## Overall",
        "",
        f"- Cases: {overall['cases']}",
        f"- Pass / Partial / Fail: {overall['pass']} / {overall['partial']} / {overall['fail']}",
        f"- Weighted score: {overall['score_pct']}%",
        "",
        "## Track Scores",
        "",
        "| Track | Pass | Partial | Fail | Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for track, row in summary_by_track.items():
        lines.append(f"| {track} | {row['pass']} | {row['partial']} | {row['fail']} | {row['score_pct']}% |")
    lines.extend(["", "## Failures And Partials", ""])
    weak = [d for d in details if d["verdict"] != "pass"]
    if not weak:
        lines.append("- None.")
    else:
        for item in weak:
            lines.append(f"- `{item['case_id']}` `{item['track']}` `{item['skill']}`: {item['verdict']} ({', '.join(item['failures'])})")
    lines.extend(["", f"![Benchmark chart](v6_hybrid_benchmark_chart.svg)", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/v6/benchmarks")
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    print(json.dumps({"overall": result["overall"], "summary_by_track": result["summary_by_track"]}, indent=2))


if __name__ == "__main__":
    main()
